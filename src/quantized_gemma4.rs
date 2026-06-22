//! Gemma 4 (E2B/E4B) GGUF quantized text model.
//!
//! Ported from llama.cpp's `src/models/gemma4.cpp` graph (the reference that produces
//! correct output for these GGUF weights). Gemma4-specific features vs gemma3:
//! - GGUF architecture prefix is "gemma4".
//! - Per-layer head_dim: global (full-attention) layers use `attention.key_length` (512),
//!   sliding-window layers use `attention.key_length_swa` (256). KV heads = 2 for both.
//! - Per-layer RoPE base: global uses `rope.freq_base` (1e6) with PARTIAL rotary
//!   (factor 0.25 — only the first 25% of head_dim is rotated, per the `rope_freqs.weight`
//!   divisor table); sliding uses `rope.freq_base_swa` (1e4) with FULL rotary.
//! - V is normalised with pure RMS (no learned weight) in every layer.
//! - GeGLU MLP uses gelu_pytorch_tanh (NOT silu).
//! - Per-Layer Embeddings (PLE): `per_layer_token_embd` (token identity) + `per_layer_model_proj`
//!   (context projection) combine into a per-token/per-layer signal that modulates each layer
//!   via `inp_gate` -> gelu -> *per_layer_input -> `proj` -> `post_norm`, added as a residual.
//! - Each layer output is scaled by a learned per-layer scalar (`layer_output_scale`).
//! - `final_logit_softcapping` (30) is applied to the output logits.
//!
//! KNOWN ISSUE: output is not yet coherent. LM Studio (llama.cpp) produces correct output from
//! the same GGUF, so the weights are good; every component here has been checked against the
//! llama.cpp graph, but a subtle numerical/architectural discrepancy remains under investigation.

use candle::D;
use candle::quantized::QTensor;
use candle::quantized::gguf_file;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 131072; // 128K context window
pub const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
// Global (full-attention) layers rotate only the first 25% of head_dim. Confirmed by the
// GGUF `rope_freqs.weight` tensor: 64 of its 256 entries are 1.0 (kept) and the rest are
// 1e30 (frequency divided to ~0 → identity), i.e. 64/256 = 0.25. Sliding layers use full rotary.
pub const PARTIAL_ROTARY_FACTOR: f64 = 0.25;

// ── QMatMul wrapper ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ── RmsNorm (learned weight; GGUF folds Gemma's +1 offset into the weight) ────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(qt: QTensor, eps: f64) -> Result<Self> {
        let weight = qt.dequantize(&qt.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let hidden = x_f32.dim(D::Minus1)?;
        let norm = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let x_normed = x_f32.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight.to_dtype(x_dtype)?)
    }
}

/// Pure-RMS normalisation without a learned weight (used for V in every gemma4 layer).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let orig = v.dtype();
    let v32 = v.to_dtype(DType::F32)?;
    let mean_sq = v32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v32.broadcast_div(&rms)?.to_dtype(orig)
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Gemma uses GeGLU with gelu_pytorch_tanh (candle's `gelu` is the tanh approx), NOT silu.
        let gate = self.feed_forward_gate.forward(xs)?.gelu()?;
        let up = self.feed_forward_up.forward(xs)?;
        self.feed_forward_down.forward(&(gate * up)?)
    }
}

// ── RotaryEmbedding (full rotary; built per layer with that layer's head_dim/theta) ──

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// Full rotary over all `head_dim` dimensions (used for sliding-window layers).
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        Self::new_with_factor(head_dim, rope_frequency, 1.0, device)
    }

    /// Partial rotary: only the first `partial_factor * head_dim` dimensions are rotated,
    /// the rest are identity (cos=1, sin=0). Used for global/full-attention layers.
    fn new_partial(
        head_dim: usize,
        rope_frequency: f32,
        partial_factor: f64,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_factor(head_dim, rope_frequency, partial_factor, device)
    }

    fn new_with_factor(
        head_dim: usize,
        rope_frequency: f32,
        partial_factor: f64,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let rope_angles = (partial_factor * head_dim as f64 / 2.0) as usize;
        let mut theta: Vec<f32> = (0..rope_angles)
            .map(|i| 1f32 / rope_frequency.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        // Zero-pad the remaining frequencies → cos=1, sin=0 → identity rotation there.
        theta.extend(std::iter::repeat(0f32).take(half_dim - rope_angles));
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            cos: idx_theta.cos()?,
            sin: idx_theta.sin()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── LayerWeights ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,

    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,

    mlp: Mlp,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,
    rms_norm_eps: f64,

    sliding_window_size: Option<usize>,

    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,

    // Per-Layer Embedding (PLE) block, applied after attention + MLP.
    per_layer_input_gate: QMatMul, // inp_gate: Linear(hidden -> ple_dim)
    per_layer_projection: QMatMul, // proj:    Linear(ple_dim -> hidden)
    post_per_layer_input_norm: RmsNorm, // post_norm
    layer_scalar: Tensor,          // layer_output_scale: [1]

    kv_cache: Option<(Tensor, Tensor)>,

    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // Q/K learned RMSNorm.
        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;
        // V pure-RMS norm (no learned weight) — gemma4 applies this in every layer.
        let v = v_norm(&v, self.rms_norm_eps)?;

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA repeat.
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled dot-product attention (per-layer 1/sqrt(head_dim)).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    // GQA repeat must BLOCK each KV head into n_rep adjacent copies
    // ([kv0,kv0,..,kv1,kv1,..]) so query-head group g maps to kv head g.
    // `cat([x;n_rep], dim=1)` would interleave ([kv0,kv1,kv0,kv1,..]) — wrong.
    let (b, n_kv, seq, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, head_dim))?
        .reshape((b, n_kv * n_rep, seq, head_dim))
}

// ── ModelWeights ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,

    // Per-Layer Embeddings (PLE) — gemma4 E-series core feature.
    per_layer_token_embd: Embedding, // [vocab, num_layers * ple_dim], scaled by sqrt(ple_dim)
    per_layer_model_proj: QMatMul,   // Linear(hidden -> num_layers * ple_dim)
    per_layer_proj_norm: RmsNorm,    // RMSNorm over ple_dim
    num_layers: usize,
    ple_dim: usize,             // hidden_size_per_layer_input (256)
    per_layer_embed_scale: f64, // sqrt(ple_dim)
    per_layer_proj_scale: f64,  // 1/sqrt(hidden_size)
    per_layer_input_scale: f64, // 1/sqrt(2)
    final_logit_softcapping: Option<f64>,

    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Detect architecture prefix — gemma4 first, then fall back to older variants.
        let prefix = ["gemma4", "gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma4");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        // Per-Layer Embedding (PLE) dimension (hidden_size_per_layer_input, 256).
        let ple_dim = md_get("embedding_length_per_layer_input")?.to_u32()? as usize;
        let final_logit_softcapping = md_get("final_logit_softcapping")
            .ok()
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);

        // Per-layer-type head dims. Global (full-attention) layers use `key_length`;
        // sliding-window layers use `key_length_swa` (falls back to key_length if absent).
        let global_head_dim = md_get("attention.key_length")?.to_u32()? as usize;
        let sliding_head_dim = md_get("attention.key_length_swa")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(global_head_dim);

        // Per-layer-type RoPE base. Global uses freq_base, sliding uses freq_base_swa.
        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_sliding = md_get("rope.freq_base_swa")
            .or_else(|_| md_get("rope.local_freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        // Sliding pattern: gemma4 carries an explicit bool array
        // (`attention.sliding_window_pattern`, true = sliding/local, false = global).
        // gemma3 carried an int (`sliding_window_type`). Resolve into a closure.
        let sliding_pattern: Option<Vec<bool>> = md_get("attention.sliding_window_pattern")
            .ok()
            .and_then(|v| v.to_vec().ok())
            .and_then(|arr| arr.iter().map(|x| x.to_bool().ok()).collect());
        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN);
        let is_sliding = |layer_idx: usize| -> bool {
            match &sliding_pattern {
                Some(p) if layer_idx < p.len() => p[layer_idx],
                // Default gemma pattern: every `sliding_window_type`-th layer is global.
                _ => (layer_idx + 1) % sliding_window_type != 0,
            }
        };

        println!(
            "arch={prefix} layers={block_count} heads={head_count} kv={head_count_kv} \
             sliding_head_dim={sliding_head_dim} global_head_dim={global_head_dim} \
             window={sliding_window_size} \
             rope_global={rope_freq_base} rope_sliding={rope_freq_base_sliding}"
        );

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?, // tied weights
        };

        // ── Per-Layer Embeddings (PLE) ───────────────────────────────────────
        // The token-identity table is huge (vocab × num_layers × ple_dim, ~2.8B params),
        // so dequantize it to F16 (≈ half the memory of F32) rather than F32.
        let per_layer_token_embd = ct
            .tensor(reader, "per_layer_token_embd.weight", device)?
            .dequantize_f16(device)?;
        let per_layer_token_embd = Embedding::new(per_layer_token_embd, block_count * ple_dim);
        let per_layer_model_proj =
            QMatMul::from_qtensor(ct.tensor(reader, "per_layer_model_proj.weight", device)?)?;
        let per_layer_proj_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
            rms_norm_eps,
        )?;

        // Two shared RoPE tables; cheap to clone the handles per layer.
        // Global layers use partial rotary (first 25% of head_dim); sliding layers full rotary.
        let rope_global = RotaryEmbedding::new_partial(
            global_head_dim,
            rope_freq_base,
            PARTIAL_ROTARY_FACTOR,
            device,
        )?;
        let rope_sliding = RotaryEmbedding::new(sliding_head_dim, rope_freq_base_sliding, device)?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let p = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{p}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_gate.weight"),
                    device,
                )?)?,
                feed_forward_up: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_up.weight"),
                    device,
                )?)?,
                feed_forward_down: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_down.weight"),
                    device,
                )?)?,
            };

            // Per-Layer Embedding (PLE) submodules.
            let per_layer_input_gate = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.inp_gate.weight"),
                device,
            )?)?;
            let per_layer_projection =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.proj.weight"), device)?)?;
            let post_per_layer_input_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let layer_scalar = ct
                .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)?
                .dequantize(device)?;

            let layer_is_sliding = is_sliding(layer_idx);
            let (head_dim, rotary_embedding) = if layer_is_sliding {
                (sliding_head_dim, rope_sliding.clone())
            } else {
                (global_head_dim, rope_global.clone())
            };
            let sliding_window_size = layer_is_sliding.then_some(sliding_window_size);

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                q_dim: head_count * head_dim,
                rms_norm_eps,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                layer_scalar,
                kv_cache: None,
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            num_layers: block_count,
            ple_dim,
            per_layer_embed_scale: (ple_dim as f64).sqrt(),
            per_layer_proj_scale: 1.0 / (embedding_length as f64).sqrt(),
            per_layer_input_scale: 1.0 / 2f64.sqrt(),
            final_logit_softcapping,
            span,
            span_output,
        })
    }

    /// Compute the per-layer input tensors [b, seq, num_layers, ple_dim] from the PLE tables.
    ///   token_identity   = per_layer_token_embd(input_ids) * sqrt(ple_dim)
    ///   context_aware    = rmsnorm(reshape(per_layer_model_proj(embeds) / sqrt(hidden)))
    ///   per_layer_inputs = (context_aware + token_identity) * (1/sqrt(2))
    fn per_layer_inputs(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;

        // The flat (num_layers * ple_dim) vector is laid out layer-major (ple_dim innermost),
        // matching llama.cpp's reshape_3d(.., n_embd_per_layer, n_layer, ..).
        let reshape_ple =
            |t: &Tensor| t.reshape((b_sz, seq_len, self.num_layers, self.ple_dim));

        // Token-identity component (lookup table is F16; bring it to the compute dtype).
        let token_identity = self
            .per_layer_token_embd
            .forward(input_ids)?
            .to_dtype(inputs_embeds.dtype())?;
        let token_identity = (reshape_ple(&token_identity)? * self.per_layer_embed_scale)?;

        // Context-aware projection of the (scaled) main embeddings.
        let context =
            (self.per_layer_model_proj.forward(inputs_embeds)? * self.per_layer_proj_scale)?;
        let context = reshape_ple(&context)?;
        let context = self.per_layer_proj_norm.forward(&context)?;

        (context + token_identity)? * self.per_layer_input_scale
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let _enter = self.span.enter();

        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        // Per-layer inputs: [b, seq, num_layers, ple_dim].
        let per_layer_inputs = self.per_layer_inputs(x, &layer_in)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            // Attention block.
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block.
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let y = layer.ffn_norm.forward(&x)?;
            let y = layer.mlp.forward(&y)?;
            let y = layer.post_ffn_norm.forward(&y)?;
            let x = (y + residual)?;
            drop(_enter);

            // Per-Layer Embedding block:
            //   h = residual + post_norm(proj(gelu(gate(h)) * per_layer_input))
            let per_layer_input = per_layer_inputs.narrow(2, layer_idx, 1)?.squeeze(2)?; // [b, seq, ple_dim]
            let residual = &x;
            let g = layer.per_layer_input_gate.forward(&x)?.gelu()?;
            let g = g.broadcast_mul(&per_layer_input)?;
            let g = layer.per_layer_projection.forward(&g)?;
            let g = layer.post_per_layer_input_norm.forward(&g)?;
            let x = (residual + g)?;
            // Each layer output is scaled by a learned per-layer scalar.
            layer_in = x.broadcast_mul(&layer.layer_scalar)?;
        }

        let _enter = self.span_output.enter();
        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let logits = self.output.forward(&x)?;
        match self.final_logit_softcapping {
            Some(sc) => (logits / sc)?.tanh()? * sc,
            None => Ok(logits),
        }
    }
}
