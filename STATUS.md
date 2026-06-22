# candle-gemma4q — status

Goal: run `lmstudio-community/gemma-4-E4B-it-GGUF` (arch `gemma4`) in candle for interactive chat.

## TL;DR
The model **loads and runs end-to-end** with the correct gemma4 architecture, but generated
text is **not yet coherent**. The GGUF itself is good — **LM Studio (llama.cpp) produces correct
output from the exact same file** (`lms chat google/gemma-4-e4b -p "..."` → "Paris"). So the bug
is in this candle port, not the weights. Every component has been checked against llama.cpp's
`src/models/gemma4.cpp` graph and matches, yet a subtle discrepancy remains.

## Architecture implemented (`src/quantized_gemma4.rs`)
Ported from llama.cpp `gemma4.cpp` (the working reference). gemma4 vs gemma3 deltas:
- Per-layer head_dim: **global**=`attention.key_length`=512, **sliding**=`key_length_swa`=256; kv heads=2 both.
- Per-layer RoPE: global `rope.freq_base`=1e6 **partial rotary 0.25** (only first 25% of head_dim rotated —
  confirmed by `rope_freqs.weight`: 64 of 256 entries are 1.0, rest 1e30); sliding `freq_base_swa`=1e4 **full**.
- **V-norm**: pure RMS (no weight) on V every layer.
- **GeGLU MLP** uses gelu_pytorch_tanh (NOT silu — gemma3's candle impl wrongly uses silu).
- **Per-Layer Embeddings (PLE)** — gemma4 E-series core feature, ~2.8B of the params:
  - `per_layer_token_embd` [vocab, 42*256] (Q6K, dequantized to **F16** to fit memory) → token identity × sqrt(256)
  - `per_layer_model_proj` [10752, 2560] → context proj × 1/sqrt(2560), reshape, `per_layer_proj_norm` (RMS)
  - combined: `(context + identity) * 1/sqrt(2)`, shape [b, seq, 42, 256]
  - per layer: `h += post_norm(proj(gelu(inp_gate(h)) * per_layer_input[i]))`
- Per-layer output scalar `layer_output_scale` multiplies each layer's output.
- `final_logit_softcapping`=30 applied to logits.

## Bugs found & fixed
- **`candle` dep**: was the dead `candle="0.1.0"` crate; aliased `candle = { package="candle-core" }`.
- **candle-nn `metal` feature** missing → "no metal implementation for rotary-emb". Added to Cargo.toml.
- **GQA `repeat_kv`**: was `cat([x;n_rep])` which *interleaves* kv heads ([kv0,kv1,kv0,kv1]); fixed to
  unsqueeze/expand/reshape *blocking* ([kv0,kv0,kv1,kv1]) so query group g → kv head g. (Real bug.)
- **Chat format**: gemma4 uses `<|turn>` (105) / `<turn|>` (106), not `<start_of_turn>`/`<end_of_turn>`.
  Updated in `src/main.rs`.
- **BOS**: tokenizer.json does NOT add `<bos>` (id 2); now prepended manually (GGUF `add_bos_token=true`).
- Earlier head_dim swap (had global/sliding reversed), wrong metadata keys, eos lookup, etc.

## Verified matching llama.cpp / HF (so NOT the bug)
- Layer order (input_norm→attn→post_attn_norm→res; ffn_norm→mlp→post_ffn_norm→res; PLE; *scalar). Verbatim match.
- PLE `build_inp_per_layer` / `project_per_layer_inputs` — verbatim match incl. scales & layer-major reshape.
- Attention: q/k/v proj→reshape→q_norm/k_norm/v_norm→rope→GQA→sdpa(1/sqrt(head_dim))→o_proj. Match.
- RMSNorm = plain `x/sqrt(mean(x²)+eps)*weight` (no +1; GGUF norm weights are large ~10, F32). Match.
- candle `rope` == HF rotate_half (verified `rope_slow` source). Partial rotary == llama.cpp freq_factors.
- Quantization is accurate: F32-dequant output projection gives identical logits to the Q6K matmul.
- Hidden-state RMS stays healthy (~1–2, no NaN/blowup) across all 42 layers.

## Symptom / remaining suspects
Output is grammatically-plausible but factually wrong / degrades to garbage. Top-5 for raw
"The capital of France is" → `" $\"`, `" stable"`, `" evenly"` (target `" France"`/Paris not present).
Q4_K vs F32 layer matmuls give *completely different* wrong tokens → model sits in an unstable
numeric regime → suspect a **scaling/normalization discrepancy** somewhere subtle. Toggle bisection
(v_norm, layer_scalar, attn scale, global rope, PLE reshape) gave noisy results, none coherent.

Likely next step: **layer-by-layer numerical diff vs llama.cpp** (dump hidden state after layer 0/1
for a fixed token from a llama.cpp debug build and compare) to pinpoint the diverging op.

## Reference / tooling
- Working reference: `~/.lmstudio/bin/lms` (LM Studio = llama.cpp). `lms server start` then
  `curl localhost:1234/v1/completions` for logits/top-token ground truth.
- llama.cpp graph: `github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp`.
- HF PyTorch: `transformers/src/transformers/models/gemma4/modeling_gemma4.py`.
- `examples/inspect.rs` — scratch GGUF inspector (metadata, tensor shapes/dtypes, tokenizer checks).
- Model/tokenizer: `models/gemma-4-E4B-it-Q4_K_M.gguf`, `models/tokenizer.json`.
