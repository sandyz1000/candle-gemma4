mod downloader;
mod quantized_gemma4;
mod token_output_stream;

use candle::{utils::{cuda_is_available, metal_is_available}, Device};
use clap::{Parser, ValueEnum};
use std::io::Write;

use candle::quantized::gguf_file;
use candle::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma3Weights;

use crate::quantized_gemma4::ModelWeights as Gemma4Weights;
use crate::token_output_stream::TokenOutputStream;

// ── Memory helpers ────────────────────────────────────────────────────────────

/// Returns the current process RSS in MiB. Returns 0.0 if unavailable.
fn resident_memory_mib() -> f64 {
    let pid = std::process::id();

    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(kb) = s.trim().parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb.parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
    }

    let _ = pid;
    0.0
}

fn log_mem(label: &str) {
    let mib = resident_memory_mib();
    if mib > 0.0 {
        eprintln!("[mem] {label}: {mib:.1} MiB RSS");
    }
}

pub fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        println!("Running on CPU, to run on GPU(metal), build with `--features metal`");
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        println!("Running on CPU, to run on GPU, build with `--features cuda`");
        Ok(Device::Cpu)
    }
}

// ── Which model ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
pub enum Which {
    #[value(name = "gemma3-4b-it")]
    Gemma3_4bIt,
    #[value(name = "gemma4-e4b-it")]
    Gemma4E4bIt,
}

// ── Model wrapper so both variants share one forward() call ───────────────────

enum AnyModel {
    Gemma3(Gemma3Weights),
    Gemma4(Gemma4Weights),
}

impl AnyModel {
    fn forward(&mut self, x: &Tensor, pos: usize) -> candle::Result<Tensor> {
        match self {
            Self::Gemma3(m) => m.forward(x, pos),
            Self::Gemma4(m) => m.forward(x, pos),
        }
    }
}

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a local GGUF file (skips HuggingFace download).
    #[arg(long)]
    model: Option<String>,

    /// Prompt text. Use "interactive" or "chat" for multi-turn modes.
    #[arg(long)]
    prompt: Option<String>,

    /// Number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    /// Path to a local tokenizer.json (skips HuggingFace download).
    #[arg(long)]
    tokenizer: Option<String>,

    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-K sampling.
    #[arg(long)]
    top_k: Option<usize>,

    /// RNG seed.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (writes trace-timestamp.json).
    #[arg(long)]
    tracing: bool,

    /// Feed prompt tokens one at a time instead of in a single batch.
    #[arg(long)]
    split_prompt: bool,

    /// Force CPU even when a GPU is available.
    #[arg(long)]
    cpu: bool,

    /// Repetition penalty (1.0 = disabled).
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Number of trailing tokens considered for repetition penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Which model to use.
    #[arg(long, default_value = "gemma4-e4b-it")]
    which: Which,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn format_size(bytes: usize) -> String {
    if bytes < 1_000 {
        format!("{bytes}B")
    } else if bytes < 1_000_000 {
        format!("{:.2}KB", bytes as f64 / 1e3)
    } else if bytes < 1_000_000_000 {
        format!("{:.2}MB", bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", bytes as f64 / 1e9)
    }
}

#[derive(Debug)]
enum Prompt {
    Interactive,
    Chat,
    One(String),
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let device = device(args.cpu)?;
    let model_path = downloader::resolve_model(args.model.clone(), args.which)?;

    let mut file = std::fs::File::open(&model_path)
        .map_err(|e| anyhow::anyhow!("cannot open model '{}': {e}", model_path.display()))?;
    let start = std::time::Instant::now();

    let mut model = {
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
        let mut total_bytes = 0usize;
        for (_, ti) in ct.tensor_infos.iter() {
            total_bytes +=
                ti.shape.elem_count() * ti.ggml_dtype.type_size() / ti.ggml_dtype.block_size();
        }
        println!(
            "loaded {} tensors ({}) in {:.2}s",
            ct.tensor_infos.len(),
            format_size(total_bytes),
            start.elapsed().as_secs_f32(),
        );
        match args.which {
            Which::Gemma3_4bIt => {
                AnyModel::Gemma3(Gemma3Weights::from_gguf(ct, &mut file, &device)?)
            }
            Which::Gemma4E4bIt => {
                AnyModel::Gemma4(Gemma4Weights::from_gguf(ct, &mut file, &device)?)
            }
        }
    };
    println!("model built");
    log_mem("after model load");

    let tokenizer = downloader::resolve_tokenizer(args.tokenizer.clone(), args.which)?;
    let mut tos = TokenOutputStream::new(tokenizer);
    println!("vocab size: {}", tos.tokenizer().get_vocab(true).len());

    // Default: run in chat mode (continuous interaction).
    // Pass --prompt <text> for a single one-shot generation.
    let prompt = match args.prompt.as_deref() {
        Some("chat") => Prompt::Chat,
        Some("interactive") => Prompt::Interactive,
        Some(s) => Prompt::One(s.to_string()),
        None => Prompt::Chat,
    };

    // Context window: 128K for Gemma 4, 8K for Gemma 3
    let max_seq_len = match args.which {
        Which::Gemma4E4bIt => 131072,
        Which::Gemma3_4bIt => 8192,
    };

    let mut pre_prompt_tokens: Vec<u32> = vec![];
    loop {
        let prompt_str = match &prompt {
            Prompt::One(s) => s.clone(),
            Prompt::Interactive | Prompt::Chat => {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                std::io::stdin().read_line(&mut line)?;
                let line = line.trim_end_matches(['\n', '\r']).to_string();
                // Gemma 4 chat format uses <|turn> / <turn|> turn delimiters.
                format!("<|turn>user\n{line}<turn|>\n<|turn>model\n")
            }
        };
        print!("{prompt_str}");

        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        // Gemma requires a leading <bos> (id 2); tokenizer.json does not add it, and the
        // model behaves erratically without it. Prepend it at the start of the sequence.
        let mut prompt_tokens = pre_prompt_tokens.clone();
        if prompt_tokens.is_empty() {
            prompt_tokens.push(2);
        }
        prompt_tokens.extend_from_slice(tokens.get_ids());
        let prompt_tokens = prompt_tokens;

        let to_sample = args.sample_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > max_seq_len - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - max_seq_len;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };

        let mut all_tokens: Vec<u32> = vec![];
        let mut logits_processor = {
            let sampling = if args.temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (args.top_k, args.top_p) {
                    (None, None) => Sampling::All { temperature: args.temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature: args.temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature: args.temperature },
                    (Some(k), Some(p)) => {
                        Sampling::TopKThenTopP { k, p, temperature: args.temperature }
                    }
                }
            };
            LogitsProcessor::from_sampling(args.seed, sampling)
        };

        let t0 = std::time::Instant::now();
        let mut next_token = if !args.split_prompt {
            let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            logits_processor.sample(&logits.squeeze(0)?)?
        } else {
            let mut tok = 0u32;
            for (pos, &t) in prompt_tokens.iter().enumerate() {
                let input = Tensor::new(&[t], &device)?.unsqueeze(0)?;
                let logits = model.forward(&input, pos)?;
                tok = logits_processor.sample(&logits.squeeze(0)?)?;
            }
            tok
        };
        let prompt_dt = t0.elapsed();

        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        // Gemma chat models stop on <end_of_turn>; fall back to <eos>. Use whichever
        // the tokenizer actually exposes (these are added/special tokens).
        let vocab = tos.tokenizer().get_vocab(true);
        let eos_token = vocab
            .get("<end_of_turn>")
            .or_else(|| vocab.get("<eos>"))
            .copied()
            .unwrap_or(u32::MAX);

        let t1 = std::time::Instant::now();
        let mut sampled = 0usize;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if args.repeat_penalty == 1.0 {
                logits
            } else {
                let start = all_tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &all_tokens[start..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            }
        }

        if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        let gen_dt = t1.elapsed();

        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / gen_dt.as_secs_f64(),
        );
        log_mem("after generation");

        match prompt {
            Prompt::One(_) => break,
            Prompt::Interactive => {}
            Prompt::Chat => {
                pre_prompt_tokens =
                    [prompt_tokens.as_slice(), all_tokens.as_slice()].concat();
            }
        }
    }

    Ok(())
}
