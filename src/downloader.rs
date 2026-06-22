use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::Which;

/// Resolve the GGUF model file path, downloading from HuggingFace if not supplied.
pub fn resolve_model(model_arg: Option<String>, which: Which) -> anyhow::Result<PathBuf> {
    if let Some(path) = model_arg {
        return Ok(PathBuf::from(path));
    }

    let (repo, filename) = match which {
        Which::Gemma3_4bIt => (
            "google/gemma-3-4b-it-qat-q4_0-gguf",
            "gemma-3-4b-it-q4_0.gguf",
        ),
        Which::Gemma4E4bIt => (
            "lmstudio-community/gemma-4-E4B-it-GGUF",
            "gemma-4-E4B-it-Q4_K_M.gguf",
        ),
    };

    println!("Downloading model {filename} from {repo}");
    let api = hf_hub::api::sync::Api::new()?;
    let path = api
        .repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ))
        .get(filename)?;
    Ok(path)
}

/// Resolve the tokenizer, downloading from HuggingFace if not supplied.
pub fn resolve_tokenizer(tokenizer_arg: Option<String>, which: Which) -> anyhow::Result<Tokenizer> {
    let path = match tokenizer_arg {
        Some(p) => PathBuf::from(p),
        None => {
            let repo = match which {
                Which::Gemma3_4bIt => "google/gemma-3-4b-it",
                Which::Gemma4E4bIt => "google/gemma-4-E4B-it",
            };
            println!("Downloading tokenizer from {repo}");
            let api = hf_hub::api::sync::Api::new()?;
            api.model(repo.to_string()).get("tokenizer.json")?
        }
    };
    println!("Loading tokenizer from {path:?}");
    Tokenizer::from_file(path).map_err(anyhow::Error::msg)
}
