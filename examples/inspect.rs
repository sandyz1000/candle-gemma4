use tokenizers::Tokenizer;
fn main() -> anyhow::Result<()> {
    let tk = Tokenizer::from_file("models/tokenizer.json").map_err(anyhow::Error::msg)?;
    for s in ["<|turn>", "<turn|>", "<|turn>user\n", "<|turn>model\n"] {
        let enc = tk.encode(s, false).map_err(anyhow::Error::msg)?;
        println!("{s:?} -> {:?}", enc.get_ids());
    }
    // full chat prompt with add_special
    let p = "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n";
    let enc = tk.encode(p, true).map_err(anyhow::Error::msg)?;
    println!("\nfull prompt ids: {:?}", enc.get_ids());
    println!("tokens: {:?}", enc.get_tokens());
    Ok(())
}
