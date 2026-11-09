use anyhow::Result;
use onnx_rs::Model;

fn main() -> Result<()> {
    let args = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("no model path provided"))?;
    let model = Model::from_file(&args)?;

    println!("input needed: {:?}", model.inputs);
    println!("output given: {:?}", model.outputs);
    println!("node count: {:?}", model.nodes.len());
    println!("operators used: {:?}", model.operators());

    Ok(())
}
