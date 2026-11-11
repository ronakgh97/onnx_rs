use anyhow::Result;
use onnx_rs::Model;
fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("no model path provided"))?;
    let start = std::time::Instant::now();
    let model = Model::from_file(&model_path)?;
    println!("loaded model from {model_path} in {:?}", start.elapsed());

    let (inputs, outputs) = model.inputs_outputs();
    println!("input needed: {:?}", inputs);
    println!("output given: {:?}", outputs);
    println!("node count: {:?}", model.nodes().len());
    println!("operators used: {:?}", model.operators());
    println!();
    for (i, node) in model.nodes().iter().enumerate() {
        let (inputs, outputs) = node.inputs_outputs();
        println!(
            "node #{i}: name: {}, op_type: {}, inputs: {:?} outputs: {:?}",
            node.name(),
            node.op(),
            inputs,
            outputs,
        );
    }

    Ok(())
}
