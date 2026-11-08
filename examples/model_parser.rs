use anyhow::Result;
use onnx_rs::Model;

fn main() -> Result<()> {
    let model = Model::from_file("srgan_onnx/srgan_single.onnx")?;

    println!("input needed: {:?}", model.inputs);
    println!("output given: {:?}", model.outputs);
    println!("node count: {:?}", model.nodes.len());
    println!("operators needed: {:?}", model.operators());

    Ok(())
}
