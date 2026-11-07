use anyhow::Result;
use onnx_rs::Model;

fn main() -> Result<()> {
    let model = Model::from_file("srgan_onnx/srgan_single.onnx")?;

    println!("node count: {}", model.nodes.len());
    for node in &model.nodes {
        println!("node: {}", node.name);
        println!("op: {}", node.op.static_name());
        println!("inputs: {:?}", node.inputs);
        println!("outputs: {:?}", node.outputs);
        println!("attrs: {:?}", node.attrs);
    }

    Ok(())
}
