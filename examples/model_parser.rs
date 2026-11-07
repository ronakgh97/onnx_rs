use anyhow::Result;
use onnx_rs::Model;

fn main() -> Result<()> {
    let _model = Model::from_file("srgan_onnx/srgan_single.onnx")?;
    Ok(())
}
