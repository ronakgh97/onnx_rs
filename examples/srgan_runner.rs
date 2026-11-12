pub mod utils;

use crate::utils::{fill_tensor_from_image, save_image_from_tensor};
use anyhow::Result;
use onnx_rs::{Model, Tensor};
use ort::ep::CPU;
use ort::value::TensorRef;
use ort::{inputs, session::Session};
use std::time::Instant;

fn main() -> Result<()> {
    // input image must be 128x128
    let in_image = image::open("images/sample_1.png")?.to_rgb8();
    let model_path = "models/srgan_onnx/srgan_single.onnx";

    println!("running onnx_rs... ");
    let start = Instant::now();
    let model = Model::from_file(model_path)?;

    let mut data = vec![0f32; 3 * 128 * 128];
    fill_tensor_from_image(&in_image, &mut data);

    let input = Tensor::create("inputs", &[1, 3, 128, 128], data)?;
    let output = model.run_single(input)?;
    save_image_from_tensor(output.data(), "images/srgan_onnx_rs_x4.png")?;
    println!("done in : {:?}", start.elapsed());

    // ---

    println!("running ort...");
    let start = Instant::now();
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?
        .with_execution_providers([CPU::default().build()])
        .map_err(|e| anyhow::anyhow!("unable to set execution providers: {e}"))?
        .commit_from_file(model_path)?;

    let mut data = vec![0f32; 3 * 128 * 128];
    fill_tensor_from_image(&in_image, &mut data);

    let input = TensorRef::from_array_view(([1, 3, 128, 128], data.as_ref()))?;
    let output = session.run(inputs!["image" => input])?;
    let (_, output) = output["upscaled_image"].try_extract_tensor::<f32>()?;
    save_image_from_tensor(output, "images/srgan_ort_x4.png")?;
    println!("done in : {:?}", start.elapsed());

    Ok(())
}
