use anyhow::Result;
use onnx_rs::{Model, Tensor};
use ort::ep::CPU;
use ort::value::TensorRef;
use ort::{inputs, session::Session};
use std::time::Instant;

fn main() -> Result<()> {
    let in_image = image::open("input_image.png")?.to_rgb8();
    let model_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("no model path provided"))?;

    println!("running onnx_rs... ");
    let start = Instant::now();
    let model = Model::from_file(&model_path)?;

    let mut data = vec![0f32; 3 * 128 * 128];
    fill_tensor_from_image(&in_image, &mut data);

    let input = Tensor::create("inputs", &[1, 3, 128, 128], &data)?;
    let output = model.run_single(input)?;
    save_image_from_tensor(output.data(), "onnx_rs_output.png")?;
    println!("done in : {:?}", start.elapsed());
    println!("saved onnx_rs_output.png");

    // ---

    println!("running ort...");
    let start = Instant::now();
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?
        .with_execution_providers([CPU::default().build()])
        .map_err(|e| anyhow::anyhow!("unable to set execution providers: {e}"))?
        .commit_from_file(&model_path)?;

    let mut data = vec![0f32; 3 * 128 * 128];
    fill_tensor_from_image(&in_image, &mut data);

    let input = TensorRef::from_array_view(([1, 3, 128, 128], data.as_ref()))?;
    let output = session.run(inputs!["image" => input])?;
    let (_, output) = output["upscaled_image"].try_extract_tensor::<f32>()?;
    save_image_from_tensor(output, "ort_output.png")?;
    println!("done in : {:?}", start.elapsed());
    println!("saved ort_output.png");

    Ok(())
}

fn fill_tensor_from_image(image: &image::RgbImage, data: &mut [f32]) {
    for (x, y, pixel) in image.enumerate_pixels() {
        let [r, g, b] = pixel.0;

        let index = (y * 128 + x) as usize;

        data[index] = r as f32 / 255.0;
        data[128 * 128 + index] = g as f32 / 255.0;
        data[2 * 128 * 128 + index] = b as f32 / 255.0;
    }
}

fn save_image_from_tensor(data: &[f32], filename: &str) -> Result<()> {
    let width: usize = 512;
    let height: usize = 512;
    let plane = width * height;
    let mut out_image =
        image::RgbImage::from_raw(width as u32, height as u32, vec![0u8; 3 * plane])
            .ok_or_else(|| anyhow::anyhow!("unable to create output image buffer"))?;

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;

            let r = data[index];
            let g = data[plane + index];
            let b = data[2 * plane + index];

            let pixel = image::Rgb([
                (r.clamp(0.0, 1.0) * 255.0).round() as u8,
                (g.clamp(0.0, 1.0) * 255.0).round() as u8,
                (b.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
            out_image.put_pixel(x as u32, y as u32, pixel);
        }
    }

    out_image.save(filename)?;
    println!("saved {}", filename);

    Ok(())
}
