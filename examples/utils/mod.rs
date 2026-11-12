pub fn fill_tensor_from_image(image: &image::RgbImage, data: &mut [f32]) {
    for (x, y, pixel) in image.enumerate_pixels() {
        let [r, g, b] = pixel.0;

        let index = (y * 128 + x) as usize;

        data[index] = r as f32 / 255.0;
        data[128 * 128 + index] = g as f32 / 255.0;
        data[2 * 128 * 128 + index] = b as f32 / 255.0;
    }
}

pub fn save_image_from_tensor(data: &[f32], filename: &str) -> anyhow::Result<()> {
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
