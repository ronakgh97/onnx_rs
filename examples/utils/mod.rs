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

pub fn assert_pixel_diff_within_tolerance(
    image1: &image::RgbImage,
    image2: &image::RgbImage,
    tolerance: u8,
) -> anyhow::Result<()> {
    if image1.dimensions() != image2.dimensions() {
        return Err(anyhow::anyhow!(
            "image dimensions do not match: {:?} vs {:?}",
            image1.dimensions(),
            image2.dimensions()
        ));
    }

    for (x, y, pixel1) in image1.enumerate_pixels() {
        let pixel2 = image2.get_pixel(x, y);
        let diff_r = (pixel1[0] as i16 - pixel2[0] as i16).unsigned_abs() as u8;
        let diff_g = (pixel1[1] as i16 - pixel2[1] as i16).unsigned_abs() as u8;
        let diff_b = (pixel1[2] as i16 - pixel2[2] as i16).unsigned_abs() as u8;

        if diff_r > tolerance || diff_g > tolerance || diff_b > tolerance {
            return Err(anyhow::anyhow!(
                "pixel difference at ({}, {}): {:?} vs {:?}, diffs: r={}, g={}, b={}",
                x,
                y,
                pixel1,
                pixel2,
                diff_r,
                diff_g,
                diff_b
            ));
        }
    }

    Ok(())
}
