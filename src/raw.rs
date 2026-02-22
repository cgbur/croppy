use std::path::Path;

use anyhow::{Result, anyhow};
use image::{GrayImage, ImageBuffer, Luma, RgbImage};
use rsraw::{BIT_DEPTH_8, ImageFormat, RawImage};

pub fn decode_raw_to_rgb(path: &Path) -> Result<RgbImage> {
    let bytes =
        std::fs::read(path).map_err(|e| anyhow!("raw read failed for {}: {e}", path.display()))?;
    decode_raw_bytes_to_rgb(&bytes, path)
}

pub fn decode_raw_bytes_to_rgb(bytes: &[u8], source: &Path) -> Result<RgbImage> {
    let mut raw = RawImage::open(bytes)
        .map_err(|e| anyhow!("raw decode failed for {}: {e}", source.display()))?;

    raw.unpack()
        .map_err(|e| anyhow!("raw unpack failed for {}: {e}", source.display()))?;
    // Fast path only: always decode half-size from LibRaw.
    raw.as_mut().params.half_size = 1;
    let image = raw
        .process::<BIT_DEPTH_8>()
        .map_err(|e| anyhow!("raw process failed for {}: {e}", source.display()))?;

    if image.image_format() != ImageFormat::Bitmap {
        return Err(anyhow!(
            "unexpected processed image format for {}: {:?}",
            source.display(),
            image.image_format()
        ));
    }

    let width = image.width();
    let height = image.height();
    let channels = image.colors() as usize;

    match channels {
        1 => {
            let gray = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, image.to_vec())
                .ok_or_else(|| {
                    anyhow!("failed to build grayscale image for {}", source.display())
                })?;
            Ok(image::DynamicImage::ImageLuma8(gray).to_rgb8())
        }
        3 => ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(width, height, image.to_vec())
            .ok_or_else(|| anyhow!("failed to build rgb image for {}", source.display())),
        4 => {
            let rgba =
                ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, image.to_vec())
                    .ok_or_else(|| {
                        anyhow!("failed to build rgba image for {}", source.display())
                    })?;
            Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8())
        }
        _ => Err(anyhow!(
            "unsupported channel count {} for {}",
            channels,
            source.display()
        )),
    }
}

pub fn rgb_to_gray(img: &RgbImage) -> GrayImage {
    image::DynamicImage::ImageRgb8(img.clone())
        .grayscale()
        .to_luma8()
}
