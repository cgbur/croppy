use image::imageops::FilterType;
use image::{GrayImage, RgbImage};

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub invert: bool,
    pub flip_180: bool,
    pub black_pct: f32,
    pub white_pct: f32,
    pub knee_pct: f32,
}

pub fn prepare_image(mut gray: GrayImage, cfg: PreprocessConfig) -> RgbImage {
    if cfg.invert {
        image::imageops::invert(&mut gray);
    }
    gray = stretch_levels(gray, cfg.black_pct, cfg.white_pct, cfg.knee_pct);
    if cfg.flip_180 {
        gray = image::imageops::rotate180(&gray);
    }
    image::DynamicImage::ImageLuma8(gray).to_rgb8()
}

pub fn resize_rgb_max_edge(img: &RgbImage, max_edge: u32) -> RgbImage {
    if max_edge == 0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= max_edge {
        return img.clone();
    }
    let scale = max_edge as f32 / long as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    image::imageops::resize(img, new_w, new_h, FilterType::Triangle)
}

fn stretch_levels(gray: GrayImage, black_pct: f32, white_pct: f32, knee_pct: f32) -> GrayImage {
    let mut samples = gray.as_raw().clone();
    if samples.is_empty() {
        return gray;
    }
    samples.sort_unstable();

    let last = samples.len() - 1;
    let lo_idx = ((black_pct.clamp(0.0, 100.0) / 100.0) * last as f32).round() as usize;
    let hi_idx = ((white_pct.clamp(0.0, 100.0) / 100.0) * last as f32).round() as usize;
    let lo = samples[lo_idx.min(last)] as f32;
    let hi = samples[hi_idx.min(last)] as f32;

    if hi <= lo + 1.0 {
        return gray;
    }

    let k = knee_pct.clamp(0.0, 49.0) / 100.0;
    let mut out = gray;
    for p in out.pixels_mut() {
        let v = p[0] as f32;
        let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
        let y = if k > 0.0 { soft_knee(t, k) } else { t };
        p[0] = (y * 255.0).round() as u8;
    }
    out
}

fn soft_knee(t: f32, k: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    if t < k {
        // Cubic toe: f(0)=0, f'(0)=0, f(k)=k, f'(k)=1
        return -t * t * t / (k * k) + 2.0 * t * t / k;
    }
    if t > 1.0 - k {
        let u = 1.0 - t;
        let fu = -u * u * u / (k * k) + 2.0 * u * u / k;
        return 1.0 - fu;
    }
    t
}
