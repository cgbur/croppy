use image::imageops::FilterType;
use image::{GrayImage, RgbImage};

const RESIZE_SKIP_REL_TOLERANCE: f32 = 0.25;

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub invert: bool,
    pub flip_180: bool,
    pub black_pct: f32,
    pub white_pct: f32,
    pub knee_pct: f32,
}

pub fn prepare_image(mut gray: GrayImage, cfg: PreprocessConfig) -> GrayImage {
    if cfg.invert {
        image::imageops::invert(&mut gray);
    }
    gray = stretch_levels(gray, cfg.black_pct, cfg.white_pct, cfg.knee_pct);
    if cfg.flip_180 {
        gray = image::imageops::rotate180(&gray);
    }
    gray
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
    // Avoid tiny downscales near target size; they cost CPU but don't help detection much.
    let near_target_ceiling = ((max_edge as f32) * (1.0 + RESIZE_SKIP_REL_TOLERANCE)).ceil() as u32;
    if long <= near_target_ceiling {
        return img.clone();
    }
    let scale = max_edge as f32 / long as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    image::imageops::resize(img, new_w, new_h, FilterType::Triangle)
}

pub fn resize_rgb_max_edge_owned(img: RgbImage, max_edge: u32) -> RgbImage {
    if max_edge == 0 {
        return img;
    }
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= max_edge {
        return img;
    }
    // Avoid tiny downscales near target size; they cost CPU but don't help detection much.
    let near_target_ceiling = ((max_edge as f32) * (1.0 + RESIZE_SKIP_REL_TOLERANCE)).ceil() as u32;
    if long <= near_target_ceiling {
        return img;
    }
    let scale = max_edge as f32 / long as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    image::imageops::resize(&img, new_w, new_h, FilterType::Triangle)
}

fn stretch_levels(gray: GrayImage, black_pct: f32, white_pct: f32, knee_pct: f32) -> GrayImage {
    let raw = gray.as_raw();
    if raw.is_empty() {
        return gray;
    }

    let mut hist = [0u32; 256];
    for &v in raw {
        hist[v as usize] += 1;
    }

    let last = raw.len() - 1;
    let lo_idx = ((black_pct.clamp(0.0, 100.0) / 100.0) * last as f32).round() as usize;
    let hi_idx = ((white_pct.clamp(0.0, 100.0) / 100.0) * last as f32).round() as usize;
    let lo = sample_at_rank(&hist, lo_idx.min(last)) as f32;
    let hi = sample_at_rank(&hist, hi_idx.min(last)) as f32;

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

fn sample_at_rank(hist: &[u32; 256], rank: usize) -> u8 {
    let mut seen = 0usize;
    for (value, count) in hist.iter().enumerate() {
        seen += *count as usize;
        if seen > rank {
            return value as u8;
        }
    }
    255
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

#[cfg(test)]
mod tests {
    use super::resize_rgb_max_edge;

    #[test]
    fn resize_skips_when_within_tolerance_above_target() {
        let img = image::RgbImage::from_pixel(1040, 700, image::Rgb([1, 2, 3]));
        let out = resize_rgb_max_edge(&img, 1000);
        assert_eq!(out.dimensions(), (1040, 700));
    }

    #[test]
    fn resize_still_happens_when_far_above_target() {
        let img = image::RgbImage::from_pixel(1300, 700, image::Rgb([1, 2, 3]));
        let out = resize_rgb_max_edge(&img, 1000);
        assert_eq!(out.dimensions(), (1000, 538));
    }

    #[test]
    fn resize_skips_up_to_twenty_five_percent_above_target() {
        let img = image::RgbImage::from_pixel(1250, 700, image::Rgb([1, 2, 3]));
        let out = resize_rgb_max_edge(&img, 1000);
        assert_eq!(out.dimensions(), (1250, 700));
    }
}
