//! 1D signal-building kernels used by detection.
//!
//! Detection works on 1D profiles derived from the image. This module contains
//! the shared profile and derivative transforms used by both detection paths.

use image::GrayImage;

/// Builds a vertical brightness profile (indexed by x) by averaging pixels in
/// one or more y-ranges.
pub fn profile_vertical_ranges(gray: &GrayImage, y_ranges: &[(u32, u32)]) -> Vec<f32> {
    let (w, _) = gray.dimensions();
    (0..w)
        .map(|x| {
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for (y0, y1) in y_ranges {
                for y in *y0..=*y1 {
                    sum += gray.get_pixel(x, y)[0] as f32;
                    n += 1;
                }
            }
            if n == 0 { 0.0 } else { sum / n as f32 }
        })
        .collect()
}

/// Builds a horizontal brightness profile (indexed by y) by averaging pixels in
/// one or more x-ranges.
pub fn profile_horizontal_ranges(gray: &GrayImage, x_ranges: &[(u32, u32)]) -> Vec<f32> {
    let (_, h) = gray.dimensions();
    (0..h)
        .map(|y| {
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for (x0, x1) in x_ranges {
                for x in *x0..=*x1 {
                    sum += gray.get_pixel(x, y)[0] as f32;
                    n += 1;
                }
            }
            if n == 0 { 0.0 } else { sum / n as f32 }
        })
        .collect()
}

/// Applies a centered boxcar smoothing window with edge clamping.
pub fn smooth_boxcar(v: &[f32], radius: usize) -> Vec<f32> {
    if v.is_empty() || radius == 0 {
        return v.to_vec();
    }

    let mut out = vec![0.0f32; v.len()];
    for (i, outv) in out.iter_mut().enumerate() {
        let a = i.saturating_sub(radius);
        let b = (i + radius).min(v.len() - 1);
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for val in v.iter().take(b + 1).skip(a) {
            sum += *val;
            n += 1;
        }
        *outv = if n == 0 { 0.0 } else { sum / n as f32 };
    }
    out
}

/// Computes a first-order signed derivative while preserving input length.
///
/// `out[0]` is always `0.0` so derivative indices stay aligned with profile
/// indices used by downstream edge picking.
pub fn signed_derivative(v: &[f32]) -> Vec<f32> {
    if v.len() < 2 {
        return vec![];
    }

    let mut out = vec![0.0f32; v.len()];
    for i in 1..v.len() {
        out[i] = v[i] - v[i - 1];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn smooth_boxcar_radius_one() {
        let out = smooth_boxcar(&[1.0, 2.0, 3.0, 10.0], 1);
        assert_eq!(out.len(), 4);
        assert_close(out[0], 1.5);
        assert_close(out[1], 2.0);
        assert_close(out[2], 5.0);
        assert_close(out[3], 6.5);
    }

    #[test]
    fn signed_derivative_keeps_length() {
        let out = signed_derivative(&[2.0, 5.5, 1.5]);
        assert_eq!(out, vec![0.0, 3.5, -4.0]);
    }

    #[test]
    fn profile_ranges_average_expected_pixels() {
        let gray = GrayImage::from_fn(3, 3, |x, y| image::Luma([((x + y * 3) * 10) as u8]));

        let vertical = profile_vertical_ranges(&gray, &[(1, 2)]);
        assert_eq!(vertical.len(), 3);
        assert_close(vertical[0], 45.0);
        assert_close(vertical[1], 55.0);
        assert_close(vertical[2], 65.0);

        let horizontal = profile_horizontal_ranges(&gray, &[(0, 1)]);
        assert_eq!(horizontal.len(), 3);
        assert_close(horizontal[0], 5.0);
        assert_close(horizontal[1], 35.0);
        assert_close(horizontal[2], 65.0);
    }
}
