//! Local peak-picking kernels on grayscale images.
//!
//! Rotation refinement samples edge points from narrow strips in the image.
//! This module provides the shared strip-based peak pickers.

use image::GrayImage;

/// Threshold config for strip peak-picking.
#[derive(Debug, Clone, Copy)]
pub struct PeakPickConfig {
    /// Minimum gradient magnitude required to accept a pick.
    pub min_gradient: f32,
}

impl Default for PeakPickConfig {
    fn default() -> Self {
        Self { min_gradient: 4.0 }
    }
}

/// Picks the y position with the strongest vertical finite-difference gradient
/// at a fixed x within `[y_min, y_max]`.
pub fn pick_peak_y(gray: &GrayImage, x: i32, y_min: i32, y_max: i32) -> Option<i32> {
    pick_peak_y_with_cfg(gray, x, y_min, y_max, PeakPickConfig::default())
}

/// Same as [`pick_peak_y`] but with explicit config.
pub fn pick_peak_y_with_cfg(
    gray: &GrayImage,
    x: i32,
    y_min: i32,
    y_max: i32,
    cfg: PeakPickConfig,
) -> Option<i32> {
    let h = gray.height() as i32;
    let y1 = y_min.clamp(2, h - 3);
    let y2 = y_max.clamp(2, h - 3);
    if y2 < y1 {
        return None;
    }

    let mut best_grad = 0.0f32;
    let mut best_y = y1;
    for y in y1..=y2 {
        let a = gray.get_pixel(x as u32, (y - 1) as u32)[0] as f32;
        let b = gray.get_pixel(x as u32, (y + 1) as u32)[0] as f32;
        let g = (b - a).abs();
        if g > best_grad {
            best_grad = g;
            best_y = y;
        }
    }

    if best_grad < cfg.min_gradient {
        None
    } else {
        Some(best_y)
    }
}

/// Picks the x position with the strongest horizontal finite-difference
/// gradient at a fixed y within `[x_min, x_max]`.
pub fn pick_peak_x(gray: &GrayImage, x_min: i32, x_max: i32, y: i32) -> Option<i32> {
    pick_peak_x_with_cfg(gray, x_min, x_max, y, PeakPickConfig::default())
}

/// Same as [`pick_peak_x`] but with explicit config.
pub fn pick_peak_x_with_cfg(
    gray: &GrayImage,
    x_min: i32,
    x_max: i32,
    y: i32,
    cfg: PeakPickConfig,
) -> Option<i32> {
    let w = gray.width() as i32;
    let x1 = x_min.clamp(2, w - 3);
    let x2 = x_max.clamp(2, w - 3);
    if x2 < x1 {
        return None;
    }

    let mut best_grad = 0.0f32;
    let mut best_x = x1;
    for x in x1..=x2 {
        let a = gray.get_pixel((x - 1) as u32, y as u32)[0] as f32;
        let b = gray.get_pixel((x + 1) as u32, y as u32)[0] as f32;
        let g = (b - a).abs();
        if g > best_grad {
            best_grad = g;
            best_x = x;
        }
    }

    if best_grad < cfg.min_gradient {
        None
    } else {
        Some(best_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_peak_y_finds_strong_transition() {
        let gray = GrayImage::from_fn(7, 7, |x, y| {
            if x == 3 {
                if y <= 2 {
                    image::Luma([10])
                } else {
                    image::Luma([220])
                }
            } else {
                image::Luma([0])
            }
        });

        let y = pick_peak_y(&gray, 3, 1, 5);
        assert_eq!(y, Some(2));
    }

    #[test]
    fn pick_peak_x_finds_strong_transition() {
        let gray = GrayImage::from_fn(7, 7, |x, _| {
            if x <= 2 {
                image::Luma([10])
            } else {
                image::Luma([220])
            }
        });

        let x = pick_peak_x(&gray, 1, 5, 3);
        assert_eq!(x, Some(2));
    }

    #[test]
    fn pick_peak_returns_none_for_flat_regions() {
        let gray = GrayImage::from_fn(8, 8, |_x, _y| image::Luma([100]));
        assert_eq!(pick_peak_y(&gray, 4, 1, 6), None);
        assert_eq!(pick_peak_x(&gray, 1, 6, 4), None);
    }
}
