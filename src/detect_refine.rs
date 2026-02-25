use image::imageops::crop_imm;
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use serde::Serialize;

use crate::detect::{BoundsNorm, Detection, detect_bounds};
use crate::kernels::line_fit::{
    fit_line_x_of_y, fit_line_y_of_x, horizontal_fit_angle_deg, pick_rotation_angle,
    vertical_fit_angle_deg,
};
use crate::kernels::peak_pick::{pick_peak_x, pick_peak_y};

#[cfg(feature = "debug-artifacts")]
#[path = "detect_refine_debug.rs"]
pub mod debug;
#[cfg(feature = "debug-artifacts")]
pub use debug::{RotationDebug, RotationLegend, rotation_debug_from_inner};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EdgeLineRotation {
    pub angle_deg: f32,
    pub top_deg: Option<f32>,
    pub bottom_deg: Option<f32>,
    pub left_deg: Option<f32>,
    pub right_deg: Option<f32>,
    pub points_top: usize,
    pub points_bottom: usize,
    pub points_left: usize,
    pub points_right: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RotationRefineConfig {
    pub refine_rotation: bool,
    pub apply_rotation_decision: bool,
    pub max_refine_abs_deg: f32,
    pub collect_debug: bool,
}

impl Default for RotationRefineConfig {
    fn default() -> Self {
        Self {
            refine_rotation: true,
            apply_rotation_decision: true,
            max_refine_abs_deg: 3.0,
            collect_debug: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetectRefineRun {
    pub detection_initial: Detection,
    pub detection_refined: Option<Detection>,
    pub detection: Detection,
    #[cfg(feature = "debug-artifacts")]
    pub rotation_initial_debug: Option<RotationDebug>,
    pub rotation_estimate: Option<EdgeLineRotation>,
    pub rotation_applied_deg: Option<f32>,
    pub rotation_residual_deg: Option<f32>,
}

pub fn run_detection_with_rotation_refine(
    gray: &GrayImage,
    refine_cfg: RotationRefineConfig,
) -> Option<DetectRefineRun> {
    let det = detect_bounds(gray)?;
    let rotation_estimate = estimate_rotation_from_inner(gray, det.inner);

    #[cfg(feature = "debug-artifacts")]
    let rotation_initial_debug = if refine_cfg.collect_debug {
        rotation_debug_from_inner(gray, det.inner)
    } else {
        None
    };

    let mut final_detection = det;
    let mut refined_detection = None;
    let mut rotation_applied_deg = None;
    let mut rotation_residual_deg = None;

    if refine_cfg.refine_rotation
        && refine_cfg.apply_rotation_decision
        && let Some(est) = rotation_estimate
        && est.angle_deg.abs() > 0.01
    {
        let apply_deg = (-est.angle_deg).clamp(
            -refine_cfg.max_refine_abs_deg.abs(),
            refine_cfg.max_refine_abs_deg.abs(),
        );
        let theta = apply_deg.to_radians();
        let rotated = rotate_about_center(gray, theta, Interpolation::Bilinear, Luma([0u8]));
        if let Some(det2) = detect_bounds(&rotated)
            && refined_bounds_plausible(det.inner, det2.inner)
        {
            final_detection = det2;
            refined_detection = Some(det2);
            rotation_applied_deg = Some(apply_deg);
            rotation_residual_deg =
                estimate_rotation_from_inner(&rotated, det2.inner).map(|r| r.angle_deg.abs());
        }
    }

    Some(DetectRefineRun {
        detection_initial: det,
        detection_refined: refined_detection,
        detection: final_detection,
        #[cfg(feature = "debug-artifacts")]
        rotation_initial_debug,
        rotation_estimate,
        rotation_applied_deg,
        rotation_residual_deg,
    })
}

pub fn estimate_rotation_from_inner(
    gray: &GrayImage,
    inner: BoundsNorm,
) -> Option<EdgeLineRotation> {
    let w = gray.width() as i32;
    let h = gray.height() as i32;
    if w < 32 || h < 32 {
        return None;
    }
    let x1 = ((inner.left * w as f32).round() as i32).clamp(2, w - 3);
    let x2 = ((inner.right * w as f32).round() as i32).clamp(2, w - 3);
    let y1 = ((inner.top * h as f32).round() as i32).clamp(2, h - 3);
    let y2 = ((inner.bottom * h as f32).round() as i32).clamp(2, h - 3);
    if x2 <= x1 + 16 || y2 <= y1 + 16 {
        return None;
    }

    let band = (((x2 - x1).min(y2 - y1) as f32) * 0.04)
        .round()
        .clamp(4.0, 14.0) as i32;
    let outward_slack = (band / 3).max(2);
    let mx = (((x2 - x1) as f32) * 0.08).round().max(3.0) as i32;
    let my = (((y2 - y1) as f32) * 0.08).round().max(3.0) as i32;
    let sx = (((x2 - x1) as f32) / 140.0).round().clamp(1.0, 4.0) as i32;
    let sy = (((y2 - y1) as f32) / 140.0).round().clamp(1.0, 4.0) as i32;

    let mut top_raw = Vec::new();
    let mut bottom_raw = Vec::new();
    let mut left_raw = Vec::new();
    let mut right_raw = Vec::new();

    let mut x = x1 + mx;
    while x <= x2 - mx {
        if let Some(y) = pick_peak_y(gray, x, y1 - outward_slack, y1 + band) {
            top_raw.push((x as f32, y as f32));
        }
        if let Some(y) = pick_peak_y(gray, x, y2 - band, y2 + outward_slack) {
            bottom_raw.push((x as f32, y as f32));
        }
        x += sx;
    }

    let mut y = y1 + my;
    while y <= y2 - my {
        if let Some(xp) = pick_peak_x(gray, x1 - outward_slack, x1 + band, y) {
            left_raw.push((xp as f32, y as f32));
        }
        if let Some(xp) = pick_peak_x(gray, x2 - band, x2 + outward_slack, y) {
            right_raw.push((xp as f32, y as f32));
        }
        y += sy;
    }

    let top_fit = fit_line_y_of_x(&top_raw);
    let bottom_fit = fit_line_y_of_x(&bottom_raw);
    let left_fit = fit_line_x_of_y(&left_raw);
    let right_fit = fit_line_x_of_y(&right_raw);

    let top_deg = top_fit.as_ref().map(horizontal_fit_angle_deg);
    let bottom_deg = bottom_fit.as_ref().map(horizontal_fit_angle_deg);
    let left_deg = left_fit.as_ref().map(vertical_fit_angle_deg);
    let right_deg = right_fit.as_ref().map(vertical_fit_angle_deg);

    let angle = pick_rotation_angle(
        top_fit.as_ref(),
        bottom_fit.as_ref(),
        left_fit.as_ref(),
        right_fit.as_ref(),
    )?;

    Some(EdgeLineRotation {
        angle_deg: angle,
        top_deg,
        bottom_deg,
        left_deg,
        right_deg,
        points_top: top_raw.len(),
        points_bottom: bottom_raw.len(),
        points_left: left_raw.len(),
        points_right: right_raw.len(),
    })
}

fn refined_bounds_plausible(initial: BoundsNorm, refined: BoundsNorm) -> bool {
    let area_initial = bounds_area(initial).max(1e-6);
    let area_refined = bounds_area(refined);
    let area_growth = area_refined / area_initial;

    let touches_refined = border_touch_count(refined, 0.01);
    let touches_initial = border_touch_count(initial, 0.01);

    if touches_refined >= 2 && touches_initial < 2 && area_growth > 1.10 {
        return false;
    }

    if area_refined > 0.95 && area_growth > 1.25 {
        return false;
    }

    true
}

fn bounds_area(b: BoundsNorm) -> f32 {
    let w = (b.right - b.left).abs().clamp(0.0, 1.0);
    let h = (b.bottom - b.top).abs().clamp(0.0, 1.0);
    w * h
}

fn border_touch_count(b: BoundsNorm, margin: f32) -> usize {
    let mut count = 0usize;
    if b.left <= margin {
        count += 1;
    }
    if b.top <= margin {
        count += 1;
    }
    if (1.0 - b.right) <= margin {
        count += 1;
    }
    if (1.0 - b.bottom) <= margin {
        count += 1;
    }
    count
}

pub fn draw_norm_rect(img: &mut RgbImage, b: BoundsNorm, color: Rgb<u8>) {
    let w = img.width() as f32;
    let h = img.height() as f32;
    let x1 = (b.left * w).clamp(0.0, w - 1.0);
    let x2 = (b.right * w).clamp(0.0, w - 1.0);
    let y1 = (b.top * h).clamp(0.0, h - 1.0);
    let y2 = (b.bottom * h).clamp(0.0, h - 1.0);
    draw_line_segment_mut(img, (x1, y1), (x2, y1), color);
    draw_line_segment_mut(img, (x2, y1), (x2, y2), color);
    draw_line_segment_mut(img, (x2, y2), (x1, y2), color);
    draw_line_segment_mut(img, (x1, y2), (x1, y1), color);
}

pub fn draw_refined_inner_backproject(
    img: &mut RgbImage,
    inner_rot: BoundsNorm,
    theta: f32,
    color: Rgb<u8>,
) {
    let unrot =
        refined_inner_backproject_px(inner_rot, theta, img.width() as f32, img.height() as f32);
    for i in 0..4 {
        let a = unrot[i];
        let b = unrot[(i + 1) % 4];
        draw_line_segment_mut(img, a, b, color);
    }
}

pub fn save_norm_crop(
    img: &RgbImage,
    b: BoundsNorm,
    out_path: &std::path::Path,
) -> anyhow::Result<()> {
    let w = img.width() as f32;
    let h = img.height() as f32;
    let x1 = (b.left * w).round().clamp(0.0, w - 1.0) as u32;
    let x2 = (b.right * w).round().clamp(0.0, w) as u32;
    let y1 = (b.top * h).round().clamp(0.0, h - 1.0) as u32;
    let y2 = (b.bottom * h).round().clamp(0.0, h) as u32;
    let cw = x2.saturating_sub(x1).max(1);
    let ch = y2.saturating_sub(y1).max(1);
    let crop = crop_imm(img, x1, y1, cw, ch).to_image();
    crop.save_with_format(out_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

pub fn refined_inner_backproject_px(
    inner_rot: BoundsNorm,
    theta: f32,
    w: f32,
    h: f32,
) -> [(f32, f32); 4] {
    let x1 = inner_rot.left * w;
    let x2 = inner_rot.right * w;
    let y1 = inner_rot.top * h;
    let y2 = inner_rot.bottom * h;
    let corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)];
    let c = ((w - 1.0) * 0.5, (h - 1.0) * 0.5);
    [
        rotate_point(corners[0].0, corners[0].1, c.0, c.1, -theta),
        rotate_point(corners[1].0, corners[1].1, c.0, c.1, -theta),
        rotate_point(corners[2].0, corners[2].1, c.0, c.1, -theta),
        rotate_point(corners[3].0, corners[3].1, c.0, c.1, -theta),
    ]
}

pub fn refined_inner_backproject_norm(
    inner_rot: BoundsNorm,
    theta: f32,
    w: f32,
    h: f32,
) -> [[f32; 2]; 4] {
    let px = refined_inner_backproject_px(inner_rot, theta, w, h);
    [
        [(px[0].0 / w).clamp(0.0, 1.0), (px[0].1 / h).clamp(0.0, 1.0)],
        [(px[1].0 / w).clamp(0.0, 1.0), (px[1].1 / h).clamp(0.0, 1.0)],
        [(px[2].0 / w).clamp(0.0, 1.0), (px[2].1 / h).clamp(0.0, 1.0)],
        [(px[3].0 / w).clamp(0.0, 1.0), (px[3].1 / h).clamp(0.0, 1.0)],
    ]
}

pub fn rotate_rgb_about_center(img: &RgbImage, theta: f32) -> RgbImage {
    rotate_about_center(img, theta, Interpolation::Bilinear, Rgb([0u8, 0u8, 0u8]))
}

fn rotate_point(x: f32, y: f32, cx: f32, cy: f32, theta: f32) -> (f32, f32) {
    let dx = x - cx;
    let dy = y - cy;
    let ct = theta.cos();
    let st = theta.sin();
    (cx + dx * ct - dy * st, cy + dx * st + dy * ct)
}
