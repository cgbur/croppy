use image::{GrayImage, Rgb, RgbImage};
use serde::Serialize;

use crate::kernels::edge_scan::{self, AxisDetectConfig, EdgePolarity};
use crate::kernels::signal_1d;

#[cfg(feature = "debug-artifacts")]
#[path = "detect_debug.rs"]
pub mod debug;
#[cfg(feature = "debug-artifacts")]
pub use debug::{
    DetectionDebug, detect_bounds_with_debug, draw_horizontal_profile_with_band, draw_profile_plot,
    draw_vertical_profile_with_band,
};

// Detection requires at least this many pixels per axis to build stable
// derivatives and edge pairs.
const MIN_DETECT_DIM: u32 = 32;
// Relative trim from each side when building profiles. Lower includes more
// border noise, higher discards useful signal.
const BAND_MARGIN_PCT: f32 = 0.22;
const BAND_MARGIN_MIN: f32 = 0.02;
const BAND_MARGIN_MAX: f32 = 0.45;
// Edge scan defaults for inner-film bounds. We always expect:
// - left/top transitions to rise
// - right/bottom transitions to fall
const EDGE_MAX_SIDE_FRAC: f32 = 0.48;
const EDGE_SIDE_GUARD_FRAC: f32 = 0.02;
const EDGE_MIN_SPAN_FRAC: f32 = 0.08;
const EDGE_CENTER_BIAS: f32 = 0.8; // 0 = outer-facing, 1 = center-facing
const EDGE_REL_THRESH_VERTICAL: f32 = 0.50;
const EDGE_REL_THRESH_HORIZONTAL: f32 = 0.35;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BoundsNorm {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Detection {
    pub inner: BoundsNorm,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RotationEstimate {
    pub angle_deg: f32,
    pub top_angle_deg: f32,
    pub bottom_angle_deg: f32,
    pub points_top: usize,
    pub points_bottom: usize,
}

/// Detects film bounds from a preprocessed grayscale image.
///
/// Method: build smoothed 1D brightness profiles (vertical/horizontal), take
/// signed first derivatives, then detect one polarity-constrained inner edge
/// pair on each axis. Returns normalized inner rectangle and confidence from
/// edge strength.
pub fn detect_bounds(gray: &GrayImage) -> Option<Detection> {
    let (w, h) = gray.dimensions();
    if w < MIN_DETECT_DIM || h < MIN_DETECT_DIM {
        return None;
    }

    let (y0, y1, x0, x1) = sample_band_bounds(w, h);
    // Vertical profile is indexed by x and is used to detect left/right edges.
    let vertical_profile =
        signal_1d::smooth_boxcar(&signal_1d::profile_vertical_ranges(gray, &[(y0, y1)]), 3);
    // Horizontal profile is indexed by y and is used to detect top/bottom edges.
    // Keep a bit more detail on horizontal transitions.
    let horizontal_profile =
        signal_1d::smooth_boxcar(&signal_1d::profile_horizontal_ranges(gray, &[(x0, x1)]), 2);

    let vertical_derivative = signal_1d::signed_derivative(&vertical_profile);
    let horizontal_derivative = signal_1d::signed_derivative(&horizontal_profile);

    let vertical_edges = edge_scan::detect_axis_edges(
        &vertical_derivative,
        w as usize,
        inner_axis_config(EDGE_REL_THRESH_VERTICAL),
    )?;
    let horizontal_edges = edge_scan::detect_axis_edges(
        &horizontal_derivative,
        h as usize,
        inner_axis_config(EDGE_REL_THRESH_HORIZONTAL),
    )?;

    let confidence = vertical_edges
        .score
        .min(horizontal_edges.score)
        .clamp(0.0, 1.0);

    Some(Detection {
        inner: BoundsNorm {
            left: vertical_edges.start as f32 / w as f32,
            top: horizontal_edges.start as f32 / h as f32,
            right: vertical_edges.end as f32 / w as f32,
            bottom: horizontal_edges.end as f32 / h as f32,
        },
        confidence,
    })
}

fn inner_axis_config(rel_thresh: f32) -> AxisDetectConfig {
    AxisDetectConfig {
        max_side_frac: EDGE_MAX_SIDE_FRAC,
        side_guard_frac: EDGE_SIDE_GUARD_FRAC,
        min_span_frac: EDGE_MIN_SPAN_FRAC,
        rel_thresh,
        center_bias: EDGE_CENTER_BIAS,
        start_polarity: EdgePolarity::Rising,
        end_polarity: EdgePolarity::Falling,
    }
}

fn sample_band_bounds(w: u32, h: u32) -> (u32, u32, u32, u32) {
    let m = BAND_MARGIN_PCT.clamp(BAND_MARGIN_MIN, BAND_MARGIN_MAX);
    let y0 = ((h as f32) * m).round() as u32;
    let y1 = ((h as f32) * (1.0 - m)).round() as u32;
    let x0 = ((w as f32) * m).round() as u32;
    let x1 = ((w as f32) * (1.0 - m)).round() as u32;
    (y0, y1, x0, x1)
}

pub fn draw_detection_overlay(img: &mut RgbImage, det: Detection) {
    let (w, h) = img.dimensions();
    draw_norm_rect(
        img,
        det.inner,
        Rgb([255, 255, 0]), // yellow = inner
        w,
        h,
    );
}

pub fn estimate_rotation_from_inner(gray: &GrayImage, det: Detection) -> Option<RotationEstimate> {
    let (w, h) = gray.dimensions();
    if w < MIN_DETECT_DIM || h < MIN_DETECT_DIM {
        return None;
    }
    let x1 = ((det.inner.left * w as f32).round() as i32).clamp(0, (w - 1) as i32) as u32;
    let x2 = ((det.inner.right * w as f32).round() as i32).clamp(0, (w - 1) as i32) as u32;
    let y1 = ((det.inner.top * h as f32).round() as i32).clamp(0, (h - 1) as i32) as u32;
    let y2 = ((det.inner.bottom * h as f32).round() as i32).clamp(0, (h - 1) as i32) as u32;
    if x2 <= x1 + 8 || y2 <= y1 + 8 {
        return None;
    }

    let inner_w = x2 - x1;
    let x_margin = ((inner_w as f32) * 0.08).round() as u32;
    let xa = x1.saturating_add(x_margin).min(x2);
    let xb = x2.saturating_sub(x_margin).max(xa + 1);
    let search_half = ((h as f32) * 0.08).round().clamp(6.0, 80.0) as i32;
    let stride = ((inner_w as f32) / 180.0).round().clamp(1.0, 4.0) as u32;

    let top_pts = sample_edge_points(gray, xa, xb, y1 as i32, search_half, stride);
    let bot_pts = sample_edge_points(gray, xa, xb, y2 as i32, search_half, stride);
    if top_pts.len() < 8 || bot_pts.len() < 8 {
        return None;
    }

    let (m_top, _b_top) = fit_line(&top_pts)?;
    let (m_bot, _b_bot) = fit_line(&bot_pts)?;
    let top_angle = m_top.atan().to_degrees();
    let bot_angle = m_bot.atan().to_degrees();
    Some(RotationEstimate {
        angle_deg: (top_angle + bot_angle) * 0.5,
        top_angle_deg: top_angle,
        bottom_angle_deg: bot_angle,
        points_top: top_pts.len(),
        points_bottom: bot_pts.len(),
    })
}

fn draw_norm_rect(img: &mut RgbImage, b: BoundsNorm, color: Rgb<u8>, w: u32, h: u32) {
    let x1 = (b.left * w as f32).round() as i32;
    let y1 = (b.top * h as f32).round() as i32;
    let x2 = (b.right * w as f32).round() as i32;
    let y2 = (b.bottom * h as f32).round() as i32;

    draw_rect(
        img,
        x1.max(0) as u32,
        y1.max(0) as u32,
        x2.max(1) as u32,
        y2.max(1) as u32,
        color,
    );
}

fn draw_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return;
    }
    let xa = x1.min(w - 1);
    let xb = x2.min(w - 1);
    let ya = y1.min(h - 1);
    let yb = y2.min(h - 1);
    if xa >= xb || ya >= yb {
        return;
    }

    for x in xa..=xb {
        *img.get_pixel_mut(x, ya) = color;
        *img.get_pixel_mut(x, yb) = color;
    }
    for y in ya..=yb {
        *img.get_pixel_mut(xa, y) = color;
        *img.get_pixel_mut(xb, y) = color;
    }
}

fn sample_edge_points(
    gray: &GrayImage,
    x_start: u32,
    x_end: u32,
    y_hint: i32,
    search_half: i32,
    stride: u32,
) -> Vec<(f32, f32)> {
    let mut points = Vec::new();
    let h = gray.height() as i32;
    let y0 = (y_hint - search_half).clamp(1, h - 2);
    let y1 = (y_hint + search_half).clamp(1, h - 2);
    if y1 <= y0 {
        return points;
    }

    let mut x = x_start;
    while x <= x_end {
        let mut best_y = y_hint.clamp(y0, y1);
        let mut best_score = f32::MIN;
        for y in y0..=y1 {
            let a = gray.get_pixel(x, y as u32 - 1)[0] as f32;
            let b = gray.get_pixel(x, y as u32 + 1)[0] as f32;
            let grad = (b - a).abs();
            let dist = (y - y_hint).abs() as f32;
            let score = grad - dist * 0.25;
            if score > best_score {
                best_score = score;
                best_y = y;
            }
        }
        if best_score > 4.0 {
            points.push((x as f32, best_y as f32));
        }
        if x_end - x < stride {
            break;
        }
        x += stride.max(1);
    }
    points
}

fn fit_line(points: &[(f32, f32)]) -> Option<(f32, f32)> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f32;
    let (sum_x, sum_y) = points
        .iter()
        .fold((0.0f32, 0.0f32), |(sx, sy), (x, y)| (sx + *x, sy + *y));
    let mx = sum_x / n;
    let my = sum_y / n;

    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (x, y) in points {
        let dx = *x - mx;
        let dy = *y - my;
        num += dx * dy;
        den += dx * dx;
    }
    if den <= 1e-6 {
        return None;
    }
    let m = num / den;
    let b = my - m * mx;
    Some((m, b))
}
