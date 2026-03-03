use image::GrayImage;
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

impl BoundsNorm {
    pub fn rotate_180(self) -> Self {
        Self {
            left: 1.0 - self.right,
            top: 1.0 - self.bottom,
            right: 1.0 - self.left,
            bottom: 1.0 - self.top,
        }
    }

    pub fn normalize(self) -> Self {
        let mut left = self.left.clamp(0.0, 1.0);
        let mut right = self.right.clamp(0.0, 1.0);
        let mut top = self.top.clamp(0.0, 1.0);
        let mut bottom = self.bottom.clamp(0.0, 1.0);
        if left > right {
            std::mem::swap(&mut left, &mut right);
        }
        if top > bottom {
            std::mem::swap(&mut top, &mut bottom);
        }
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    pub fn scale_about_center(self, scale_pct: f32) -> Self {
        let scale_pct = scale_pct.clamp(SCALE_MIN_PCT, SCALE_MAX_PCT);
        let scale = (1.0 + (scale_pct / 100.0)).max(0.01);
        let cx = (self.left + self.right) * 0.5;
        let cy = (self.top + self.bottom) * 0.5;
        let half_w = (self.right - self.left).abs() * 0.5 * scale;
        let half_h = (self.bottom - self.top).abs() * 0.5 * scale;
        Self {
            left: cx - half_w,
            top: cy - half_h,
            right: cx + half_w,
            bottom: cy + half_h,
        }
        .normalize()
    }

    pub fn apply_trim(self, horizontal: f32, vertical: f32) -> Self {
        let b = self.normalize();
        let width = (b.right - b.left).abs();
        let height = (b.bottom - b.top).abs();
        // Prevent positive trim from collapsing an axis past zero width/height.
        let max_trim_x = (width * 0.5 - 1e-6).max(0.0);
        let max_trim_y = (height * 0.5 - 1e-6).max(0.0);
        let trim_x = horizontal.min(max_trim_x);
        let trim_y = vertical.min(max_trim_y);
        Self {
            left: b.left + trim_x,
            top: b.top + trim_y,
            right: b.right - trim_x,
            bottom: b.bottom - trim_y,
        }
        .normalize()
    }
}

// Crop scale limits exposed for the TUI and pipeline.
pub const SCALE_MIN_PCT: f32 = -20.0;
pub const SCALE_MAX_PCT: f32 = 20.0;
pub const TRIM_MIN: f32 = -0.05;
pub const TRIM_MAX: f32 = 0.05;

pub fn clamp_scale_pct(value: f32) -> f32 {
    value.clamp(SCALE_MIN_PCT, SCALE_MAX_PCT)
}

pub fn clamp_trim(value: f32) -> f32 {
    value.clamp(TRIM_MIN, TRIM_MAX)
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Detection {
    pub inner: BoundsNorm,
    pub confidence: f32,
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
