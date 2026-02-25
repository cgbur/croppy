use image::GrayImage;
use serde::Serialize;

use super::EdgeLineRotation;
use crate::detect::BoundsNorm;
use crate::kernels::line_fit::{
    fit_line_x_of_y, fit_line_y_of_x, horizontal_fit_angle_deg, pick_rotation_angle,
    vertical_fit_angle_deg,
};
use crate::kernels::peak_pick::{pick_peak_x, pick_peak_y};

#[derive(Debug, Clone, Serialize)]
pub struct RotationDebug {
    pub summary: EdgeLineRotation,
    pub inner_px: [i32; 4],
    pub top_points: Vec<[f32; 2]>,
    pub bottom_points: Vec<[f32; 2]>,
    pub left_points: Vec<[f32; 2]>,
    pub right_points: Vec<[f32; 2]>,
    pub top_fit: Option<[f32; 2]>,    // y = m*x + b
    pub bottom_fit: Option<[f32; 2]>, // y = m*x + b
    pub left_fit: Option<[f32; 2]>,   // x = m*y + b
    pub right_fit: Option<[f32; 2]>,  // x = m*y + b
    pub legend: RotationLegend,
}

#[derive(Debug, Clone, Serialize)]
pub struct RotationLegend {
    pub top_points: &'static str,
    pub bottom_points: &'static str,
    pub left_points: &'static str,
    pub right_points: &'static str,
    pub top_line: &'static str,
    pub bottom_line: &'static str,
    pub left_line: &'static str,
    pub right_line: &'static str,
}

pub fn rotation_debug_from_inner(gray: &GrayImage, inner: BoundsNorm) -> Option<RotationDebug> {
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

    Some(RotationDebug {
        summary: EdgeLineRotation {
            angle_deg: angle,
            top_deg,
            bottom_deg,
            left_deg,
            right_deg,
            points_top: top_raw.len(),
            points_bottom: bottom_raw.len(),
            points_left: left_raw.len(),
            points_right: right_raw.len(),
        },
        inner_px: [x1, y1, x2, y2],
        top_points: top_raw.iter().map(|(x, y)| [*x, *y]).collect(),
        bottom_points: bottom_raw.iter().map(|(x, y)| [*x, *y]).collect(),
        left_points: left_raw.iter().map(|(x, y)| [*x, *y]).collect(),
        right_points: right_raw.iter().map(|(x, y)| [*x, *y]).collect(),
        top_fit: top_fit.map(|f| [f.slope(), f.intercept()]),
        bottom_fit: bottom_fit.map(|f| [f.slope(), f.intercept()]),
        left_fit: left_fit.map(|f| [f.slope(), f.intercept()]),
        right_fit: right_fit.map(|f| [f.slope(), f.intercept()]),
        legend: RotationLegend {
            top_points: "green points",
            bottom_points: "cyan points",
            left_points: "magenta points",
            right_points: "orange points",
            top_line: "green line",
            bottom_line: "cyan line",
            left_line: "magenta line",
            right_line: "orange line",
        },
    })
}
