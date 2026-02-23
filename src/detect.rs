use image::{GrayImage, Rgb, RgbImage};
use serde::Serialize;

use crate::kernels::edge_scan::{self, AxisDetectConfig, EdgePolarity};
use crate::kernels::signal_1d;

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

#[derive(Debug, Clone, Serialize)]
pub struct DetectionDebug {
    pub width: u32,
    pub height: u32,
    pub vertical_profile: Vec<f32>,
    pub horizontal_profile: Vec<f32>,
    pub vertical_derivative: Vec<f32>,
    pub horizontal_derivative: Vec<f32>,
    pub inner_left_idx: usize,
    pub inner_right_idx: usize,
    pub inner_top_idx: usize,
    pub inner_bottom_idx: usize,
    pub vertical_sample_y0: u32,
    pub vertical_sample_y1: u32,
    pub horizontal_sample_x0: u32,
    pub horizontal_sample_x1: u32,
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
    detect_bounds_with_debug(gray).map(|(d, _)| d)
}

pub fn detect_bounds_with_debug(gray: &GrayImage) -> Option<(Detection, DetectionDebug)> {
    let (w, h) = gray.dimensions();
    if w < MIN_DETECT_DIM || h < MIN_DETECT_DIM {
        return None;
    }

    let m = BAND_MARGIN_PCT.clamp(BAND_MARGIN_MIN, BAND_MARGIN_MAX);
    let y0 = ((h as f32) * m).round() as u32;
    let y1 = ((h as f32) * (1.0 - m)).round() as u32;
    let x0 = ((w as f32) * m).round() as u32;
    let x1 = ((w as f32) * (1.0 - m)).round() as u32;
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

    let inner_l = vertical_edges.start;
    let inner_r = vertical_edges.end;
    let inner_t = horizontal_edges.start;
    let inner_b = horizontal_edges.end;

    let confidence = vertical_edges
        .score
        .min(horizontal_edges.score)
        .clamp(0.0, 1.0);

    let detection = Detection {
        inner: BoundsNorm {
            left: inner_l as f32 / w as f32,
            top: inner_t as f32 / h as f32,
            right: inner_r as f32 / w as f32,
            bottom: inner_b as f32 / h as f32,
        },
        confidence,
    };
    let debug = DetectionDebug {
        width: w,
        height: h,
        vertical_profile,
        horizontal_profile,
        vertical_derivative,
        horizontal_derivative,
        inner_left_idx: inner_l,
        inner_right_idx: inner_r,
        inner_top_idx: inner_t,
        inner_bottom_idx: inner_b,
        vertical_sample_y0: y0,
        vertical_sample_y1: y1,
        horizontal_sample_x0: x0,
        horizontal_sample_x1: x1,
    };

    Some((detection, debug))
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

pub fn draw_profile_plot(
    series_a: &[f32],
    series_b: &[f32],
    markers: &[usize],
    out: &mut RgbImage,
) {
    let (w, h) = out.dimensions();
    if w < 4 || h < 4 || series_a.is_empty() {
        return;
    }

    for px in out.pixels_mut() {
        *px = Rgb([16, 16, 16]);
    }

    let pad = 10u32;
    let pw = w.saturating_sub(pad * 2).max(1);
    let ph = h.saturating_sub(pad * 2).max(1);
    let area = PlotArea { pad, pw, ph };

    let (min_a, max_a) = series_min_max(series_a);
    let (min_b, max_b) = series_min_max(series_b);

    draw_series(
        series_a,
        min_a,
        max_a,
        out,
        area,
        Rgb([90, 170, 255]), // blue profile
    );
    draw_series(
        series_b,
        min_b,
        max_b,
        out,
        area,
        Rgb([255, 200, 80]), // orange derivative
    );

    if min_b < 0.0 && max_b > 0.0 {
        let y0 = scale_y(0.0, min_b, max_b, pad, ph, h);
        draw_line(
            out,
            pad as i32,
            y0 as i32,
            (pad + pw).min(w - 1) as i32,
            y0 as i32,
            Rgb([80, 80, 80]),
        );
    }

    for &m in markers {
        let x = pad
            + ((m as f32 / (series_a.len().saturating_sub(1).max(1) as f32)) * pw as f32) as u32;
        draw_vline(
            out,
            x.min(w - 1),
            pad,
            (pad + ph).min(h - 1),
            Rgb([255, 70, 70]),
        );
    }
}

fn draw_series(
    v: &[f32],
    min_v: f32,
    max_v: f32,
    out: &mut RgbImage,
    area: PlotArea,
    color: Rgb<u8>,
) {
    if v.len() < 2 {
        return;
    }
    let n = v.len() - 1;
    for i in 1..v.len() {
        let x1 = area.pad + ((i - 1) as f32 / n as f32 * area.pw as f32) as u32;
        let x2 = area.pad + (i as f32 / n as f32 * area.pw as f32) as u32;
        let y1 = scale_y(v[i - 1], min_v, max_v, area.pad, area.ph, out.height());
        let y2 = scale_y(v[i], min_v, max_v, area.pad, area.ph, out.height());
        draw_line(
            out,
            x1.min(out.width() - 1) as i32,
            y1 as i32,
            x2.min(out.width() - 1) as i32,
            y2 as i32,
            color,
        );
    }
}

#[derive(Clone, Copy)]
struct PlotArea {
    pad: u32,
    pw: u32,
    ph: u32,
}

fn series_min_max(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 1.0);
    }
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for val in v {
        min_v = min_v.min(*val);
        max_v = max_v.max(*val);
    }
    if (max_v - min_v).abs() <= 1e-6 {
        (min_v - 1.0, max_v + 1.0)
    } else {
        (min_v, max_v)
    }
}

fn scale_y(v: f32, min_v: f32, max_v: f32, pad: u32, ph: u32, img_h: u32) -> u32 {
    let span = (max_v - min_v).max(1e-6);
    let t = ((v - min_v) / span).clamp(0.0, 1.0);
    let y = pad + ph - (t * ph as f32) as u32;
    y.min(img_h.saturating_sub(1))
}

fn draw_vline(out: &mut RgbImage, x: u32, y1: u32, y2: u32, color: Rgb<u8>) {
    let ya = y1.min(y2);
    let yb = y1.max(y2);
    for y in ya..=yb {
        *out.get_pixel_mut(x, y) = color;
    }
}

fn draw_line(out: &mut RgbImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as u32) < out.width() && (y0 as u32) < out.height() {
            *out.get_pixel_mut(x0 as u32, y0 as u32) = color;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

pub fn draw_vertical_profile_with_band(
    gray: &GrayImage,
    dbg: &DetectionDebug,
    markers: &[usize],
    out: &mut RgbImage,
) {
    let spec = ProfileBandSpec {
        is_vertical_axis: true,
        horizontal_sample_x0: dbg.horizontal_sample_x0,
        horizontal_sample_x1: dbg.horizontal_sample_x1,
        vertical_sample_y0: dbg.vertical_sample_y0,
        vertical_sample_y1: dbg.vertical_sample_y1,
    };
    draw_profile_with_band(
        gray,
        &dbg.vertical_profile,
        &dbg.vertical_derivative,
        markers,
        out,
        spec,
    );
}

pub fn draw_horizontal_profile_with_band(
    gray: &GrayImage,
    dbg: &DetectionDebug,
    markers: &[usize],
    out: &mut RgbImage,
) {
    let spec = ProfileBandSpec {
        is_vertical_axis: false,
        horizontal_sample_x0: dbg.horizontal_sample_x0,
        horizontal_sample_x1: dbg.horizontal_sample_x1,
        vertical_sample_y0: dbg.vertical_sample_y0,
        vertical_sample_y1: dbg.vertical_sample_y1,
    };
    draw_profile_with_band(
        gray,
        &dbg.horizontal_profile,
        &dbg.horizontal_derivative,
        markers,
        out,
        spec,
    );
}

#[derive(Clone, Copy)]
struct ProfileBandSpec {
    is_vertical_axis: bool,
    horizontal_sample_x0: u32,
    horizontal_sample_x1: u32,
    vertical_sample_y0: u32,
    vertical_sample_y1: u32,
}

fn draw_profile_with_band(
    gray: &GrayImage,
    series_a: &[f32],
    series_b: &[f32],
    markers: &[usize],
    out: &mut RgbImage,
    spec: ProfileBandSpec,
) {
    let (w, h) = out.dimensions();
    if w < 8 || h < 8 {
        return;
    }
    let band_h = (h / 4).max(40);
    let plot_h = h - band_h;

    // Plot in top region.
    {
        let mut top = RgbImage::new(w, plot_h);
        draw_profile_plot(series_a, series_b, markers, &mut top);
        overlay(out, &top, 0, 0);
    }

    // Band in bottom region from exact sampling strip.
    let strip = if spec.is_vertical_axis {
        // Vertical profile used y-range [vertical_sample_y0..vertical_sample_y1] over full width.
        crop_gray(
            gray,
            0,
            spec.vertical_sample_y0,
            gray.width().saturating_sub(1),
            spec.vertical_sample_y1,
        )
    } else {
        // Horizontal profile used x-range [horizontal_sample_x0..horizontal_sample_x1] over full height.
        // Rotate so row index (y) maps left->right like plot x-axis.
        let vertical = crop_gray(
            gray,
            spec.horizontal_sample_x0,
            0,
            spec.horizontal_sample_x1,
            gray.height().saturating_sub(1),
        );
        image::imageops::rotate90(&vertical)
    };

    let band = image::imageops::resize(&strip, w, band_h, image::imageops::FilterType::Triangle);
    let band_rgb = image::DynamicImage::ImageLuma8(band).to_rgb8();
    overlay(out, &band_rgb, 0, plot_h);
}

fn crop_gray(gray: &GrayImage, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImage {
    let xa = x0.min(x1).min(gray.width().saturating_sub(1));
    let xb = x0.max(x1).min(gray.width().saturating_sub(1));
    let ya = y0.min(y1).min(gray.height().saturating_sub(1));
    let yb = y0.max(y1).min(gray.height().saturating_sub(1));
    let w = (xb - xa + 1).max(1);
    let h = (yb - ya + 1).max(1);
    image::imageops::crop_imm(gray, xa, ya, w, h).to_image()
}

fn overlay(dst: &mut RgbImage, src: &RgbImage, x0: u32, y0: u32) {
    for y in 0..src.height() {
        for x in 0..src.width() {
            let dx = x0 + x;
            let dy = y0 + y;
            if dx < dst.width() && dy < dst.height() {
                *dst.get_pixel_mut(dx, dy) = *src.get_pixel(x, y);
            }
        }
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
