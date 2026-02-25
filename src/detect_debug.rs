use image::{GrayImage, Rgb, RgbImage};
use serde::Serialize;

use super::{
    Detection, EDGE_REL_THRESH_HORIZONTAL, EDGE_REL_THRESH_VERTICAL, MIN_DETECT_DIM,
    inner_axis_config, sample_band_bounds,
};
use crate::kernels::{edge_scan, signal_1d};

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

pub fn detect_bounds_with_debug(gray: &GrayImage) -> Option<(Detection, DetectionDebug)> {
    let (w, h) = gray.dimensions();
    if w < MIN_DETECT_DIM || h < MIN_DETECT_DIM {
        return None;
    }

    let (y0, y1, x0, x1) = sample_band_bounds(w, h);
    let vertical_profile =
        signal_1d::smooth_boxcar(&signal_1d::profile_vertical_ranges(gray, &[(y0, y1)]), 3);
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
        inner: super::BoundsNorm {
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
    let pad = 10u32;
    let band_w = w.saturating_sub(pad * 2).max(1);

    {
        let mut top = RgbImage::new(w, plot_h);
        draw_profile_plot(series_a, series_b, markers, &mut top);
        overlay(out, &top, 0, 0);
    }

    let strip = if spec.is_vertical_axis {
        crop_gray(
            gray,
            0,
            spec.vertical_sample_y0,
            gray.width().saturating_sub(1),
            spec.vertical_sample_y1,
        )
    } else {
        let vertical = crop_gray(
            gray,
            spec.horizontal_sample_x0,
            0,
            spec.horizontal_sample_x1,
            gray.height().saturating_sub(1),
        );
        image::imageops::rotate270(&vertical)
    };

    let band = image::imageops::resize(
        &strip,
        band_w,
        band_h,
        image::imageops::FilterType::Triangle,
    );
    let band_rgb = image::DynamicImage::ImageLuma8(band).to_rgb8();
    let mut band_canvas = RgbImage::new(w, band_h);
    for px in band_canvas.pixels_mut() {
        *px = Rgb([16, 16, 16]);
    }
    overlay(&mut band_canvas, &band_rgb, pad.min(w.saturating_sub(1)), 0);
    overlay(out, &band_canvas, 0, plot_h);
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
