use std::fs;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
use image::{GrayImage, Luma};
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};

use crate::detect::BoundsNorm;
use crate::detect_refine::{
    DetectRefineRun, RotationRefineConfig, draw_norm_rect, draw_refined_inner_backproject,
    run_detection_with_rotation_refine,
};
use crate::discover::is_supported_raw;
use crate::preprocess::{PreprocessConfig, prepare_image, resize_rgb_max_edge_owned};
use crate::raw::decode_raw_to_rgb_with_hint;

pub const PREVIEW_SUBDIR: &str = "previews";
pub const CANCELLED_MARKER: &str = "__croppy_cancelled__";
pub const FINAL_CROP_SCALE_MIN_PCT: f32 = -20.0;
pub const FINAL_CROP_SCALE_MAX_PCT: f32 = 20.0;
pub const FINAL_CROP_SCALE_FINE_STEP_PCT: f32 = 0.25;
pub const XMP_HORIZONTAL_TRIM_DEFAULT: f32 = 0.0;
pub const XMP_VERTICAL_TRIM_DEFAULT: f32 = 0.005;
pub const XMP_TRIM_MIN: f32 = -0.05;
pub const XMP_TRIM_MAX: f32 = 0.05;
pub const XMP_TRIM_STEP: f32 = 0.001;
const FRAME_PAD_FRAC: f32 = 0.06;
const FRAME_PAD_MIN_PX: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    DebugOverlay,
    FinalCrop,
    FinalCropFramed,
}

impl PreviewMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::DebugOverlay => "Overlay",
            Self::FinalCrop => "Final Crop",
            Self::FinalCropFramed => "Crop + Frame",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::DebugOverlay => Self::FinalCrop,
            Self::FinalCrop => Self::FinalCropFramed,
            Self::FinalCropFramed => Self::DebugOverlay,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub write_xmp: bool,
    pub write_preview: bool,
    pub preview_mode: PreviewMode,
    pub max_edge: u32,
    pub out_dir: PathBuf,
    pub final_crop_scale_pct: f32,
    pub horizontal_trim: f32,
    pub vertical_trim: f32,
}

#[derive(Debug, Clone)]
pub struct ProcessOutput {
    pub preview: Option<PathBuf>,
    pub xmp: Option<PathBuf>,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct XmpCrop {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    angle_deg: f32,
}

pub fn preview_dir(out_dir: &Path) -> PathBuf {
    out_dir.join(PREVIEW_SUBDIR)
}

pub fn adjust_final_crop_scale_pct(current: f32, delta: f32) -> f32 {
    clamp_final_crop_scale_pct(current + delta)
}

pub fn clamp_final_crop_scale_pct(value: f32) -> f32 {
    value.clamp(FINAL_CROP_SCALE_MIN_PCT, FINAL_CROP_SCALE_MAX_PCT)
}

pub fn adjust_xmp_trim(current: f32, delta: f32) -> f32 {
    clamp_xmp_trim(current + delta)
}

pub fn clamp_xmp_trim(value: f32) -> f32 {
    value.clamp(XMP_TRIM_MIN, XMP_TRIM_MAX)
}

pub fn is_cancelled_message(message: &str) -> bool {
    message == CANCELLED_MARKER
}

pub fn process_raw_file(
    raw: &Path,
    opts: &PipelineOptions,
    cancel: &AtomicBool,
) -> Result<ProcessOutput> {
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!(CANCELLED_MARKER));
    }

    let decoded = decode_raw_to_rgb_with_hint(raw, opts.max_edge)?;
    let warning = decoded.warning;
    let rgb = resize_rgb_max_edge_owned(decoded.image, opts.max_edge);
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!(CANCELLED_MARKER));
    }
    let gray = image::imageops::grayscale(&rgb);
    let preprocess_cfg = PreprocessConfig {
        invert: true,
        flip_180: true,
        black_pct: 2.0,
        white_pct: 80.0,
        knee_pct: 2.0,
    };
    let prepared = prepare_image(gray, preprocess_cfg);
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!(CANCELLED_MARKER));
    }

    let refine = run_detection_with_rotation_refine(&prepared, RotationRefineConfig::default())
        .ok_or_else(|| anyhow!("boundary detection failed"))?;

    let mut out = ProcessOutput {
        preview: None,
        xmp: None,
        warning,
    };

    if opts.write_preview {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(CANCELLED_MARKER));
        }
        let preview_path = preview_path_for(raw, &opts.out_dir);
        guard_write_target(raw, &preview_path, "preview")?;
        match opts.preview_mode {
            PreviewMode::DebugOverlay => {
                write_overlay_preview(
                    &prepared,
                    &refine,
                    opts.final_crop_scale_pct,
                    opts.horizontal_trim,
                    opts.vertical_trim,
                    &preview_path,
                )?;
            }
            PreviewMode::FinalCrop => {
                write_final_crop_preview(
                    &prepared,
                    &refine,
                    opts.final_crop_scale_pct,
                    opts.horizontal_trim,
                    opts.vertical_trim,
                    &preview_path,
                )?;
            }
            PreviewMode::FinalCropFramed => {
                write_framed_final_crop_preview(
                    &prepared,
                    &refine,
                    opts.final_crop_scale_pct,
                    opts.horizontal_trim,
                    opts.vertical_trim,
                    &preview_path,
                )?;
            }
        }
        out.preview = Some(preview_path);
    }

    if opts.write_xmp {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(CANCELLED_MARKER));
        }
        let xmp_crop = xmp_crop_from_detection(
            refine.detection.inner,
            refine.rotation_applied_deg,
            preprocess_cfg.flip_180,
            opts.final_crop_scale_pct,
            opts.horizontal_trim,
            opts.vertical_trim,
        );
        let xmp_path = raw.with_extension("xmp");
        guard_write_target(raw, &xmp_path, "xmp sidecar")?;
        write_xmp_sidecar(
            &xmp_path,
            xmp_crop.left,
            xmp_crop.top,
            xmp_crop.right,
            xmp_crop.bottom,
            xmp_crop.angle_deg,
        )?;
        out.xmp = Some(xmp_path);
    }

    Ok(out)
}

fn preview_path_for(raw: &Path, out_dir: &Path) -> PathBuf {
    let mut base = raw
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("preview")
        .to_string();
    base.push_str(".jpg");
    preview_dir(out_dir).join(base)
}

fn write_overlay_preview(
    prepared: &GrayImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
    preview_path: &Path,
) -> Result<()> {
    let mut overlay = gray_to_rgb(prepared);
    draw_norm_rect(
        &mut overlay,
        refine.detection_initial.inner,
        image::Rgb([255, 255, 0]),
    );
    if let (Some(refined_det), Some(applied_deg)) =
        (refine.detection_refined, refine.rotation_applied_deg)
    {
        draw_refined_inner_backproject(
            &mut overlay,
            refined_det.inner,
            applied_deg.to_radians(),
            image::Rgb([80, 255, 255]),
        );
    }
    let show_adjusted = final_crop_scale_pct.abs() > f32::EPSILON
        || horizontal_trim.abs() > f32::EPSILON
        || vertical_trim.abs() > f32::EPSILON;
    if show_adjusted {
        let (final_inner, theta_opt) = final_preview_inner_bounds(
            refine,
            final_crop_scale_pct,
            horizontal_trim,
            vertical_trim,
        );
        if let Some(theta) = theta_opt {
            draw_refined_inner_backproject(
                &mut overlay,
                final_inner,
                theta,
                image::Rgb([120, 255, 120]),
            );
        } else {
            draw_norm_rect(&mut overlay, final_inner, image::Rgb([120, 255, 120]));
        }
    }
    overlay.save_with_format(preview_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

fn write_final_crop_preview(
    prepared: &GrayImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
    preview_path: &Path,
) -> Result<()> {
    let crop = render_final_crop_image(
        prepared,
        refine,
        final_crop_scale_pct,
        horizontal_trim,
        vertical_trim,
    );
    gray_to_rgb(&crop).save_with_format(preview_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

fn write_framed_final_crop_preview(
    prepared: &GrayImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
    preview_path: &Path,
) -> Result<()> {
    let crop = render_final_crop_image(
        prepared,
        refine,
        final_crop_scale_pct,
        horizontal_trim,
        vertical_trim,
    );
    let framed = render_crop_with_white_frame(&crop);
    gray_to_rgb(&framed).save_with_format(preview_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

fn final_preview_inner_bounds(
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
) -> (BoundsNorm, Option<f32>) {
    if let (Some(refined_det), Some(applied_deg)) =
        (refine.detection_refined, refine.rotation_applied_deg)
    {
        let adjusted = adjust_final_inner_bounds(
            refined_det.inner,
            final_crop_scale_pct,
            horizontal_trim,
            vertical_trim,
        );
        (adjusted, Some(applied_deg.to_radians()))
    } else {
        let adjusted = adjust_final_inner_bounds(
            refine.detection.inner,
            final_crop_scale_pct,
            horizontal_trim,
            vertical_trim,
        );
        (adjusted, None)
    }
}

fn render_final_crop_image(
    prepared: &GrayImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
) -> GrayImage {
    let (final_inner, theta_opt) =
        final_preview_inner_bounds(refine, final_crop_scale_pct, horizontal_trim, vertical_trim);
    if let Some(theta) = theta_opt {
        let rotated = rotate_about_center(prepared, theta, Interpolation::Bilinear, Luma([0u8]));
        extract_norm_crop(&rotated, final_inner)
    } else {
        extract_norm_crop(prepared, final_inner)
    }
}

fn extract_norm_crop(img: &GrayImage, b: BoundsNorm) -> GrayImage {
    let w = img.width() as f32;
    let h = img.height() as f32;
    let x1 = (b.left * w).round().clamp(0.0, w - 1.0) as u32;
    let x2 = (b.right * w).round().clamp(0.0, w) as u32;
    let y1 = (b.top * h).round().clamp(0.0, h - 1.0) as u32;
    let y2 = (b.bottom * h).round().clamp(0.0, h) as u32;
    let cw = x2.saturating_sub(x1).max(1);
    let ch = y2.saturating_sub(y1).max(1);
    image::imageops::crop_imm(img, x1, y1, cw, ch).to_image()
}

fn render_crop_with_white_frame(crop: &GrayImage) -> GrayImage {
    let pad_x = ((crop.width() as f32 * FRAME_PAD_FRAC).round() as u32).max(FRAME_PAD_MIN_PX);
    let pad_y = ((crop.height() as f32 * FRAME_PAD_FRAC).round() as u32).max(FRAME_PAD_MIN_PX);
    let out_w = crop.width().saturating_add(pad_x.saturating_mul(2)).max(1);
    let out_h = crop.height().saturating_add(pad_y.saturating_mul(2)).max(1);
    let mut framed = GrayImage::from_pixel(out_w, out_h, Luma([255]));

    for (x, y, px) in crop.enumerate_pixels() {
        framed.put_pixel(x + pad_x, y + pad_y, *px);
    }

    framed
}

fn gray_to_rgb(img: &GrayImage) -> image::RgbImage {
    let mut out = image::RgbImage::new(img.width(), img.height());
    for (x, y, px) in img.enumerate_pixels() {
        let v = px[0];
        out.put_pixel(x, y, image::Rgb([v, v, v]));
    }
    out
}

fn xmp_crop_from_detection(
    inner_detected: BoundsNorm,
    rotation_applied_deg: Option<f32>,
    preprocess_flip_180: bool,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
) -> XmpCrop {
    let mut bounds = inner_detected;
    if preprocess_flip_180 {
        // Detection runs in a frame rotated 180 degrees; mirror bounds back so
        // sidecar geometry matches RAW orientation. CropAngle sign conversion is
        // handled separately below.
        bounds = rotate_bounds_180(bounds);
    }
    bounds = normalize_bounds(bounds);
    bounds = scale_bounds_about_center(bounds, final_crop_scale_pct);
    bounds = apply_axis_trim(
        bounds,
        clamp_xmp_trim(horizontal_trim),
        clamp_xmp_trim(vertical_trim),
    );
    XmpCrop {
        left: bounds.left,
        top: bounds.top,
        right: bounds.right,
        bottom: bounds.bottom,
        // Internal refine rotates image pixels by `rotation_applied_deg`, while
        // Lightroom's CropAngle uses the opposite sign convention.
        angle_deg: -rotation_applied_deg.unwrap_or(0.0),
    }
}

fn rotate_bounds_180(b: BoundsNorm) -> BoundsNorm {
    BoundsNorm {
        left: 1.0 - b.right,
        top: 1.0 - b.bottom,
        right: 1.0 - b.left,
        bottom: 1.0 - b.top,
    }
}

fn normalize_bounds(b: BoundsNorm) -> BoundsNorm {
    let mut left = b.left.clamp(0.0, 1.0);
    let mut right = b.right.clamp(0.0, 1.0);
    let mut top = b.top.clamp(0.0, 1.0);
    let mut bottom = b.bottom.clamp(0.0, 1.0);
    if left > right {
        std::mem::swap(&mut left, &mut right);
    }
    if top > bottom {
        std::mem::swap(&mut top, &mut bottom);
    }
    BoundsNorm {
        left,
        top,
        right,
        bottom,
    }
}

fn scale_bounds_about_center(bounds: BoundsNorm, scale_pct: f32) -> BoundsNorm {
    let scale_pct = clamp_final_crop_scale_pct(scale_pct);
    let scale = (1.0 + (scale_pct / 100.0)).max(0.01);

    let cx = (bounds.left + bounds.right) * 0.5;
    let cy = (bounds.top + bounds.bottom) * 0.5;
    let half_w = (bounds.right - bounds.left).abs() * 0.5 * scale;
    let half_h = (bounds.bottom - bounds.top).abs() * 0.5 * scale;

    normalize_bounds(BoundsNorm {
        left: cx - half_w,
        top: cy - half_h,
        right: cx + half_w,
        bottom: cy + half_h,
    })
}

fn adjust_final_inner_bounds(
    bounds: BoundsNorm,
    final_crop_scale_pct: f32,
    horizontal_trim: f32,
    vertical_trim: f32,
) -> BoundsNorm {
    let scaled = scale_bounds_about_center(normalize_bounds(bounds), final_crop_scale_pct);
    apply_axis_trim(
        scaled,
        clamp_xmp_trim(horizontal_trim),
        clamp_xmp_trim(vertical_trim),
    )
}

fn apply_axis_trim(bounds: BoundsNorm, horizontal_trim: f32, vertical_trim: f32) -> BoundsNorm {
    let b = normalize_bounds(bounds);
    let width = (b.right - b.left).abs();
    let height = (b.bottom - b.top).abs();
    // Prevent positive trim from collapsing an axis past zero width/height.
    let max_trim_x = (width * 0.5 - 1e-6).max(0.0);
    let max_trim_y = (height * 0.5 - 1e-6).max(0.0);
    let trim_x = horizontal_trim.min(max_trim_x);
    let trim_y = vertical_trim.min(max_trim_y);

    normalize_bounds(BoundsNorm {
        left: b.left + trim_x,
        top: b.top + trim_y,
        right: b.right - trim_x,
        bottom: b.bottom - trim_y,
    })
}

fn guard_write_target(raw: &Path, target: &Path, output_kind: &str) -> Result<()> {
    if raw == target {
        return Err(anyhow!(
            "refusing to write {output_kind}: target path equals source RAW path ({})",
            raw.display()
        ));
    }
    if is_supported_raw(target) {
        return Err(anyhow!(
            "refusing to write {output_kind}: target has RAW extension ({})",
            target.display()
        ));
    }

    let raw_abs = fs::canonicalize(raw).unwrap_or_else(|_| raw.to_path_buf());
    let target_abs = fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    if raw_abs == target_abs {
        return Err(anyhow!(
            "refusing to write {output_kind}: target resolves to source RAW path ({})",
            target.display()
        ));
    }

    if let (Ok(raw_meta), Ok(target_meta)) = (fs::metadata(raw), fs::metadata(target)) {
        #[cfg(unix)]
        {
            if raw_meta.dev() == target_meta.dev() && raw_meta.ino() == target_meta.ino() {
                return Err(anyhow!(
                    "refusing to write {output_kind}: target points to same file inode as source RAW ({})",
                    target.display()
                ));
            }
        }
    }
    Ok(())
}

fn write_xmp_sidecar(
    path: &Path,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    angle_deg: f32,
) -> Result<()> {
    let xmp = format!(
        concat!(
            "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n",
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n",
            " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n",
            "  <rdf:Description rdf:about=\"\"\n",
            "    xmlns:crs=\"http://ns.adobe.com/camera-raw-settings/1.0/\"\n",
            "    crs:HasCrop=\"True\"\n",
            "    crs:CropConstrainAspectRatio=\"True\"\n",
            "    crs:CropLeft=\"{left:.6}\"\n",
            "    crs:CropTop=\"{top:.6}\"\n",
            "    crs:CropRight=\"{right:.6}\"\n",
            "    crs:CropBottom=\"{bottom:.6}\"\n",
            "    crs:CropAngle=\"{angle:.6}\" />\n",
            " </rdf:RDF>\n",
            "</x:xmpmeta>\n",
            "<?xpacket end=\"w\"?>\n"
        ),
        left = left,
        top = top,
        right = right,
        bottom = bottom,
        angle = angle_deg
    );
    fs::write(path, xmp)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::{BoundsNorm, PreviewMode, xmp_crop_from_detection};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn xmp_crop_keeps_bounds_without_flip() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.1,
                top: 0.2,
                right: 0.8,
                bottom: 0.9,
            },
            Some(-0.5),
            false,
            0.0,
            0.0,
            0.0,
        );
        assert_close(out.left, 0.1);
        assert_close(out.top, 0.2);
        assert_close(out.right, 0.8);
        assert_close(out.bottom, 0.9);
        assert_close(out.angle_deg, 0.5);
    }

    #[test]
    fn xmp_crop_mirrors_bounds_when_preprocess_flipped() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.1,
                top: 0.2,
                right: 0.8,
                bottom: 0.9,
            },
            Some(1.25),
            true,
            0.0,
            0.0,
            0.0,
        );
        assert_close(out.left, 0.2);
        assert_close(out.top, 0.1);
        assert_close(out.right, 0.9);
        assert_close(out.bottom, 0.8);
        assert_close(out.angle_deg, -1.25);
    }

    #[test]
    fn xmp_crop_clamps_and_normalizes_bounds() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 1.2,
                top: 0.9,
                right: -0.1,
                bottom: 0.2,
            },
            None,
            false,
            0.0,
            0.0,
            0.0,
        );
        assert_close(out.left, 0.0);
        assert_close(out.top, 0.2);
        assert_close(out.right, 1.0);
        assert_close(out.bottom, 0.9);
        assert_close(out.angle_deg, 0.0);
    }

    #[test]
    fn xmp_crop_expands_around_center() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            10.0,
            0.0,
            0.0,
        );
        assert_close(out.left, 0.17);
        assert_close(out.top, 0.28);
        assert_close(out.right, 0.83);
        assert_close(out.bottom, 0.72);
        assert_close(out.angle_deg, 0.0);
    }

    #[test]
    fn xmp_crop_shrinks_around_center() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            -10.0,
            0.0,
            0.0,
        );
        assert_close(out.left, 0.23);
        assert_close(out.top, 0.32);
        assert_close(out.right, 0.77);
        assert_close(out.bottom, 0.68);
        assert_close(out.angle_deg, 0.0);
    }

    #[test]
    fn xmp_crop_scale_is_clamped() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            100.0,
            0.0,
            0.0,
        );
        assert_close(out.left, 0.14);
        assert_close(out.top, 0.26);
        assert_close(out.right, 0.86);
        assert_close(out.bottom, 0.74);
    }

    #[test]
    fn xmp_crop_applies_horizontal_and_vertical_trim() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            0.0,
            0.01,
            0.02,
        );
        assert_close(out.left, 0.21);
        assert_close(out.top, 0.32);
        assert_close(out.right, 0.79);
        assert_close(out.bottom, 0.68);
    }

    #[test]
    fn xmp_crop_trim_uses_limits() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            0.0,
            1.0,
            -1.0,
        );
        assert_close(out.left, 0.25);
        assert_close(out.top, 0.25);
        assert_close(out.right, 0.75);
        assert_close(out.bottom, 0.75);
    }

    #[test]
    fn preview_mode_cycles() {
        assert_eq!(PreviewMode::DebugOverlay.next(), PreviewMode::FinalCrop);
        assert_eq!(PreviewMode::FinalCrop.next(), PreviewMode::FinalCropFramed);
        assert_eq!(
            PreviewMode::FinalCropFramed.next(),
            PreviewMode::DebugOverlay
        );
    }

    #[test]
    fn framed_crop_adds_white_border_area() {
        let crop = GrayImage::from_pixel(100, 50, Luma([56]));
        let framed = super::render_crop_with_white_frame(&crop);
        let pad_x = ((crop.width() as f32 * super::FRAME_PAD_FRAC).round() as u32)
            .max(super::FRAME_PAD_MIN_PX);
        let pad_y = ((crop.height() as f32 * super::FRAME_PAD_FRAC).round() as u32)
            .max(super::FRAME_PAD_MIN_PX);
        assert!(framed.width() > crop.width());
        assert!(framed.height() > crop.height());
        assert_eq!(*framed.get_pixel(0, 0), Luma([255]));
        assert_eq!(*framed.get_pixel(pad_x, pad_y), Luma([56]));
    }
}
