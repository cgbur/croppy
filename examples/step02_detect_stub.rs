use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::Parser;
use croppy::detect::{
    BoundsNorm, DetectConfig, Detection, DetectionDebug, detect_bounds_with_debug_cfg,
    draw_horizontal_profile_with_band, draw_vertical_profile_with_band,
};
use croppy::detect_refine::{
    EdgeLineRotation, RotationDebug, RotationRefineConfig, draw_norm_rect,
    draw_refined_inner_backproject, refined_inner_backproject_norm, rotate_rgb_about_center,
    run_detection_with_rotation_refine, save_norm_crop,
};
use croppy::handoff::{Step01Transform, read_handoff};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(about = "Step 02: boundary detection + deterministic rotation refine")]
struct Args {
    #[arg(long, default_value = "tmp/step01/next.json")]
    input: PathBuf,

    #[arg(long, default_value = "tmp/step02")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = true)]
    dump_debug: bool,

    #[arg(long, default_value_t = 0.22)]
    band_margin_pct: f32,

    #[arg(long, default_value_t = true)]
    refine_rotation: bool,

    #[arg(long, default_value_t = true)]
    apply_rotation_decision: bool,

    #[arg(long, default_value_t = 3.0)]
    max_refine_abs_deg: f32,
}

#[derive(Debug, Serialize)]
struct Step02Result {
    raw: String,
    prepared: String,
    preprocess: String,
    transform: Step01Transform,
    detection_initial: Detection,
    detection_refined: Option<Detection>,
    detection: Detection,
    rotation_estimate: Option<EdgeLineRotation>,
    rotation_applied_deg: Option<f32>,
    rotation_residual_deg: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OverlayJson {
    outer: BoundsNorm,
    inner: BoundsNorm,
    rotated_inner: Option<[[f32; 2]; 4]>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.input.exists() {
        return Err(anyhow!("handoff json not found: {}", args.input.display()));
    }

    let handoff = read_handoff(&args.input)?;
    let prepared = PathBuf::from(&handoff.prepared);
    if !prepared.exists() {
        return Err(anyhow!("prepared image missing: {}", prepared.display()));
    }

    std::fs::create_dir_all(&args.out_dir)?;
    let dyn_img = image::open(&prepared)?;
    let gray = dyn_img.to_luma8();
    let base_rgb = dyn_img.to_rgb8();

    let cfg = DetectConfig {
        band_margin_pct: args.band_margin_pct,
        ..DetectConfig::default()
    };
    let refine_cfg = RotationRefineConfig {
        refine_rotation: args.refine_rotation,
        apply_rotation_decision: args.apply_rotation_decision,
        max_refine_abs_deg: args.max_refine_abs_deg,
    };

    let Some(refine) = run_detection_with_rotation_refine(&gray, cfg, refine_cfg) else {
        return Err(anyhow!("failed detecting boundaries"));
    };
    let Some((_, dbg)) = detect_bounds_with_debug_cfg(&gray, cfg) else {
        return Err(anyhow!("failed collecting debug vectors"));
    };

    if args.dump_debug {
        write_debug_artifacts(&args.out_dir, &gray, &dbg)?;
        if let Some(d) = &refine.rotation_initial_debug {
            write_rotation_decision_artifacts(
                &args.out_dir,
                &base_rgb,
                refine.detection_initial.inner,
                d,
            )?;
        }
    }

    let mut overlay_initial = base_rgb.clone();
    draw_norm_rect(
        &mut overlay_initial,
        refine.detection_initial.inner,
        Rgb([255, 255, 0]),
    );
    let overlay_initial_path = args.out_dir.join("overlay_initial.jpg");
    overlay_initial.save_with_format(&overlay_initial_path, image::ImageFormat::Jpeg)?;

    let mut overlay = base_rgb.clone();
    draw_norm_rect(
        &mut overlay,
        refine.detection_initial.outer,
        Rgb([255, 0, 0]),
    );
    draw_norm_rect(
        &mut overlay,
        refine.detection_initial.inner,
        Rgb([255, 255, 0]),
    );
    if let (Some(refined), Some(applied)) = (refine.detection_refined, refine.rotation_applied_deg)
    {
        draw_refined_inner_backproject(
            &mut overlay,
            refined.inner,
            applied.to_radians(),
            Rgb([80, 255, 255]),
        );
    }
    let overlay_path = args.out_dir.join("overlay.jpg");
    overlay.save_with_format(&overlay_path, image::ImageFormat::Jpeg)?;

    let overlay_cropped_path = args.out_dir.join("overlay_cropped.jpg");
    let overlay_cropped_written = if let (Some(refined), Some(applied)) =
        (refine.detection_refined, refine.rotation_applied_deg)
    {
        let theta = applied.to_radians();
        let rotated_rgb = rotate_rgb_about_center(&base_rgb, theta);
        save_norm_crop(&rotated_rgb, refined.inner, &overlay_cropped_path)?;
        true
    } else {
        let _ = std::fs::remove_file(&overlay_cropped_path);
        false
    };
    let overlay_json = OverlayJson {
        outer: refine.detection_initial.outer,
        inner: refine.detection_initial.inner,
        rotated_inner: if let (Some(refined), Some(applied)) =
            (refine.detection_refined, refine.rotation_applied_deg)
        {
            Some(refined_inner_backproject_norm(
                refined.inner,
                applied.to_radians(),
                overlay.width() as f32,
                overlay.height() as f32,
            ))
        } else {
            None
        },
    };
    let overlay_json_path = args.out_dir.join("overlay.json");
    std::fs::write(
        &overlay_json_path,
        serde_json::to_string_pretty(&overlay_json)?,
    )?;

    let out = Step02Result {
        raw: handoff.raw,
        prepared: handoff.prepared,
        preprocess: handoff.preprocess,
        transform: handoff.transform,
        detection_initial: refine.detection_initial,
        detection_refined: refine.detection_refined,
        detection: refine.detection,
        rotation_estimate: refine.rotation_estimate,
        rotation_applied_deg: refine.rotation_applied_deg,
        rotation_residual_deg: refine.rotation_residual_deg,
    };
    let result_path = args.out_dir.join("result.json");
    std::fs::write(&result_path, serde_json::to_string_pretty(&out)?)?;

    println!("step02 ok");
    println!("overlay initial: {}", overlay_initial_path.display());
    println!("overlay: {}", overlay_path.display());
    if overlay_cropped_written {
        println!("overlay cropped: {}", overlay_cropped_path.display());
    }
    println!("overlay json: {}", overlay_json_path.display());
    println!("result: {}", result_path.display());
    if let Some(est) = refine.rotation_estimate {
        println!(
            "rotation estimate: {:.3} deg (t={:?} b={:?} l={:?} r={:?})",
            est.angle_deg, est.top_deg, est.bottom_deg, est.left_deg, est.right_deg
        );
    }
    if let Some(applied) = refine.rotation_applied_deg {
        println!("rotation applied: {:.3} deg", applied);
    }
    if let Some(res) = refine.rotation_residual_deg {
        println!("rotation residual: {:.3} deg", res);
    }
    if args.dump_debug {
        println!("debug: {}", args.out_dir.display());
        if args.out_dir.join("rotation_decision.jpg").exists() {
            println!(
                "rotation decision: {}",
                args.out_dir.join("rotation_decision.jpg").display()
            );
        }
    }
    Ok(())
}

fn write_debug_artifacts(
    out_dir: &std::path::Path,
    gray: &image::GrayImage,
    dbg: &DetectionDebug,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let dbg_json = out_dir.join("debug.json");
    std::fs::write(&dbg_json, serde_json::to_string_pretty(dbg)?)?;

    let vertical_csv = out_dir.join("vertical_profile.csv");
    let horizontal_csv = out_dir.join("horizontal_profile.csv");
    write_profile_csv(
        &vertical_csv,
        "x",
        &dbg.vertical_profile,
        &dbg.vertical_derivative,
    )?;
    write_profile_csv(
        &horizontal_csv,
        "y",
        &dbg.horizontal_profile,
        &dbg.horizontal_derivative,
    )?;

    let mut vertical_plot = image::RgbImage::new(1200, 420);
    draw_vertical_profile_with_band(
        gray,
        dbg,
        &[
            dbg.outer_left_idx,
            dbg.inner_left_idx,
            dbg.inner_right_idx,
            dbg.outer_right_idx,
        ],
        &mut vertical_plot,
    );
    vertical_plot.save_with_format(
        out_dir.join("vertical_profile_plot.jpg"),
        image::ImageFormat::Jpeg,
    )?;

    let mut horizontal_plot = image::RgbImage::new(1200, 420);
    draw_horizontal_profile_with_band(
        gray,
        dbg,
        &[
            dbg.outer_top_idx,
            dbg.inner_top_idx,
            dbg.inner_bottom_idx,
            dbg.outer_bottom_idx,
        ],
        &mut horizontal_plot,
    );
    horizontal_plot.save_with_format(
        out_dir.join("horizontal_profile_plot.jpg"),
        image::ImageFormat::Jpeg,
    )?;

    Ok(())
}

fn write_rotation_decision_artifacts(
    out_dir: &std::path::Path,
    base_rgb: &RgbImage,
    inner: BoundsNorm,
    dbg: &RotationDebug,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;

    let mut img = base_rgb.clone();
    draw_norm_rect(&mut img, inner, Rgb([120, 120, 50]));

    draw_points(&mut img, &dbg.top_points, Rgb([80, 255, 80]));
    draw_points(&mut img, &dbg.bottom_points, Rgb([80, 255, 255]));
    draw_points(&mut img, &dbg.left_points, Rgb([255, 80, 255]));
    draw_points(&mut img, &dbg.right_points, Rgb([255, 180, 80]));

    let [x1, y1, x2, y2] = dbg.inner_px;
    if let Some([m, b]) = dbg.top_fit {
        draw_line_segment_mut(
            &mut img,
            (x1 as f32, m * x1 as f32 + b),
            (x2 as f32, m * x2 as f32 + b),
            Rgb([0, 255, 0]),
        );
    }
    if let Some([m, b]) = dbg.bottom_fit {
        draw_line_segment_mut(
            &mut img,
            (x1 as f32, m * x1 as f32 + b),
            (x2 as f32, m * x2 as f32 + b),
            Rgb([0, 220, 220]),
        );
    }
    if let Some([m, b]) = dbg.left_fit {
        draw_line_segment_mut(
            &mut img,
            (m * y1 as f32 + b, y1 as f32),
            (m * y2 as f32 + b, y2 as f32),
            Rgb([220, 0, 220]),
        );
    }
    if let Some([m, b]) = dbg.right_fit {
        draw_line_segment_mut(
            &mut img,
            (m * y1 as f32 + b, y1 as f32),
            (m * y2 as f32 + b, y2 as f32),
            Rgb([255, 140, 0]),
        );
    }

    img.save_with_format(
        out_dir.join("rotation_decision.jpg"),
        image::ImageFormat::Jpeg,
    )?;
    std::fs::write(
        out_dir.join("rotation_decision.json"),
        serde_json::to_string_pretty(dbg)?,
    )?;
    Ok(())
}

fn draw_points(img: &mut RgbImage, pts: &[[f32; 2]], color: Rgb<u8>) {
    for p in pts {
        let x = p[0];
        let y = p[1];
        draw_line_segment_mut(img, (x - 1.0, y), (x + 1.0, y), color);
        draw_line_segment_mut(img, (x, y - 1.0), (x, y + 1.0), color);
    }
}

fn write_profile_csv(
    path: &std::path::Path,
    axis: &str,
    profile: &[f32],
    grad: &[f32],
) -> Result<()> {
    let mut s = String::new();
    s.push_str(&format!("{axis},profile,grad\\n"));
    let n = profile.len().max(grad.len());
    for i in 0..n {
        let p = profile.get(i).copied().unwrap_or(0.0);
        let g = grad.get(i).copied().unwrap_or(0.0);
        s.push_str(&format!("{i},{p},{g}\\n"));
    }
    std::fs::write(path, s)?;
    Ok(())
}
