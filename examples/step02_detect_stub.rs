use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use clap::Parser;
use croppy::detect::{
    BoundsNorm, Detection, DetectionDebug, detect_bounds_with_debug,
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

    #[arg(long)]
    out_dir: Option<PathBuf>,

    #[arg(long, default_value_t = true)]
    dump_debug: bool,

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
    inner: BoundsNorm,
    rotated_inner: Option<[[f32; 2]; 4]>,
}

#[derive(Debug, Serialize)]
struct Step02Timing {
    handoff_read_ms: u128,
    image_open_ms: u128,
    detect_refine_ms: u128,
    detect_debug_ms: u128,
    write_debug_ms: u128,
    write_rotation_debug_ms: u128,
    write_overlay_initial_ms: u128,
    write_overlay_ms: u128,
    write_overlay_cropped_ms: u128,
    write_overlay_json_ms: u128,
    write_result_ms: u128,
    total_ms: u128,
}

fn main() -> Result<()> {
    let t_total = Instant::now();
    let args = Args::parse();
    if !args.input.exists() {
        return Err(anyhow!("handoff json not found: {}", args.input.display()));
    }
    let out_dir = args
        .out_dir
        .clone()
        .unwrap_or_else(|| default_out_dir(&args.input));

    let t_handoff = Instant::now();
    let handoff = read_handoff(&args.input)?;
    let dt_handoff = t_handoff.elapsed();
    let prepared = PathBuf::from(&handoff.prepared);
    if !prepared.exists() {
        return Err(anyhow!("prepared image missing: {}", prepared.display()));
    }

    std::fs::create_dir_all(&out_dir)?;
    let t_open = Instant::now();
    let dyn_img = image::open(&prepared)?;
    let gray = dyn_img.to_luma8();
    let base_rgb = dyn_img.to_rgb8();
    let dt_open = t_open.elapsed();

    let refine_cfg = RotationRefineConfig {
        refine_rotation: args.refine_rotation,
        apply_rotation_decision: args.apply_rotation_decision,
        max_refine_abs_deg: args.max_refine_abs_deg,
    };

    let t_detect_refine = Instant::now();
    let Some(refine) = run_detection_with_rotation_refine(&gray, refine_cfg) else {
        return Err(anyhow!("failed detecting boundaries"));
    };
    let dt_detect_refine = t_detect_refine.elapsed();
    let t_detect_debug = Instant::now();
    let Some((_, dbg)) = detect_bounds_with_debug(&gray) else {
        return Err(anyhow!("failed collecting debug vectors"));
    };
    let dt_detect_debug = t_detect_debug.elapsed();

    let mut dt_write_debug = Duration::default();
    let mut dt_write_rotation_debug = Duration::default();

    if args.dump_debug {
        let t_write_debug = Instant::now();
        write_debug_artifacts(&out_dir, &gray, &dbg)?;
        dt_write_debug = t_write_debug.elapsed();
        if let Some(d) = &refine.rotation_initial_debug {
            let t_write_rotation_debug = Instant::now();
            write_rotation_decision_artifacts(
                &out_dir,
                &base_rgb,
                refine.detection_initial.inner,
                d,
            )?;
            dt_write_rotation_debug = t_write_rotation_debug.elapsed();
        }
    }

    let mut overlay_initial = base_rgb.clone();
    draw_norm_rect(
        &mut overlay_initial,
        refine.detection_initial.inner,
        Rgb([255, 255, 0]),
    );
    let overlay_initial_path = out_dir.join("overlay_initial.jpg");
    let t_overlay_initial = Instant::now();
    overlay_initial.save_with_format(&overlay_initial_path, image::ImageFormat::Jpeg)?;
    let dt_overlay_initial = t_overlay_initial.elapsed();

    let mut overlay = base_rgb.clone();
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
    let overlay_path = out_dir.join("overlay.jpg");
    let t_overlay = Instant::now();
    overlay.save_with_format(&overlay_path, image::ImageFormat::Jpeg)?;
    let dt_overlay = t_overlay.elapsed();

    let overlay_cropped_path = out_dir.join("overlay_cropped.jpg");
    let t_overlay_cropped = Instant::now();
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
    let dt_overlay_cropped = t_overlay_cropped.elapsed();
    let overlay_json = OverlayJson {
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
    let overlay_json_path = out_dir.join("overlay.json");
    let t_overlay_json = Instant::now();
    std::fs::write(
        &overlay_json_path,
        serde_json::to_string_pretty(&overlay_json)?,
    )?;
    let dt_overlay_json = t_overlay_json.elapsed();

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
    let result_path = out_dir.join("result.json");
    let t_result = Instant::now();
    std::fs::write(&result_path, serde_json::to_string_pretty(&out)?)?;
    let dt_result = t_result.elapsed();

    let timing = Step02Timing {
        handoff_read_ms: dt_handoff.as_millis(),
        image_open_ms: dt_open.as_millis(),
        detect_refine_ms: dt_detect_refine.as_millis(),
        detect_debug_ms: dt_detect_debug.as_millis(),
        write_debug_ms: dt_write_debug.as_millis(),
        write_rotation_debug_ms: dt_write_rotation_debug.as_millis(),
        write_overlay_initial_ms: dt_overlay_initial.as_millis(),
        write_overlay_ms: dt_overlay.as_millis(),
        write_overlay_cropped_ms: dt_overlay_cropped.as_millis(),
        write_overlay_json_ms: dt_overlay_json.as_millis(),
        write_result_ms: dt_result.as_millis(),
        total_ms: t_total.elapsed().as_millis(),
    };
    let timing_path = out_dir.join("timing.json");
    std::fs::write(&timing_path, serde_json::to_string_pretty(&timing)?)?;

    println!("step02 ok");
    println!("overlay initial: {}", overlay_initial_path.display());
    println!("overlay: {}", overlay_path.display());
    if overlay_cropped_written {
        println!("overlay cropped: {}", overlay_cropped_path.display());
    }
    println!("overlay json: {}", overlay_json_path.display());
    println!("result: {}", result_path.display());
    println!("timing json: {}", timing_path.display());
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
        println!("debug: {}", out_dir.display());
        if out_dir.join("rotation_decision.jpg").exists() {
            println!(
                "rotation decision: {}",
                out_dir.join("rotation_decision.jpg").display()
            );
        }
    }
    println!("timing: handoff_read = {} ms", timing.handoff_read_ms);
    println!("timing: image_open = {} ms", timing.image_open_ms);
    println!("timing: detect_refine = {} ms", timing.detect_refine_ms);
    println!("timing: detect_debug = {} ms", timing.detect_debug_ms);
    println!("timing: write_debug = {} ms", timing.write_debug_ms);
    println!(
        "timing: write_rotation_debug = {} ms",
        timing.write_rotation_debug_ms
    );
    println!(
        "timing: write_overlay_initial = {} ms",
        timing.write_overlay_initial_ms
    );
    println!("timing: write_overlay = {} ms", timing.write_overlay_ms);
    println!(
        "timing: write_overlay_cropped = {} ms",
        timing.write_overlay_cropped_ms
    );
    println!(
        "timing: write_overlay_json = {} ms",
        timing.write_overlay_json_ms
    );
    println!("timing: write_result = {} ms", timing.write_result_ms);
    println!("timing: total_step02 = {} ms", timing.total_ms);
    Ok(())
}

fn default_out_dir(input_handoff: &std::path::Path) -> PathBuf {
    input_handoff
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
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
        &[dbg.inner_left_idx, dbg.inner_right_idx],
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
        &[dbg.inner_top_idx, dbg.inner_bottom_idx],
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
