use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use opencv::core::{self, Mat, Point, Point2f, Scalar, Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use rawloader::RawImageData;
use rusqlite::Connection;
use serde::Serialize;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(name = "croppy")]
#[command(about = "Detect crop/rotation candidates for scanned film RAW files")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Detect(DetectArgs),
}

#[derive(Parser, Debug)]
struct DetectArgs {
    /// Folder containing RAW files.
    input: PathBuf,

    /// Include subdirectories.
    #[arg(long, default_value_t = false)]
    recursive: bool,

    /// Optional Lightroom catalog (.lrcat) for skip filtering.
    #[arg(long)]
    lrcat: Option<PathBuf>,

    /// Include files with existing edits (XMP sidecar or crop keys in catalog).
    #[arg(long, default_value_t = false)]
    include_modified: bool,

    /// Limit number of files processed after filtering.
    #[arg(long)]
    max_files: Option<usize>,

    /// Max preview dimension (long edge).
    #[arg(long, default_value_t = 2400)]
    downscale_max: i32,

    /// Directory for overlay previews. If omitted, previews are not written.
    #[arg(long)]
    preview_dir: Option<PathBuf>,

    /// Output JSON report path.
    #[arg(long, default_value = "croppy-detect-report.json")]
    out: PathBuf,
}

#[derive(Debug, Serialize)]
struct DetectReport {
    schema_version: String,
    source_path: String,
    total_files_seen: usize,
    total_processed: usize,
    total_skipped: usize,
    items: Vec<DetectItem>,
}

#[derive(Debug, Serialize)]
struct DetectItem {
    file: String,
    path: String,
    status: String,
    skip_reason: Option<String>,
    detection: Option<CropCandidate>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CropCandidate {
    crop_left: f32,
    crop_top: f32,
    crop_right: f32,
    crop_bottom: f32,
    crop_angle: f32,
    confidence: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Detect(args) => run_detect(args),
    }
}

fn run_detect(args: DetectArgs) -> Result<()> {
    if !args.input.exists() {
        return Err(anyhow!(
            "Input path does not exist: {}",
            args.input.display()
        ));
    }
    if let Some(dir) = &args.preview_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("Failed creating preview dir: {}", dir.display()))?;
    }

    let raw_files = discover_raw_files(&args.input, args.recursive)?;
    let lrcat_edited = if args.include_modified {
        HashSet::new()
    } else if let Some(catalog) = &args.lrcat {
        load_catalog_cropped_paths(catalog)?
    } else {
        HashSet::new()
    };

    let mut items = Vec::new();
    let mut processed = 0usize;
    let mut skipped = 0usize;

    for raw_path in raw_files {
        if let Some(max) = args.max_files {
            if processed >= max {
                break;
            }
        }

        let file_name = raw_path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("<unknown>")
            .to_string();
        let path_str = raw_path.to_string_lossy().to_string();

        if !args.include_modified {
            if has_xmp_sidecar(&raw_path) {
                skipped += 1;
                items.push(DetectItem {
                    file: file_name,
                    path: path_str,
                    status: "skipped".to_string(),
                    skip_reason: Some("existing_xmp_sidecar".to_string()),
                    detection: None,
                    error: None,
                });
                continue;
            }
            if !lrcat_edited.is_empty() && lrcat_edited.contains(&normalize_path_key(&raw_path)) {
                skipped += 1;
                items.push(DetectItem {
                    file: file_name,
                    path: path_str,
                    status: "skipped".to_string(),
                    skip_reason: Some("existing_catalog_crop".to_string()),
                    detection: None,
                    error: None,
                });
                continue;
            }
        }

        match process_one_raw(&raw_path, args.downscale_max) {
            Ok((candidate, preview_mat)) => {
                processed += 1;
                if let Some(dir) = &args.preview_dir {
                    let mut preview_path = dir.join(
                        raw_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("preview"),
                    );
                    preview_path.set_extension("jpg");
                    write_preview(&preview_path, &preview_mat)?;
                }
                items.push(DetectItem {
                    file: file_name,
                    path: path_str,
                    status: "processed".to_string(),
                    skip_reason: None,
                    detection: Some(candidate),
                    error: None,
                });
            }
            Err(err) => {
                items.push(DetectItem {
                    file: file_name,
                    path: path_str,
                    status: "error".to_string(),
                    skip_reason: None,
                    detection: None,
                    error: Some(format!("{err:#}")),
                });
            }
        }
    }

    let report = DetectReport {
        schema_version: "1".to_string(),
        source_path: args.input.to_string_lossy().to_string(),
        total_files_seen: items.len(),
        total_processed: processed,
        total_skipped: skipped,
        items,
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&args.out, json)
        .with_context(|| format!("Failed writing report: {}", args.out.display()))?;
    println!(
        "Done. processed={}, skipped={}, report={}",
        report.total_processed,
        report.total_skipped,
        args.out.display()
    );
    Ok(())
}

fn discover_raw_files(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let exts = [
        "arw", "cr2", "cr3", "nef", "nrw", "raf", "orf", "rw2", "dng", "pef",
    ];
    let mut out = Vec::new();
    let mut walker = WalkDir::new(input).follow_links(false);
    if !recursive {
        walker = walker.max_depth(1);
    }
    for entry in walker {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if exts.contains(&ext.as_str()) {
            out.push(path.to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

fn has_xmp_sidecar(raw_path: &Path) -> bool {
    let stem = match raw_path.file_stem().and_then(|s| s.to_str()) {
        Some(v) => v,
        None => return false,
    };
    let parent = match raw_path.parent() {
        Some(v) => v,
        None => return false,
    };
    parent.join(format!("{stem}.xmp")).exists() || parent.join(format!("{stem}.XMP")).exists()
}

fn load_catalog_cropped_paths(catalog_path: &Path) -> Result<HashSet<String>> {
    let conn =
        Connection::open_with_flags(catalog_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| {
                format!(
                    "Failed opening catalog read-only: {}",
                    catalog_path.display()
                )
            })?;

    let sql = r#"
select
  replace(r.absolutePath || fo.pathFromRoot || f.baseName || '.' || f.extension, '\', '/')
from Adobe_images i
join AgLibraryFile f on f.id_local=i.rootFile
join AgLibraryFolder fo on fo.id_local=f.folder
join AgLibraryRootFolder r on r.id_local=fo.rootFolder
join Adobe_imageDevelopSettings ds on ds.image=i.id_local
where ds.text like '%CropLeft%'
   or ds.text like '%CropRight%'
   or ds.text like '%CropTop%'
   or ds.text like '%CropBottom%'
   or ds.text like '%CropAngle%'
"#;

    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([])?;
    let mut keys = HashSet::new();
    while let Some(row) = rows.next()? {
        let p: String = row.get(0)?;
        keys.insert(p.to_ascii_lowercase());
    }
    Ok(keys)
}

fn normalize_path_key(path: &Path) -> String {
    let raw = path.to_string_lossy().replace('\\', "/");
    let parts: Vec<&str> = raw.split('/').collect();
    if parts.len() >= 4 && parts[1] == "mnt" && parts[2].len() == 1 {
        let drive = parts[2].to_ascii_uppercase();
        let rest = parts[3..].join("/");
        return format!("{drive}:/{rest}").to_ascii_lowercase();
    }
    raw.to_ascii_lowercase()
}

fn process_one_raw(path: &Path, downscale_max: i32) -> Result<(CropCandidate, Mat)> {
    let decoded = rawloader::decode_file(path)
        .map_err(|e| anyhow!("raw decode failed for {}: {e}", path.display()))?;
    let gray = raw_to_gray_mat(&decoded.data, decoded.width, decoded.height)?;
    let gray = resize_long_edge(&gray, downscale_max)?;

    let mut blur = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blur,
        Size::new(5, 5),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut binary = Mat::default();
    imgproc::threshold(
        &blur,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    let kernel =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(5, 5), Point::new(-1, -1))?;
    let mut morph = Mat::default();
    imgproc::morphology_ex(
        &binary,
        &mut morph,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &morph,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;
    if contours.is_empty() {
        return Err(anyhow!("no contours found"));
    }

    let mut best_idx = 0;
    let mut best_area = 0.0f64;
    for (i, c) in contours.iter().enumerate() {
        let area = imgproc::contour_area(&c, false)?;
        if area > best_area {
            best_area = area;
            best_idx = i;
        }
    }
    let best = contours.get(best_idx)?;
    let rect = imgproc::min_area_rect(&best)?;
    let size = gray.size()?;

    let rect_center = rect.center;
    let rect_w = rect.size.width.max(1.0);
    let rect_h = rect.size.height.max(1.0);
    let left = ((rect_center.x - rect_w / 2.0) / size.width as f32).clamp(0.0, 1.0);
    let right = ((rect_center.x + rect_w / 2.0) / size.width as f32).clamp(0.0, 1.0);
    let top = ((rect_center.y - rect_h / 2.0) / size.height as f32).clamp(0.0, 1.0);
    let bottom = ((rect_center.y + rect_h / 2.0) / size.height as f32).clamp(0.0, 1.0);
    let area_ratio = (rect_w * rect_h) as f64 / (size.width * size.height) as f64;
    let confidence = area_ratio.clamp(0.0, 1.0) as f32;

    let candidate = CropCandidate {
        crop_left: left,
        crop_top: top,
        crop_right: right,
        crop_bottom: bottom,
        crop_angle: normalize_angle(rect.angle),
        confidence,
    };

    let mut preview = Mat::default();
    imgproc::cvt_color(
        &gray,
        &mut preview,
        imgproc::COLOR_GRAY2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    draw_rotated_rect(&mut preview, rect)?;
    let label = format!(
        "L {:.4} T {:.4} R {:.4} B {:.4} A {:.3}",
        candidate.crop_left,
        candidate.crop_top,
        candidate.crop_right,
        candidate.crop_bottom,
        candidate.crop_angle
    );
    imgproc::put_text(
        &mut preview,
        &label,
        Point::new(20, 40),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.9,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;
    Ok((candidate, preview))
}

fn normalize_angle(angle: f32) -> f32 {
    if angle > 45.0 { angle - 90.0 } else { angle }
}

fn draw_rotated_rect(img: &mut Mat, rect: core::RotatedRect) -> Result<()> {
    let mut pts = [Point2f::default(); 4];
    rect.points(&mut pts)?;
    for i in 0..4 {
        let p1 = Point::new(pts[i].x.round() as i32, pts[i].y.round() as i32);
        let p2 = Point::new(
            pts[(i + 1) % 4].x.round() as i32,
            pts[(i + 1) % 4].y.round() as i32,
        );
        imgproc::line(
            img,
            p1,
            p2,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            3,
            imgproc::LINE_AA,
            0,
        )?;
    }
    Ok(())
}

fn write_preview(path: &Path, mat: &Mat) -> Result<()> {
    let params: Vector<i32> = Vector::new();
    let ok = imgcodecs::imwrite(path.to_string_lossy().as_ref(), mat, &params)?;
    if !ok {
        return Err(anyhow!("OpenCV imwrite failed for {}", path.display()));
    }
    Ok(())
}

fn raw_to_gray_mat(data: &RawImageData, width: usize, height: usize) -> Result<Mat> {
    let pixels: Vec<u8> = match data {
        RawImageData::Integer(v) => {
            let max_v = v.iter().copied().max().unwrap_or(1) as f32;
            v.iter()
                .map(|px| ((*px as f32 / max_v) * 255.0).round().clamp(0.0, 255.0) as u8)
                .collect()
        }
        RawImageData::Float(v) => {
            let max_v = v
                .iter()
                .copied()
                .fold(0.0f32, |a, b| if b > a { b } else { a })
                .max(1.0);
            v.iter()
                .map(|px| ((*px / max_v) * 255.0).round().clamp(0.0, 255.0) as u8)
                .collect()
        }
    };
    if pixels.len() != width * height {
        return Err(anyhow!(
            "unexpected pixel count {} for {}x{}",
            pixels.len(),
            width,
            height
        ));
    }
    let mat = Mat::from_slice(&pixels)?;
    let reshaped = mat.reshape(1, height as i32)?;
    Ok(reshaped.try_clone()?)
}

fn resize_long_edge(src: &Mat, max_dim: i32) -> Result<Mat> {
    let size = src.size()?;
    let w = size.width;
    let h = size.height;
    let long = w.max(h);
    if long <= max_dim {
        return Ok(src.try_clone()?);
    }
    let scale = max_dim as f64 / long as f64;
    let new_w = (w as f64 * scale).round() as i32;
    let new_h = (h as f64 * scale).round() as i32;
    let mut dst = Mat::default();
    imgproc::resize(
        src,
        &mut dst,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    Ok(dst)
}
