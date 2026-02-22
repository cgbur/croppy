use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow};
use clap::Parser;
use croppy::discover::{is_supported_raw, list_raw_files};
use croppy::handoff::{Step01Handoff, Step01Transform, write_handoff};
use croppy::preprocess::{PreprocessConfig, prepare_image, resize_rgb_max_edge};
use croppy::raw::{decode_raw_to_rgb, rgb_to_gray};
use image::ImageFormat;
use rand::prelude::IndexedRandom;

#[derive(Parser, Debug)]
#[command(about = "Step 01: pick raw, render, invert/levels/flip, write minimal handoff")]
struct Args {
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    recursive: bool,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long)]
    raw: Option<PathBuf>,

    #[arg(long, default_value = "tmp/step01")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = true)]
    invert: bool,

    #[arg(long, default_value_t = true)]
    flip_180: bool,

    #[arg(long, default_value_t = 2.0)]
    black_pct: f32,

    #[arg(long, default_value_t = 80.0)]
    white_pct: f32,

    #[arg(long, default_value_t = 2.0)]
    knee_pct: f32,

    #[arg(long, default_value_t = false)]
    keep_color: bool,

    #[arg(long, default_value_t = 1000)]
    max_edge: u32,
}

fn main() -> Result<()> {
    let t0 = Instant::now();
    let args = Args::parse();
    if !args.input.exists() {
        return Err(anyhow!(
            "input path does not exist: {}",
            args.input.display()
        ));
    }
    if !(0.0 <= args.black_pct && args.black_pct < args.white_pct && args.white_pct <= 100.0) {
        return Err(anyhow!("require 0 <= black_pct < white_pct <= 100"));
    }
    if !(0.0 <= args.knee_pct && args.knee_pct < 50.0) {
        return Err(anyhow!("require 0 <= knee_pct < 50"));
    }

    let t_select = Instant::now();
    let selected = select_raw(&args)?;
    let dt_select = t_select.elapsed();
    fs::create_dir_all(&args.out_dir)?;

    println!("selected raw: {}", selected.display());
    let t_decode = Instant::now();
    let rgb_full = decode_raw_to_rgb(&selected)?;
    let dt_decode = t_decode.elapsed();
    let (raw_w, raw_h) = rgb_full.dimensions();
    let t_resize = Instant::now();
    let rgb = resize_rgb_max_edge(&rgb_full, args.max_edge);
    let dt_resize = t_resize.elapsed();
    let (w, h) = rgb.dimensions();
    println!(
        "prepared resolution: {}x{} (max_edge={})",
        w, h, args.max_edge
    );
    if args.keep_color {
        let t_write_color = Instant::now();
        let color_path = args.out_dir.join("prepared_color.jpg");
        rgb.save_with_format(&color_path, ImageFormat::Jpeg)?;
        println!("wrote color: {}", color_path.display());
        println!(
            "timing: write prepared_color.jpg = {} ms",
            t_write_color.elapsed().as_millis()
        );
    }

    let t_preprocess = Instant::now();
    let gray = rgb_to_gray(&rgb);
    let prepared = prepare_image(
        gray,
        PreprocessConfig {
            invert: args.invert,
            flip_180: args.flip_180,
            black_pct: args.black_pct,
            white_pct: args.white_pct,
            knee_pct: args.knee_pct,
        },
    );
    let dt_preprocess = t_preprocess.elapsed();

    let t_write_outputs = Instant::now();
    let prepared_path = args.out_dir.join("prepared.jpg");
    prepared.save_with_format(&prepared_path, ImageFormat::Jpeg)?;
    fs::write(
        args.out_dir.join("selected.raw.txt"),
        format!("{}\n", selected.display()),
    )?;

    let mut tags = vec!["bw"]; // grayscale path
    if args.invert {
        tags.push("invert");
    }
    tags.push("levels");
    if args.knee_pct > 0.0 {
        tags.push("softknee");
    }
    if args.flip_180 {
        tags.push("flip180");
    }

    let handoff = Step01Handoff {
        raw: selected.to_string_lossy().to_string(),
        prepared: prepared_path.to_string_lossy().to_string(),
        preprocess: tags.join("+"),
        transform: Step01Transform {
            raw_width: raw_w,
            raw_height: raw_h,
            prepared_width: w,
            prepared_height: h,
            flip_180: args.flip_180,
        },
    };
    let handoff_path = args.out_dir.join("next.json");
    write_handoff(&handoff_path, &handoff)?;
    let dt_write_outputs = t_write_outputs.elapsed();

    println!("wrote prepared: {}", prepared_path.display());
    println!("wrote handoff: {}", handoff_path.display());
    println!("timing: select_raw = {} ms", dt_select.as_millis());
    println!("timing: decode_raw_to_rgb = {} ms", dt_decode.as_millis());
    println!("timing: resize_rgb_max_edge = {} ms", dt_resize.as_millis());
    println!(
        "timing: grayscale+prepare_image = {} ms",
        dt_preprocess.as_millis()
    );
    println!(
        "timing: write outputs = {} ms",
        dt_write_outputs.as_millis()
    );
    println!(
        "timing: total step01_prepare = {} ms",
        t0.elapsed().as_millis()
    );
    Ok(())
}

fn select_raw(args: &Args) -> Result<PathBuf> {
    if let Some(path) = &args.raw {
        if !path.exists() {
            return Err(anyhow!("--raw does not exist: {}", path.display()));
        }
        if !is_supported_raw(path) {
            return Err(anyhow!("unsupported raw extension: {}", path.display()));
        }
        return Ok(path.clone());
    }

    let raws = list_raw_files(&args.input, args.recursive)?;
    if raws.is_empty() {
        return Err(anyhow!("no raw files found under {}", args.input.display()));
    }

    if let Some(seed) = args.seed {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        return raws
            .choose(&mut rng)
            .cloned()
            .ok_or_else(|| anyhow!("failed selecting raw file"));
    }

    let mut rng = rand::rng();
    raws.choose(&mut rng)
        .cloned()
        .ok_or_else(|| anyhow!("failed selecting raw file"))
}
