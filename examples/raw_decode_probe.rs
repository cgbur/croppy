use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow};
use clap::Parser;
use croppy::raw::{decode_raw_to_rgb_with_hint, extract_preferred_jpeg_thumbnail};
use image::ImageFormat;

#[derive(Parser, Debug)]
#[command(about = "Probe RAW thumbnail-first decode behavior for one file")]
struct Args {
    raw: PathBuf,

    #[arg(long, default_value_t = 1000)]
    max_edge: u32,

    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.raw.exists() {
        return Err(anyhow!("raw path does not exist: {}", args.raw.display()));
    }

    let t_thumb = Instant::now();
    let thumb = extract_preferred_jpeg_thumbnail(&args.raw, args.max_edge)?;
    let dt_thumb = t_thumb.elapsed();
    match &thumb {
        Some(t) => {
            println!(
                "thumb: found idx={}/{} {}x{} ({} bytes)",
                t.index,
                t.total,
                t.width,
                t.height,
                t.jpeg.len()
            );
        }
        None => println!("thumb: no suitable embedded JPEG found"),
    }
    println!(
        "timing: extract_preferred_jpeg_thumbnail = {} ms",
        dt_thumb.as_millis()
    );

    let t_decode = Instant::now();
    let decoded = decode_raw_to_rgb_with_hint(&args.raw, args.max_edge)?;
    let dt_decode = t_decode.elapsed();
    let (w, h) = decoded.image.dimensions();
    println!("decode source: {}", decoded.source.label());
    if let Some(warning) = &decoded.warning {
        println!("warning: {warning}");
    }
    println!("decoded: {}x{}", w, h);
    println!(
        "timing: decode_raw_to_rgb_with_hint = {} ms",
        dt_decode.as_millis()
    );

    if let Some(out) = args.out {
        decoded.image.save_with_format(&out, ImageFormat::Jpeg)?;
        println!("wrote: {}", out.display());
    }

    Ok(())
}
