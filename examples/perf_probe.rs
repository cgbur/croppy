use std::hint::black_box;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use croppy::detect::{BoundsNorm, detect_bounds};
use croppy::detect_refine::{RotationRefineConfig, run_detection_with_rotation_refine};
use croppy::preprocess::{
    PreprocessConfig, prepare_image, resize_gray_max_edge, resize_rgb_max_edge,
};
use image::{GrayImage, Luma, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use imageproc::rect::Rect;

const DEFAULT_WIDTH: u32 = 2400;
const DEFAULT_HEIGHT: u32 = 1600;
const DEFAULT_SYNTH_ROTATION_DEG: f32 = 1.35;
const DEFAULT_MAX_EDGE: u32 = 1000;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    PrepareImage,
    DetectRefine,
    DetectBounds,
    EstimateRotation,
    RotateGray,
    Resize,
    Pipeline,
}

#[derive(Parser, Debug)]
#[command(about = "Single-purpose perf probe runner for hot image stages")]
struct Args {
    #[arg(value_enum)]
    mode: Mode,

    #[arg(long, default_value_t = 100)]
    iters: usize,

    #[arg(long, default_value_t = 5)]
    warmup_iters: usize,

    #[arg(long, default_value_t = DEFAULT_WIDTH)]
    width: u32,

    #[arg(long, default_value_t = DEFAULT_HEIGHT)]
    height: u32,

    #[arg(long, default_value_t = DEFAULT_SYNTH_ROTATION_DEG)]
    synthetic_rotation_deg: f32,

    #[arg(long, default_value_t = DEFAULT_MAX_EDGE)]
    max_edge: u32,
}

#[derive(Clone)]
struct ProbeFixture {
    source_rgb: RgbImage,
    resized_gray: GrayImage,
    prepared_gray: GrayImage,
    detected_inner: BoundsNorm,
}

fn preprocess_cfg() -> PreprocessConfig {
    PreprocessConfig {
        invert: true,
        flip_180: true,
        black_pct: 2.0,
        white_pct: 80.0,
        knee_pct: 2.0,
    }
}

fn prepare_for_detection(decoded_rgb: &RgbImage, max_edge: u32) -> GrayImage {
    let gray = image::imageops::grayscale(decoded_rgb);
    let gray = resize_gray_max_edge(gray, max_edge);
    prepare_image(gray, preprocess_cfg())
}

fn build_fixture(args: &Args) -> ProbeFixture {
    let source_rgb =
        generate_synthetic_source(args.width, args.height, args.synthetic_rotation_deg);
    let resized_rgb = resize_rgb_max_edge(&source_rgb, args.max_edge);
    let resized_gray = image::imageops::grayscale(&resized_rgb);
    let prepared_gray = prepare_image(resized_gray.clone(), preprocess_cfg());
    let detected_inner = detect_bounds(&prepared_gray)
        .expect("synthetic probe fixture failed detect_bounds; adjust generator params")
        .inner;

    ProbeFixture {
        source_rgb,
        resized_gray,
        prepared_gray,
        detected_inner,
    }
}

fn generate_synthetic_source(width: u32, height: u32, rotation_deg: f32) -> RgbImage {
    let mut gray = GrayImage::from_pixel(width, height, Luma([226]));

    let margin_x = ((width as f32) * 0.12).round() as u32;
    let margin_y = ((height as f32) * 0.14).round() as u32;
    let inner_w = width.saturating_sub(margin_x.saturating_mul(2)).max(32);
    let inner_h = height.saturating_sub(margin_y.saturating_mul(2)).max(32);
    draw_filled_rect_mut(
        &mut gray,
        Rect::at(margin_x as i32, margin_y as i32).of_size(inner_w, inner_h),
        Luma([24]),
    );

    let rebate_x = margin_x.saturating_sub(8);
    let rebate_y = margin_y.saturating_sub(8);
    let rebate_w = inner_w
        .saturating_add(16)
        .min(width.saturating_sub(rebate_x));
    let rebate_h = inner_h
        .saturating_add(16)
        .min(height.saturating_sub(rebate_y));
    draw_filled_rect_mut(
        &mut gray,
        Rect::at(rebate_x as i32, rebate_y as i32).of_size(rebate_w, rebate_h),
        Luma([58]),
    );
    draw_filled_rect_mut(
        &mut gray,
        Rect::at(margin_x as i32, margin_y as i32).of_size(inner_w, inner_h),
        Luma([24]),
    );

    let rotated = rotate_about_center(
        &gray,
        rotation_deg.to_radians(),
        Interpolation::Bilinear,
        Luma([226]),
    );

    let mut textured = rotated;
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;
    for (x, y, px) in textured.enumerate_pixels_mut() {
        let base = px[0] as i16;
        let noise = ((((x as u64) * 37) ^ ((y as u64) * 101) ^ (((x as u64) * (y as u64)) * 17))
            & 0x0f) as i16
            - 8;
        let banding = (((y / 9) % 5) as i16) - 2;
        let dx = ((x as f32) - cx).abs() / cx.max(1.0);
        let dy = ((y as f32) - cy).abs() / cy.max(1.0);
        let vignette = ((dx * dx + dy * dy) * 14.0).round() as i16;
        let value = (base + noise + banding - vignette).clamp(0, 255) as u8;
        px[0] = value;
    }

    image::DynamicImage::ImageLuma8(textured).to_rgb8()
}

fn main() {
    let args = Args::parse();
    let fixture = build_fixture(&args);
    let refine_cfg = RotationRefineConfig::default();

    for _ in 0..args.warmup_iters {
        run_mode_once(args.mode, &fixture, args.max_edge, refine_cfg);
    }

    let start = Instant::now();
    let mut sink = 0u64;
    for _ in 0..args.iters {
        sink ^= run_mode_once(args.mode, &fixture, args.max_edge, refine_cfg);
    }
    let elapsed = start.elapsed();
    let per_iter_ns = elapsed.as_nanos() as f64 / args.iters.max(1) as f64;

    println!("mode: {:?}", args.mode);
    println!("iters: {}", args.iters);
    println!("warmup_iters: {}", args.warmup_iters);
    println!("elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("per_iter_us: {:.3}", per_iter_ns / 1000.0);
    println!("sink: {sink}");
}

fn run_mode_once(
    mode: Mode,
    fixture: &ProbeFixture,
    max_edge: u32,
    refine_cfg: RotationRefineConfig,
) -> u64 {
    match mode {
        Mode::PrepareImage => {
            let out = prepare_image(
                black_box(fixture.resized_gray.clone()),
                black_box(preprocess_cfg()),
            );
            u64::from(black_box(out.get_pixel(0, 0)[0]))
        }
        Mode::DetectRefine => {
            let out = run_detection_with_rotation_refine(
                black_box(&fixture.prepared_gray),
                black_box(refine_cfg),
            )
            .expect("detect_refine should succeed for fixture");
            u64::from(black_box(out.detection.inner.left.to_bits()))
        }
        Mode::DetectBounds => {
            let out = detect_bounds(black_box(&fixture.prepared_gray))
                .expect("detect_bounds should succeed for fixture");
            u64::from(black_box(out.inner.left.to_bits()))
        }
        Mode::EstimateRotation => {
            let out = croppy::detect_refine::estimate_rotation_from_inner(
                black_box(&fixture.prepared_gray),
                black_box(fixture.detected_inner),
            )
            .expect("estimate_rotation should succeed for fixture");
            u64::from(black_box(out.angle_deg.to_bits()))
        }
        Mode::RotateGray => {
            let out = rotate_about_center(
                black_box(&fixture.prepared_gray),
                black_box((-1.35f32).to_radians()),
                Interpolation::Bilinear,
                Luma([0u8]),
            );
            u64::from(black_box(out.get_pixel(0, 0)[0]))
        }
        Mode::Resize => {
            let out = resize_rgb_max_edge(black_box(&fixture.source_rgb), black_box(max_edge));
            u64::from(black_box(out.get_pixel(0, 0)[0]))
        }
        Mode::Pipeline => {
            let prepared_gray =
                prepare_for_detection(black_box(&fixture.source_rgb), black_box(max_edge));
            let out = run_detection_with_rotation_refine(
                black_box(&prepared_gray),
                black_box(refine_cfg),
            )
            .expect("pipeline detect_refine should succeed for fixture");
            u64::from(black_box(out.detection.inner.left.to_bits()))
        }
    }
}
