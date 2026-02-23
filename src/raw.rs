//! RAW decode helpers and strategy notes.
//!
//! This module is the single place where we decode source RAWs for detection.
//!
//! ## Strategy
//!
//! 1. Prefer embedded JPEG preview extraction via LibRaw `open_file + unpack_thumb_ex`.
//!    This avoids reading the entire RAW into a Rust `Vec<u8>`, which is useful when
//!    files are on slow mounted Windows paths.
//! 2. Pick a "medium" JPEG preview when available (typically large enough for
//!    detection while much faster than full RAW decode).
//! 3. Fall back to half-size RAW demosaic via LibRaw when no suitable JPEG exists.
//!
//! ## Why this exists
//!
//! In this project, the prepared image is further resized (`--max-edge` is usually
//! around 1000), so medium embedded previews are often sufficient. This reduces
//! decode time substantially and avoids full-file userspace buffering on the hot path.
//!
//! ## Important constraints
//!
//! - The current `rsraw` crate API is buffer-based (`open(&[u8])`), so path-based
//!   access uses `rsraw-sys` C API directly.
//! - The vendored `rsraw-sys` build does not currently enable RawSpeed macros.
//! - Fallback RAW decode remains half-size to preserve existing behavior.

use std::ffi::{CStr, CString};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use image::{GrayImage, ImageBuffer, Luma, RgbImage};
use rsraw::{BIT_DEPTH_8, ImageFormat, RawImage};
use rsraw_sys as sys;

const DEFAULT_MAX_EDGE_HINT: u32 = 1000;
const MEDIUM_THUMB_MAX_LONG_EDGE: u32 = 3200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawDecodeSource {
    EmbeddedJpeg,
    RawHalfDemosaic,
}

impl RawDecodeSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::EmbeddedJpeg => "embedded-jpeg",
            Self::RawHalfDemosaic => "raw-half",
        }
    }
}

#[derive(Debug)]
pub struct RawDecodeResult {
    pub image: RgbImage,
    pub source: RawDecodeSource,
    pub warning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EmbeddedJpeg {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub index: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThumbCandidate {
    index: usize,
    width: u32,
    height: u32,
}

pub fn decode_raw_to_rgb(path: &Path) -> Result<RgbImage> {
    Ok(decode_raw_to_rgb_with_hint(path, DEFAULT_MAX_EDGE_HINT)?.image)
}

/// Decode RAW to RGB for detection/prep.
///
/// Attempts:
/// 1. Embedded JPEG thumbnail (medium-first selection).
/// 2. Fallback to half-size RAW demosaic.
///
/// `max_edge_hint` should match the downstream resize target (e.g. CLI `--max-edge`).
pub fn decode_raw_to_rgb_with_hint(path: &Path, max_edge_hint: u32) -> Result<RawDecodeResult> {
    match extract_preferred_jpeg_thumbnail(path, max_edge_hint) {
        Ok(Some(thumb)) => {
            let decoded = image::load_from_memory(&thumb.jpeg).map_err(|e| {
                anyhow!(
                    "embedded jpeg decode failed for {} (thumb #{}/{} {}x{}): {e}",
                    path.display(),
                    thumb.index,
                    thumb.total,
                    thumb.width,
                    thumb.height
                )
            })?;
            Ok(RawDecodeResult {
                image: decoded.to_rgb8(),
                source: RawDecodeSource::EmbeddedJpeg,
                warning: None,
            })
        }
        Ok(None) => {
            let rgb = decode_raw_file_to_rgb(path)?;
            Ok(RawDecodeResult {
                image: rgb,
                source: RawDecodeSource::RawHalfDemosaic,
                warning: Some(format!(
                    "no suitable embedded JPEG for {}; used RAW half-size decode",
                    path.display()
                )),
            })
        }
        Err(thumb_err) => {
            let rgb = decode_raw_file_to_rgb(path)?;
            Ok(RawDecodeResult {
                image: rgb,
                source: RawDecodeSource::RawHalfDemosaic,
                warning: Some(format!(
                    "thumbnail extraction failed for {} ({thumb_err}); used RAW half-size decode",
                    path.display()
                )),
            })
        }
    }
}

/// Extract preferred embedded JPEG without reading the whole RAW into a Rust buffer.
///
/// Returns `Ok(None)` when no JPEG thumbnail is present.
pub fn extract_preferred_jpeg_thumbnail(
    path: &Path,
    max_edge_hint: u32,
) -> Result<Option<EmbeddedJpeg>> {
    let cpath = path_to_cstring(path)?;
    let raw = LibRawHandle::new().context("libraw init failed")?;
    check_libraw(
        unsafe { sys::libraw_open_file(raw.ptr, cpath.as_ptr()) },
        "raw open file",
        path,
    )?;

    let thumbs_list = unsafe { &(*raw.ptr).thumbs_list };
    let thumb_count = thumbs_list.thumbcount.max(0) as usize;
    let list_len = thumbs_list.thumblist.len();
    let mut jpeg_candidates = Vec::new();
    for (idx, item) in thumbs_list
        .thumblist
        .iter()
        .take(thumb_count.min(list_len))
        .enumerate()
    {
        if item.tformat == sys::LibRaw_internal_thumbnail_formats_LIBRAW_INTERNAL_THUMBNAIL_JPEG {
            jpeg_candidates.push(ThumbCandidate {
                index: idx,
                width: u32::from(item.twidth),
                height: u32::from(item.theight),
            });
        }
    }

    let Some(chosen) = choose_preferred_thumbnail(&jpeg_candidates, max_edge_hint) else {
        return Ok(None);
    };

    check_libraw(
        unsafe { sys::libraw_unpack_thumb_ex(raw.ptr, chosen.index as i32) },
        "raw unpack thumbnail",
        path,
    )?;

    let processed = ProcessedImageHandle::make_thumb(raw.ptr, path)?;
    let p = unsafe { &*processed.ptr };
    if p.type_ != sys::LibRaw_image_formats_LIBRAW_IMAGE_JPEG {
        return Ok(None);
    }
    let data = processed.data();

    Ok(Some(EmbeddedJpeg {
        jpeg: data.to_vec(),
        width: chosen.width,
        height: chosen.height,
        index: chosen.index,
        total: thumb_count,
    }))
}

/// Byte-buffer decode path (legacy/compatibility).
pub fn decode_raw_bytes_to_rgb(bytes: &[u8], source: &Path) -> Result<RgbImage> {
    let mut raw = RawImage::open(bytes)
        .map_err(|e| anyhow!("raw decode failed for {}: {e}", source.display()))?;

    raw.unpack()
        .map_err(|e| anyhow!("raw unpack failed for {}: {e}", source.display()))?;
    raw.as_mut().params.half_size = 1;
    let image = raw
        .process::<BIT_DEPTH_8>()
        .map_err(|e| anyhow!("raw process failed for {}: {e}", source.display()))?;

    let data = image.to_vec();
    processed_u8_to_rgb(
        source,
        image.image_format() == ImageFormat::Jpeg,
        image.width(),
        image.height(),
        image.colors() as usize,
        8,
        &data,
    )
}

pub fn rgb_to_gray(img: &RgbImage) -> GrayImage {
    image::DynamicImage::ImageRgb8(img.clone())
        .grayscale()
        .to_luma8()
}

fn decode_raw_file_to_rgb(path: &Path) -> Result<RgbImage> {
    let cpath = path_to_cstring(path)?;
    let raw = LibRawHandle::new().context("libraw init failed")?;
    check_libraw(
        unsafe { sys::libraw_open_file(raw.ptr, cpath.as_ptr()) },
        "raw open file",
        path,
    )?;

    unsafe {
        // Preserve existing behavior: half-size decode.
        (*raw.ptr).params.half_size = 1;
    }

    check_libraw(unsafe { sys::libraw_unpack(raw.ptr) }, "raw unpack", path)?;
    check_libraw(
        unsafe { sys::libraw_dcraw_process(raw.ptr) },
        "raw process",
        path,
    )?;
    let processed = ProcessedImageHandle::make_image(raw.ptr, path)?;
    decode_processed_to_rgb(path, &processed)
}

fn decode_processed_to_rgb(path: &Path, processed: &ProcessedImageHandle) -> Result<RgbImage> {
    let p = unsafe { &*processed.ptr };
    let width = u32::from(p.width);
    let height = u32::from(p.height);
    let channels = p.colors as usize;
    let bits = p.bits;
    let is_jpeg = p.type_ == sys::LibRaw_image_formats_LIBRAW_IMAGE_JPEG;
    let data = processed.data();
    processed_u8_to_rgb(path, is_jpeg, width, height, channels, bits, data)
}

fn processed_u8_to_rgb(
    source: &Path,
    is_jpeg: bool,
    width: u32,
    height: u32,
    channels: usize,
    bits: u16,
    data: &[u8],
) -> Result<RgbImage> {
    if is_jpeg {
        return Ok(image::load_from_memory(data)
            .map_err(|e| anyhow!("jpeg decode failed for {}: {e}", source.display()))?
            .to_rgb8());
    }

    let bytes_8 = if bits == 8 {
        data.to_vec()
    } else if bits == 16 {
        data.chunks_exact(2).map(|c| c[1]).collect()
    } else {
        return Err(anyhow!(
            "unsupported bit depth {} for {}",
            bits,
            source.display()
        ));
    };

    match channels {
        1 => {
            let gray = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, bytes_8)
                .ok_or_else(|| {
                    anyhow!("failed to build grayscale image for {}", source.display())
                })?;
            Ok(image::DynamicImage::ImageLuma8(gray).to_rgb8())
        }
        3 => ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(width, height, bytes_8)
            .ok_or_else(|| anyhow!("failed to build rgb image for {}", source.display())),
        4 => {
            let rgba = ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, bytes_8)
                .ok_or_else(|| anyhow!("failed to build rgba image for {}", source.display()))?;
            Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8())
        }
        _ => Err(anyhow!(
            "unsupported channel count {} for {}",
            channels,
            source.display()
        )),
    }
}

fn check_libraw(code: i32, context: &str, source: &Path) -> Result<()> {
    if code == sys::LibRaw_errors_LIBRAW_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let ptr = sys::libraw_strerror(code);
        if ptr.is_null() {
            String::from("unknown libraw error")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(anyhow!(
        "{context} failed for {}: {msg} ({code})",
        source.display()
    ))
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        CString::new(path.as_os_str().as_bytes())
            .map_err(|_| anyhow!("path contains interior NUL: {}", path.display()))
    }
    #[cfg(not(unix))]
    {
        CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| anyhow!("path contains interior NUL: {}", path.display()))
    }
}

fn choose_preferred_thumbnail(
    candidates: &[ThumbCandidate],
    max_edge_hint: u32,
) -> Option<ThumbCandidate> {
    if candidates.is_empty() {
        return None;
    }

    let min_long = max_edge_hint.max(256);
    let cap_long = MEDIUM_THUMB_MAX_LONG_EDGE.max(min_long);

    let mut sorted = candidates.to_vec();
    sorted.sort_by_key(|c| c.width.max(c.height));

    let medium = sorted
        .iter()
        .copied()
        .filter(|c| {
            let long = c.width.max(c.height);
            long >= min_long && long <= cap_long
        })
        .max_by_key(|c| c.width.max(c.height));
    if medium.is_some() {
        return medium;
    }

    let above_min = sorted
        .iter()
        .copied()
        .find(|c| c.width.max(c.height) >= min_long);
    if above_min.is_some() {
        return above_min;
    }

    sorted.into_iter().max_by_key(|c| c.width.max(c.height))
}

struct LibRawHandle {
    ptr: *mut sys::libraw_data_t,
}

impl LibRawHandle {
    fn new() -> Result<Self> {
        let ptr = unsafe { sys::libraw_init(0) };
        if ptr.is_null() {
            return Err(anyhow!("libraw_init returned null"));
        }
        Ok(Self { ptr })
    }
}

impl Drop for LibRawHandle {
    fn drop(&mut self) {
        unsafe { sys::libraw_close(self.ptr) };
    }
}

struct ProcessedImageHandle {
    ptr: *mut sys::libraw_processed_image_t,
}

impl ProcessedImageHandle {
    fn make_image(raw_ptr: *mut sys::libraw_data_t, source: &Path) -> Result<Self> {
        let mut err = 0_i32;
        let ptr = unsafe { sys::libraw_dcraw_make_mem_image(raw_ptr, &mut err) };
        if ptr.is_null() || err != sys::LibRaw_errors_LIBRAW_SUCCESS {
            let msg = unsafe {
                let cmsg = sys::libraw_strerror(err);
                if cmsg.is_null() {
                    String::from("unknown libraw error")
                } else {
                    CStr::from_ptr(cmsg).to_string_lossy().into_owned()
                }
            };
            return Err(anyhow!(
                "raw make memory image failed for {}: {msg} ({err})",
                source.display()
            ));
        }
        Ok(Self { ptr })
    }

    fn make_thumb(raw_ptr: *mut sys::libraw_data_t, source: &Path) -> Result<Self> {
        let mut err = 0_i32;
        let ptr = unsafe { sys::libraw_dcraw_make_mem_thumb(raw_ptr, &mut err) };
        if ptr.is_null() || err != sys::LibRaw_errors_LIBRAW_SUCCESS {
            let msg = unsafe {
                let cmsg = sys::libraw_strerror(err);
                if cmsg.is_null() {
                    String::from("unknown libraw error")
                } else {
                    CStr::from_ptr(cmsg).to_string_lossy().into_owned()
                }
            };
            return Err(anyhow!(
                "raw make memory thumbnail failed for {}: {msg} ({err})",
                source.display()
            ));
        }
        Ok(Self { ptr })
    }

    fn data(&self) -> &[u8] {
        let p = unsafe { &*self.ptr };
        unsafe { std::slice::from_raw_parts(p.data.as_ptr(), p.data_size as usize) }
    }
}

impl Drop for ProcessedImageHandle {
    fn drop(&mut self) {
        unsafe { sys::libraw_dcraw_clear_mem(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::{ThumbCandidate, choose_preferred_thumbnail};

    #[test]
    fn picks_medium_over_full() {
        let candidates = vec![
            ThumbCandidate {
                index: 0,
                width: 1616,
                height: 1080,
            },
            ThumbCandidate {
                index: 1,
                width: 9504,
                height: 6336,
            },
        ];
        let got = choose_preferred_thumbnail(&candidates, 1000).expect("thumbnail");
        assert_eq!(got.index, 0);
    }

    #[test]
    fn picks_smallest_above_min_if_no_medium() {
        let candidates = vec![
            ThumbCandidate {
                index: 0,
                width: 7000,
                height: 5000,
            },
            ThumbCandidate {
                index: 1,
                width: 5000,
                height: 3500,
            },
        ];
        let got = choose_preferred_thumbnail(&candidates, 1000).expect("thumbnail");
        assert_eq!(got.index, 1);
    }

    #[test]
    fn picks_largest_if_all_below_min() {
        let candidates = vec![
            ThumbCandidate {
                index: 0,
                width: 640,
                height: 480,
            },
            ThumbCandidate {
                index: 1,
                width: 800,
                height: 600,
            },
        ];
        let got = choose_preferred_thumbnail(&candidates, 2000).expect("thumbnail");
        assert_eq!(got.index, 1);
    }
}
