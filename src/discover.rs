use std::path::{Path, PathBuf};

use anyhow::Result;
use walkdir::WalkDir;

const RAW_EXTS: &[&str] = &[
    "arw", "cr2", "cr3", "nef", "nrw", "raf", "orf", "rw2", "dng", "pef",
];

pub fn list_raw_files(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
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
        if RAW_EXTS.contains(&ext.as_str()) {
            out.push(path.to_path_buf());
        }
    }

    out.sort();
    Ok(out)
}

pub fn is_supported_raw(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| RAW_EXTS.contains(&s.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}
