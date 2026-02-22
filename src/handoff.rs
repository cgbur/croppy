use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step01Handoff {
    pub raw: String,
    pub prepared: String,
    pub preprocess: String,
    pub transform: Step01Transform,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step01Transform {
    pub raw_width: u32,
    pub raw_height: u32,
    pub prepared_width: u32,
    pub prepared_height: u32,
    pub flip_180: bool,
}

pub fn write_handoff(path: &Path, handoff: &Step01Handoff) -> Result<()> {
    let json = serde_json::to_string_pretty(handoff)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn read_handoff(path: &Path) -> Result<Step01Handoff> {
    let text = fs::read_to_string(path)?;
    let handoff: Step01Handoff = serde_json::from_str(&text)?;
    Ok(handoff)
}
