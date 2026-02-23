//! Reusable numeric kernels for boundary detection and rotation refinement.
//!
//! Detection and rotation pipelines depend on these low-level operations.
//! Keeping them in one module makes reuse and unit testing straightforward.

pub mod edge_scan;
pub mod line_fit;
pub mod peak_pick;
pub mod signal_1d;
