//! Edge-scanning kernels on 1D derivative signals.
//!
//! Detection needs stable edge picks from noisy derivative profiles. This
//! module provides the shared polarity-aware scanning primitives.

/// Expected derivative sign for an edge transition.
#[derive(Debug, Clone, Copy)]
pub enum EdgePolarity {
    Rising,
    Falling,
}

impl EdgePolarity {
    /// Returns the positive response for the configured polarity.
    pub fn response(self, derivative_value: f32) -> f32 {
        match self {
            EdgePolarity::Rising => derivative_value.max(0.0),
            EdgePolarity::Falling => (-derivative_value).max(0.0),
        }
    }
}

/// Tuning values for solving one axis worth of target edges.
#[derive(Debug, Clone, Copy)]
pub struct AxisDetectConfig {
    /// Maximum distance from each side to search for the corresponding edge.
    pub max_side_frac: f32,
    /// Guard band to skip at both extremes where border artifacts are common.
    pub side_guard_frac: f32,
    /// Minimum required spacing between the solved start/end edge pair.
    pub min_span_frac: f32,
    /// Relative threshold for accepting a hump.
    pub rel_thresh: f32,
    /// Position inside a hump to choose (`0` outer-facing, `1` center-facing).
    pub center_bias: f32,
    /// Expected polarity of the edge near the start of the axis.
    pub start_polarity: EdgePolarity,
    /// Expected polarity of the edge near the end of the axis.
    pub end_polarity: EdgePolarity,
}

/// Solved edge indices and normalized confidence score for one axis.
#[derive(Debug, Clone, Copy)]
pub struct AxisEdges {
    /// Edge near the start of the axis.
    pub start: usize,
    /// Edge near the end of the axis.
    pub end: usize,
    /// Strength estimate for the edge pair.
    pub score: f32,
}

/// Detects one polarity-constrained edge pair on a single derivative profile.
///
/// This is intentionally simple: scan from the left for the first strong hump
/// of `start_polarity`, scan from the right for the first strong hump of
/// `end_polarity`, then validate spacing and score the pair.
pub fn detect_axis_edges(
    derivative: &[f32],
    axis_len: usize,
    cfg: AxisDetectConfig,
) -> Option<AxisEdges> {
    if derivative.len() < 3 || axis_len < 3 {
        return None;
    }

    let max_idx = derivative.len() - 2;
    let max_side = ((axis_len as f32) * cfg.max_side_frac).round().max(4.0) as usize;
    let side_guard = ((axis_len as f32) * cfg.side_guard_frac).round().max(1.0) as usize;

    let left_start = side_guard.min(max_idx);
    let left_end = max_side.min(max_idx);
    if left_start > left_end {
        return None;
    }

    let right_start = derivative
        .len()
        .saturating_sub(1 + max_side.min(derivative.len().saturating_sub(1)))
        .min(max_idx);
    let right_end = max_idx.saturating_sub(side_guard);
    if right_start > right_end {
        return None;
    }

    let start = first_strong_hump_polarity(
        derivative,
        HumpSearchSpec {
            start: left_start,
            end: left_end,
            rel_thresh: cfg.rel_thresh,
            center_bias: cfg.center_bias,
            dir: ScanDir::Forward,
            polarity: cfg.start_polarity,
        },
    )?;

    let end = first_strong_hump_polarity(
        derivative,
        HumpSearchSpec {
            start: right_start,
            end: right_end,
            rel_thresh: cfg.rel_thresh,
            center_bias: cfg.center_bias,
            dir: ScanDir::Backward,
            polarity: cfg.end_polarity,
        },
    )?;

    let min_span = ((axis_len as f32) * cfg.min_span_frac).round().max(8.0) as usize;
    if start + min_span >= end {
        return None;
    }

    let score = side_score_polarity(derivative, start, cfg.start_polarity)
        .min(side_score_polarity(derivative, end, cfg.end_polarity));

    Some(AxisEdges { start, end, score })
}

/// Returns the strongest polarity response in an inclusive index window.
pub fn peak_idx_polarity(
    derivative: &[f32],
    start: usize,
    end: usize,
    polarity: EdgePolarity,
) -> Option<usize> {
    if derivative.is_empty() {
        return None;
    }
    let s = start.min(derivative.len() - 1);
    let e = end.min(derivative.len() - 1);
    if s >= e {
        return None;
    }

    let mut best_i = s;
    let mut best_v = f32::NEG_INFINITY;
    for (i, val) in derivative.iter().enumerate().take(e + 1).skip(s) {
        let resp = polarity.response(*val);
        if resp > best_v {
            best_v = resp;
            best_i = i;
        }
    }
    Some(best_i)
}

/// Converts a picked edge response into a `[0, 1]` score using a per-signal
/// percentile normalization.
pub fn side_score_polarity(derivative: &[f32], idx: usize, polarity: EdgePolarity) -> f32 {
    if derivative.is_empty() || idx >= derivative.len() {
        return 0.0;
    }

    let mut s: Vec<f32> = derivative.iter().map(|v| polarity.response(*v)).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = ((s.len() - 1) as f32 * 0.95).round() as usize;
    let p95 = s[p95_idx].max(1e-6);
    (polarity.response(derivative[idx]) / p95).clamp(0.0, 1.0)
}

/// Direction for scanning hump candidates.
#[derive(Debug, Clone, Copy)]
pub enum ScanDir {
    Forward,
    Backward,
}

/// Parameters for finding a strong hump in a derivative range.
#[derive(Debug, Clone, Copy)]
pub struct HumpSearchSpec {
    /// Scan start index.
    pub start: usize,
    /// Scan end index.
    pub end: usize,
    /// Relative threshold against local max polarity response.
    pub rel_thresh: f32,
    /// Position bias inside a contiguous hump run.
    pub center_bias: f32,
    /// Forward/backward scanning direction.
    pub dir: ScanDir,
    /// Target edge polarity.
    pub polarity: EdgePolarity,
}

/// Picks the first hump above threshold in the configured scan direction.
///
/// Falls back to the maximum-polarity point when no run survives filtering.
pub fn first_strong_hump_polarity(derivative: &[f32], spec: HumpSearchSpec) -> Option<usize> {
    if derivative.len() < 3 {
        return None;
    }

    let s = spec.start.max(1).min(derivative.len() - 2);
    let e = spec.end.min(derivative.len() - 2);
    if s > e {
        return None;
    }

    let max_v = derivative[s..=e]
        .iter()
        .map(|v| spec.polarity.response(*v))
        .fold(0.0f32, f32::max);
    let threshold = max_v * spec.rel_thresh.clamp(0.0, 1.0);
    let b = spec.center_bias.clamp(0.0, 1.0);

    match spec.dir {
        ScanDir::Forward => {
            let mut i = s;
            while i <= e {
                if spec.polarity.response(derivative[i]) >= threshold {
                    let run_start = i;
                    let mut run_end = i;
                    while run_end < e
                        && spec.polarity.response(derivative[run_end + 1]) >= threshold
                    {
                        run_end += 1;
                    }
                    return Some(pick_from_run(run_start, run_end, b, spec.dir));
                }
                i += 1;
            }
        }
        ScanDir::Backward => {
            let mut i = e;
            loop {
                if spec.polarity.response(derivative[i]) >= threshold {
                    let run_end = i;
                    let mut run_start = i;
                    while run_start > s
                        && spec.polarity.response(derivative[run_start - 1]) >= threshold
                    {
                        run_start -= 1;
                    }
                    return Some(pick_from_run(run_start, run_end, b, spec.dir));
                }
                if i == s {
                    break;
                }
                i -= 1;
            }
        }
    }

    peak_idx_polarity(derivative, s, e, spec.polarity)
}

fn pick_from_run(run_start: usize, run_end: usize, center_bias: f32, dir: ScanDir) -> usize {
    let span = run_end.saturating_sub(run_start);
    let offset = ((span as f32) * center_bias.clamp(0.0, 1.0)).round() as usize;
    match dir {
        ScanDir::Forward => (run_start + offset).min(run_end),
        ScanDir::Backward => run_end.saturating_sub(offset).max(run_start),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_idx_picks_expected_with_polarity() {
        let derivative = [0.0, -1.0, 4.0, -8.0, 3.0, 0.0];
        let rising = peak_idx_polarity(&derivative, 1, 4, EdgePolarity::Rising);
        assert_eq!(rising, Some(2));

        let falling = peak_idx_polarity(&derivative, 1, 4, EdgePolarity::Falling);
        assert_eq!(falling, Some(3));
    }

    #[test]
    fn first_strong_hump_picks_first_qualifying_run() {
        let derivative = [0.0, 0.0, 5.0, 5.0, 0.0, 4.0, 4.0, 0.0, 5.0, 5.0];
        let out = first_strong_hump_polarity(
            &derivative,
            HumpSearchSpec {
                start: 1,
                end: 9,
                rel_thresh: 0.8,
                center_bias: 1.0,
                dir: ScanDir::Forward,
                polarity: EdgePolarity::Rising,
            },
        );
        assert_eq!(out, Some(3));
    }

    #[test]
    fn detect_axis_edges_finds_polarity_targeted_pair_on_synthetic_signal() {
        let mut derivative = vec![0.0f32; 40];
        derivative[3] = -10.0;
        derivative[10] = 6.0;
        derivative[11] = 5.0;
        derivative[28] = -6.0;
        derivative[29] = -7.0;
        derivative[36] = 9.0;

        let out = detect_axis_edges(
            &derivative,
            40,
            AxisDetectConfig {
                max_side_frac: 0.45,
                side_guard_frac: 0.02,
                min_span_frac: 0.15,
                rel_thresh: 0.5,
                center_bias: 0.8,
                start_polarity: EdgePolarity::Rising,
                end_polarity: EdgePolarity::Falling,
            },
        )
        .expect("axis edges should be found");

        assert_eq!(out.start, 11);
        assert_eq!(out.end, 28);
        assert!(out.score > 0.5);
    }
}
