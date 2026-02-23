//! Robust line-fitting and angle fusion kernels.
//!
//! Rotation refinement needs robust line fits from noisy edge samples. This
//! module contains the shared fit and angle-combination routines.

/// Robust line fit returned by consensus search.
///
/// Depending on call site this represents either:
/// - `y = m*x + b` for horizontal-edge fitting
/// - `x = m*y + b` for vertical-edge fitting
#[derive(Clone, Debug)]
pub struct ConsensusLineFit {
    m: f32,
    b: f32,
    score: f32,
}

impl ConsensusLineFit {
    /// Fitted slope (`m`).
    pub fn slope(&self) -> f32 {
        self.m
    }

    /// Fitted intercept (`b`).
    pub fn intercept(&self) -> f32 {
        self.b
    }

    /// Heuristic confidence from inlier ratio/span/residual.
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// Converts a `y = m*x + b` fit to edge angle in degrees.
pub fn horizontal_fit_angle_deg(f: &ConsensusLineFit) -> f32 {
    f.m.atan().to_degrees()
}

/// Converts an `x = m*y + b` fit to edge angle in degrees.
pub fn vertical_fit_angle_deg(f: &ConsensusLineFit) -> f32 {
    // Vertical fits are represented as x = m*y + b. For the edge angle relative
    // to the x-axis, this sign is opposite to y = m*x + b.
    (-f.m).atan().to_degrees()
}

/// Fits `y = m*x + b` using consensus over point pairs.
pub fn fit_line_y_of_x(points: &[(f32, f32)]) -> Option<ConsensusLineFit> {
    let pairs: Vec<(f32, f32)> = points.iter().map(|(x, y)| (*x, *y)).collect();
    consensus_fit_line_t_of_u(&pairs)
}

/// Fits `x = m*y + b` using consensus over point pairs.
pub fn fit_line_x_of_y(points: &[(f32, f32)]) -> Option<ConsensusLineFit> {
    let pairs: Vec<(f32, f32)> = points.iter().map(|(x, y)| (*y, *x)).collect();
    consensus_fit_line_t_of_u(&pairs)
}

/// Combines per-edge angle estimates into a single rotation estimate.
///
/// Horizontal edges are preferred when both horizontal and vertical estimates
/// disagree, because they have been empirically more stable on this data.
pub fn pick_rotation_angle(
    top: Option<&ConsensusLineFit>,
    bottom: Option<&ConsensusLineFit>,
    left: Option<&ConsensusLineFit>,
    right: Option<&ConsensusLineFit>,
) -> Option<f32> {
    fn weighted_angle(edges: &[(f32, f32)]) -> Option<f32> {
        let mut wsum = 0.0f32;
        let mut asum = 0.0f32;
        for (a, w) in edges {
            if *w > 0.0 {
                wsum += *w;
                asum += *a * *w;
            }
        }
        if wsum <= 1e-6 {
            None
        } else {
            Some(asum / wsum)
        }
    }

    let top_edge = top.map(|f| (horizontal_fit_angle_deg(f), f.score()));
    let bot_edge = bottom.map(|f| (horizontal_fit_angle_deg(f), f.score()));
    let left_edge = left.map(|f| (vertical_fit_angle_deg(f), f.score()));
    let right_edge = right.map(|f| (vertical_fit_angle_deg(f), f.score()));

    let mut horiz = Vec::new();
    if let Some(e) = top_edge {
        horiz.push(e);
    }
    if let Some(e) = bot_edge {
        horiz.push(e);
    }

    let mut vert = Vec::new();
    if let Some(e) = left_edge {
        vert.push(e);
    }
    if let Some(e) = right_edge {
        vert.push(e);
    }

    let h = weighted_angle(&horiz);
    let v = weighted_angle(&vert);

    match (h, v) {
        (Some(ha), Some(va)) => {
            if (ha - va).abs() <= 0.8 {
                let hw = horiz.iter().map(|(_, w)| *w).sum::<f32>() * 1.5;
                let vw = vert.iter().map(|(_, w)| *w).sum::<f32>();
                weighted_angle(&[(ha, hw), (va, vw)])
            } else {
                Some(ha)
            }
        }
        (Some(ha), None) => Some(ha),
        (None, Some(va)) => Some(va),
        (None, None) => None,
    }
}

fn consensus_fit_line_t_of_u(points: &[(f32, f32)]) -> Option<ConsensusLineFit> {
    if points.len() < 6 {
        return None;
    }

    let (min_t, max_t) = points
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, p| {
            (acc.0.min(p.0), acc.1.max(p.0))
        });
    let span_t = max_t - min_t;
    if span_t <= 1e-3 {
        return None;
    }
    let inlier_thresh = (span_t * 0.0035).clamp(0.8, 2.5);

    let mut best_count = 0usize;
    let mut best_err = f32::INFINITY;
    let mut best_mb: Option<(f32, f32)> = None;

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let (t1, u1) = points[i];
            let (t2, u2) = points[j];
            let dt = t2 - t1;
            if dt.abs() <= 1e-3 {
                continue;
            }
            let m = (u2 - u1) / dt;
            let b = u1 - m * t1;

            let mut count = 0usize;
            let mut err = 0.0f32;
            for (t, u) in points {
                let r = (*u - (m * *t + b)).abs();
                if r <= inlier_thresh {
                    count += 1;
                    err += r;
                }
            }

            if count > best_count || (count == best_count && err < best_err) {
                best_count = count;
                best_err = err;
                best_mb = Some((m, b));
            }
        }
    }

    let inlier_ratio = best_count as f32 / points.len() as f32;
    if best_count < 6 || inlier_ratio < 0.55 {
        return None;
    }
    let mean_inlier_residual = best_err / best_count as f32;
    let span_score = span_t / (span_t + 120.0);
    let residual_score = 1.0 / (1.0 + mean_inlier_residual);
    let score = inlier_ratio * span_score * residual_score;

    best_mb.map(|(m, b)| ConsensusLineFit { m, b, score })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit_from_deg_y_of_x(deg: f32) -> ConsensusLineFit {
        ConsensusLineFit {
            m: deg.to_radians().tan(),
            b: 0.0,
            score: 1.0,
        }
    }

    fn fit_from_deg_x_of_y(deg: f32) -> ConsensusLineFit {
        // For x = m*y + b, m = -tan(theta) where theta is edge angle wrt x-axis.
        ConsensusLineFit {
            m: -deg.to_radians().tan(),
            b: 0.0,
            score: 1.0,
        }
    }

    #[test]
    fn pick_rotation_angle_does_not_cancel_from_vertical_sign_mismatch() {
        let top = fit_from_deg_y_of_x(-0.30);
        let bottom = fit_from_deg_y_of_x(-0.20);
        let left = fit_from_deg_x_of_y(-0.40);
        let right = fit_from_deg_x_of_y(-0.45);

        let angle = pick_rotation_angle(Some(&top), Some(&bottom), Some(&left), Some(&right))
            .expect("angle should be estimated");
        assert!(angle < -0.2, "unexpected angle_deg={angle}");
    }

    #[test]
    fn pick_rotation_angle_vertical_only_uses_same_sign_convention() {
        let left = fit_from_deg_x_of_y(-0.35);
        let right = fit_from_deg_x_of_y(-0.30);

        let angle =
            pick_rotation_angle(None, None, Some(&left), Some(&right)).expect("angle expected");
        assert!(angle < -0.2, "unexpected angle_deg={angle}");
    }

    #[test]
    fn fit_line_y_of_x_handles_outliers() {
        let mut points = vec![
            (0.0, 0.0),
            (10.0, 5.0),
            (20.0, 10.0),
            (30.0, 15.0),
            (40.0, 20.0),
            (50.0, 25.0),
            (60.0, 30.0),
            (70.0, 35.0),
            (80.0, 40.0),
        ];
        points.push((25.0, 200.0));
        points.push((55.0, -120.0));

        let fit = fit_line_y_of_x(&points).expect("line should fit");
        assert!((fit.slope() - 0.5).abs() < 0.05, "slope={}", fit.slope());
        assert!(fit.score() > 0.1, "score={}", fit.score());
    }
}
