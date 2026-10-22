//! A likelihood peaked at the boundary of a periodic parameter.

use std::f64::consts::PI;
use std::sync::Arc;

use nestrs::likelihood::Likelihood;
use nestrs::nest::{NestedSampler, SamplerOptions, ThreadedNestedSampler};
use statrs::distribution::{ContinuousCDF, Uniform};

/// `exp(kappa cos x)`, peaked at `x = 0 = 2 pi`, times a narrow Gaussian in a
/// second, ordinary parameter.
#[derive(Clone)]
struct VonMises {
    kappa: f64,
}

impl Likelihood for VonMises {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        self.kappa * x[0].cos() - (x[1] - 0.5).powi(2) / (2.0 * 0.01)
    }
}

fn run(periodic: bool, seed: u64) -> (f64, f64, f64) {
    let prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>> = Arc::new(vec![
        Box::new(Uniform::new(0.0, 2.0 * PI).unwrap()),
        Box::new(Uniform::new(0.0, 1.0).unwrap()),
    ]);
    let kappa = 50.0;
    let mut sampler = NestedSampler::with_options(
        Arc::new(VonMises { kappa }),
        prior,
        SamplerOptions {
            nlive: 200,
            walks: 20,
            naccept: 20,
            seed: Some(seed),
            periodic: vec![periodic, false],
            ..Default::default()
        },
    );
    let result = ThreadedNestedSampler::sample(&mut sampler, 0.1, 4);
    let upper = result.posterior.iter().filter(|x| x[0] > PI).count() as f64
        / result.posterior.len() as f64;
    // ln Z = ln(I0(kappa)) + ln(sqrt(2 pi 0.01)) for the uniform priors.
    let expected = ln_i0(kappa) + (2.0 * PI * 0.01f64).sqrt().ln();
    (
        result.ln_evidence - expected,
        result.ln_evidence_error,
        upper,
    )
}

/// ln I0 by the asymptotic series, accurate to ~1e-8 for kappa = 50.
fn ln_i0(x: f64) -> f64 {
    let series = 1.0 + 1.0 / (8.0 * x) + 9.0 / (128.0 * x * x) + 225.0 / (3072.0 * x * x * x);
    x - 0.5 * (2.0 * PI * x).ln() + series.ln()
}

#[test]
fn periodic_parameter_crosses_the_boundary() {
    let (offset, error, upper) = run(true, 11);
    println!("periodic: ln Z offset {offset:.3} +/- {error:.3}, fraction above pi {upper:.3}");
    assert!(offset.abs() < 4.0 * error, "{offset} {error}");
    assert!((upper - 0.5).abs() < 0.1, "{upper}");
    // For comparison: in one easy dimension the non-periodic sampler copes
    // too, by keeping live points near both ends.
    let (offset, error, upper) = run(false, 11);
    println!("not periodic: ln Z offset {offset:.3} +/- {error:.3}, fraction above pi {upper:.3}");
}
