//! Stage 0 of the gwnestrs P–P test: the coverage of the sampler alone, on
//! a likelihood whose posterior is known to be calibrated.
//!
//! Each injection draws true parameters from the prior (four uniform on
//! [0, 1] and one periodic, uniform on [0, 2 pi)), simulates data from
//! them (Gaussian, and von Mises for the periodic one), and samples the
//! posterior. The credible level of each true value (the fraction of
//! posterior samples below it) is then uniform over injections, which a
//! Kolmogorov–Smirnov test checks per parameter, combined over parameters
//! by Fisher's method.

use std::f64::consts::PI;
use std::sync::Arc;

use nestrs::{
    likelihood::Likelihood,
    nest::{NestedSampler, SamplerOptions, SingleThreadedNestedSampler},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, Uniform};

const SIGMA: f64 = 0.05;
const KAPPA: f64 = 20.0;
const INJECTIONS: usize = 500;

type Prior = Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>;

#[derive(Clone)]
struct Data {
    values: Vec<f64>,
}

impl Likelihood for Data {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        let gaussian: f64 = x[..4]
            .iter()
            .zip(&self.values)
            .map(|(x, d)| -0.5 * ((x - d) / SIGMA).powi(2))
            .sum();
        gaussian + KAPPA * (x[4] - self.values[4]).cos()
    }
}

fn prior() -> Prior {
    let mut prior: Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>> = (0..4)
        .map(|_| {
            Box::new(Uniform::new(0.0, 1.0).unwrap())
                as Box<dyn ContinuousCDF<f64, f64> + Send + Sync>
        })
        .collect();
    prior.push(Box::new(Uniform::new(0.0, 2.0 * PI).unwrap()));
    Arc::new(prior)
}

/// The credible levels of one injection.
fn injection(seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut truth: Vec<f64> = (0..4).map(|_| rng.gen::<f64>()).collect();
    truth.push(2.0 * PI * rng.gen::<f64>());
    let normal = Normal::new(0.0, SIGMA).unwrap();
    let mut values: Vec<f64> = truth[..4].iter().map(|t| t + rng.sample(normal)).collect();
    // A von Mises draw around the truth, by rejection from uniform.
    let periodic = loop {
        let d = 2.0 * PI * rng.gen::<f64>();
        if rng.gen::<f64>() < (KAPPA * ((d - truth[4]).cos() - 1.0)).exp() {
            break d;
        }
    };
    values.push(periodic);
    let mut sampler = NestedSampler::with_options(
        Arc::new(Data { values }),
        prior(),
        SamplerOptions {
            nlive: 100,
            walks: 20,
            naccept: 20,
            seed: Some(seed),
            periodic: vec![false, false, false, false, true],
            ..Default::default()
        },
    );
    let result = SingleThreadedNestedSampler::sample(&mut sampler, 0.1);
    (0..5)
        .map(|i| {
            let below = result.posterior.iter().filter(|x| x[i] < truth[i]).count();
            below as f64 / result.posterior.len() as f64
        })
        .collect()
}

/// The Kolmogorov–Smirnov p-value of `levels` against the uniform
/// distribution (the asymptotic distribution, with Stephens's correction).
fn ks_pvalue(levels: &mut [f64]) -> f64 {
    levels.sort_by(f64::total_cmp);
    let n = levels.len() as f64;
    let statistic = levels
        .iter()
        .enumerate()
        .map(|(i, &x)| (x - i as f64 / n).max((i + 1) as f64 / n - x))
        .fold(0.0, f64::max);
    let lambda = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * statistic;
    let p: f64 = (1..=100)
        .map(|k| {
            let k = k as f64;
            2.0 * (-1f64).powf(k - 1.0) * (-2.0 * k * k * lambda * lambda).exp()
        })
        .sum();
    p.clamp(0.0, 1.0)
}

#[test]
fn credible_levels_are_uniform() {
    let levels: Vec<Vec<f64>> = (0..INJECTIONS as u64)
        .into_par_iter()
        .map(|i| injection(1000 + i))
        .collect();
    let pvalues: Vec<f64> = (0..5)
        .map(|i| ks_pvalue(&mut levels.iter().map(|l| l[i]).collect::<Vec<_>>()))
        .collect();
    let fisher = -2.0 * pvalues.iter().map(|p| p.max(1e-300).ln()).sum::<f64>();
    let combined = 1.0
        - ChiSquared::new(2.0 * pvalues.len() as f64)
            .unwrap()
            .cdf(fisher);
    println!("KS p-values {pvalues:?}, combined {combined}");
    assert!(pvalues.iter().all(|&p| p > 1e-3), "{pvalues:?}");
    assert!(combined > 0.01, "combined p {combined} ({pvalues:?})");
}
