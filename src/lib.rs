pub use rand_chacha::rand_core::SeedableRng;
use statrs::distribution::{FullContinuous, Normal, Uniform};
use std::sync::Arc;
use std::time::Duration;

pub mod acceptance;
pub mod likelihood;
pub mod mcmc;
pub mod nest;
pub mod point;
pub mod propose;

/// The random number generator used throughout. It is seedable, so runs can be
/// reproduced.
pub type NestRng = rand_chacha::ChaCha8Rng;

/// The space the sampler explores, one dimension per parameter. Ordinary
/// dimensions use the named base space (`"normal"` or `"uniform"`), mapped
/// to the prior through its CDF. Periodic dimensions are always on `[0, 1)`
/// and wrap around, so that the sampler can move across the boundary of,
/// e.g., a phase.
#[derive(Clone)]
pub struct SampleSpace {
    base: Arc<Box<dyn FullContinuous<f64, f64> + Send + Sync>>,
    periodic: Arc<Vec<bool>>,
}

impl SampleSpace {
    /// The base space `name` with the periodic dimensions flagged in
    /// `periodic` (dimensions beyond its length are not periodic).
    pub fn new(name: &str, periodic: Vec<bool>) -> Self {
        Self {
            base: _get_sample_space(&name.to_lowercase()),
            periodic: Arc::new(periodic),
        }
    }

    pub fn is_periodic(&self, dimension: usize) -> bool {
        self.periodic.get(dimension).copied().unwrap_or(false)
    }

    /// A draw from the space's distribution in `dimension`, given a uniform
    /// draw `u`.
    pub fn from_uniform(&self, dimension: usize, u: f64) -> f64 {
        if self.is_periodic(dimension) {
            u
        } else {
            self.base.inverse_cdf(u)
        }
    }

    /// The point `x` of `dimension` as a fraction of the prior mass.
    pub fn to_unit(&self, dimension: usize, x: f64) -> f64 {
        if self.is_periodic(dimension) {
            x
        } else {
            self.base.cdf(x)
        }
    }

    /// The log density of the space's distribution at `x` in `dimension`.
    pub fn ln_pdf(&self, dimension: usize, x: f64) -> f64 {
        if self.is_periodic(dimension) {
            if (0.0..1.0).contains(&x) {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            self.base.ln_pdf(x)
        }
    }

    /// Wrap the periodic dimensions of `x` into `[0, 1)`.
    pub fn wrap(&self, x: &mut [f64]) {
        for (dimension, value) in x.iter_mut().enumerate() {
            if self.is_periodic(dimension) {
                *value = value.rem_euclid(1.0);
            }
        }
    }

    /// The sum of [`SampleSpace::ln_pdf`] over dimensions.
    pub fn ln_prior(&self, x: &[f64]) -> f64 {
        x.iter().enumerate().map(|(i, &x)| self.ln_pdf(i, x)).sum()
    }
}

pub(crate) fn _get_sample_space(
    sample_space: &str,
) -> Arc<Box<dyn FullContinuous<f64, f64> + Send + Sync>> {
    if sample_space == "normal" {
        Arc::new(Box::new(Normal::new(0.0, 1.0).unwrap()))
    } else if sample_space == "uniform" {
        Arc::new(Box::new(Uniform::new(0.0, 1.0).unwrap()))
    } else {
        panic!("Invalid sample space");
    }
}

pub trait Result {
    fn print_summary(&self);
}

#[derive(Debug, Clone, Default)]
pub struct SamplerResult {
    /// Equally weighted posterior samples in the prior (physical) space.
    pub posterior: Vec<Vec<f64>>,
    /// The log likelihood of each posterior sample.
    pub posterior_ln_likelihood: Vec<f64>,
    /// Every dead point (and the final live points) in the prior space.
    pub nested_samples: Vec<Vec<f64>>,
    /// The log likelihood of each nested sample.
    pub nested_ln_likelihood: Vec<f64>,
    /// The normalised log posterior weight of each nested sample.
    pub nested_ln_weights: Vec<f64>,
    pub ln_evidence: f64,
    /// Estimate of the statistical uncertainty in `ln_evidence`, sqrt(H / nlive).
    pub ln_evidence_error: f64,
    pub duration: Duration,
    /// Whether the run was stopped early (see `SamplerOptions::stop`), in
    /// which case the evidence and posterior are not converged.
    pub interrupted: bool,
}

impl Result for SamplerResult {
    fn print_summary(&self) {
        println!("Sampling time: {:.2}s", self.duration.as_secs_f64());
        println!("ln evidence: {:.2}", self.ln_evidence);
        println!("{} posterior samples", self.posterior.len());
        println!("Posterior summary:");
        for idx in 0..self.posterior[0].len() {
            let samples: Vec<f64> = self.posterior.iter().map(|point| point[idx]).collect();
            println!(
                "    {:.2} +/- {:.2}",
                mean(&samples),
                variance(&samples).powf(0.5)
            );
        }
    }
}

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|&x| (x - m).powf(2.0)).sum::<f64>() / x.len() as f64
}
