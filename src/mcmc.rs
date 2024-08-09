use std::f64::NAN;
use std::sync::Arc;
use std::time::Instant;
use std::vec;

use rand::rngs::ThreadRng;
use statrs::distribution::{ContinuousCDF, FullContinuous};

use crate::likelihood::Likelihood;
use crate::SamplerResult;

use super::acceptance::Acceptance;
use super::{_get_sample_space, point::Point};

#[allow(clippy::upper_case_acronyms)]
pub struct MCMC {
    pub likelihood: Arc<dyn Likelihood>,
    prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
    pub samples: Vec<Point>,
    acceptance: Box<dyn Acceptance>,
    pub accepted: usize,
    pub rejected: usize,
    sample_space: Arc<Box<dyn FullContinuous<f64, f64> + Send + Sync>>,
}

impl MCMC {
    pub fn new(
        likelihood: Arc<dyn Likelihood>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        acceptance: Box<dyn Acceptance>,
        sample_space: &str,
    ) -> Self {
        Self {
            likelihood,
            prior,
            samples: Vec::new(),
            acceptance,
            accepted: 0,
            rejected: 0,
            sample_space: _get_sample_space(&sample_space.to_lowercase()),
        }
    }

    pub fn rescale(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.prior.iter())
            .map(|(&x, prior)| {
                let unit = self.sample_space.cdf(x);
                prior.inverse_cdf(unit)
            })
            .collect()
    }

    pub fn sample(
        &mut self,
        n: usize,
        thin: usize,
        rng: &mut ThreadRng,
        propose: fn(&mut MCMC, &Point, &mut ThreadRng) -> Point,
    ) -> SamplerResult {
        let start = Instant::now();
        let mut point: Point;
        if self.samples.is_empty() {
            let init = vec![0.0; self.prior.len()];
            let scaled = self.rescale(&init);
            point = Point::new(
                init.clone(),
                scaled.clone(),
                self.ln_prior(&init),
                self.likelihood.ln_likelihood(&scaled),
            );
        } else {
            point = self.samples.pop().unwrap();
        }
        for _ in 0..n {
            for _ in 0..thin {
                point = self.step(&point, &mut rng.clone(), propose);
            }
            self.samples.push(point.clone());
        }
        SamplerResult {
            posterior: self.samples.iter().map(|point| point.x.clone()).collect(),
            ln_evidence: NAN,
            duration: start.elapsed(),
        }
    }

    fn step(
        &mut self,
        point: &Point,
        rng: &mut ThreadRng,
        propose: fn(&mut MCMC, &Point, &mut ThreadRng) -> Point,
    ) -> Point {
        let proposed = propose(self, point, &mut rng.clone());
        if self.acceptance.acceptance(point, &proposed) {
            self.accepted += 1;
            proposed
        } else {
            self.rejected += 1;
            point.clone()
        }
    }

    pub fn mean(&self, idx: usize) -> f64 {
        self.samples
            .iter()
            .map(|point| point.scaled[idx])
            .sum::<f64>()
            / self.samples.len() as f64
    }

    pub fn variance(&self, idx: usize) -> f64 {
        let mean = self.mean(idx);
        self.samples
            .iter()
            .map(|point| (point.scaled[idx] - mean).powi(2))
            .sum::<f64>()
            / self.samples.len() as f64
    }

    pub fn ln_prior(&self, x: &[f64]) -> f64 {
        x.iter().map(|&x| self.sample_space.ln_pdf(x)).sum()
    }
}
