use std::f64::NAN;
use std::sync::Arc;
use std::time::Instant;
use std::vec;

use statrs::distribution::ContinuousCDF;

use crate::likelihood::Likelihood;
use crate::{NestRng, SamplerResult};

use super::acceptance::Acceptance;
use super::{point::Point, SampleSpace};

#[allow(clippy::upper_case_acronyms)]
pub struct MCMC {
    pub likelihood: Arc<dyn Likelihood>,
    prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
    pub samples: Vec<Point>,
    acceptance: Box<dyn Acceptance>,
    pub accepted: usize,
    pub rejected: usize,
    space: SampleSpace,
}

impl MCMC {
    pub fn new(
        likelihood: Arc<dyn Likelihood>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        acceptance: Box<dyn Acceptance>,
        sample_space: &str,
    ) -> Self {
        Self::with_space(
            likelihood,
            prior,
            acceptance,
            SampleSpace::new(sample_space, Vec::new()),
        )
    }

    /// A chain exploring `space`, which may have periodic dimensions.
    pub fn with_space(
        likelihood: Arc<dyn Likelihood>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        acceptance: Box<dyn Acceptance>,
        space: SampleSpace,
    ) -> Self {
        Self {
            likelihood,
            prior,
            samples: Vec::new(),
            acceptance,
            accepted: 0,
            rejected: 0,
            space,
        }
    }

    /// The space the chain explores.
    pub fn space(&self) -> &SampleSpace {
        &self.space
    }

    pub fn rescale(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.prior.iter())
            .enumerate()
            .map(|(i, (&x, prior))| prior.inverse_cdf(self.space.to_unit(i, x)))
            .collect()
    }

    pub fn sample(
        &mut self,
        n: usize,
        thin: usize,
        rng: &mut NestRng,
        propose: fn(&mut MCMC, &Point, &mut NestRng) -> Point,
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
                point = self.step(&point, rng, propose);
            }
            self.samples.push(point.clone());
        }
        SamplerResult {
            posterior: self.samples.iter().map(|point| point.x.clone()).collect(),
            ln_evidence: NAN,
            ln_evidence_error: NAN,
            duration: start.elapsed(),
            ..Default::default()
        }
    }

    fn step(
        &mut self,
        point: &Point,
        rng: &mut NestRng,
        propose: fn(&mut MCMC, &Point, &mut NestRng) -> Point,
    ) -> Point {
        let proposed = propose(self, point, rng);
        if self.acceptance.acceptance(point, &proposed, rng) {
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
        self.space.ln_prior(x)
    }
}
