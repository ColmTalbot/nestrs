use std::fmt::Debug;

use rand::Rng;

use crate::NestRng;

use super::point::Point;

pub trait Acceptance: Debug {
    fn acceptance(&mut self, x: &Point, x_prime: &Point, rng: &mut NestRng) -> bool;
}

#[derive(Debug, Default)]
pub struct MetropolisHastings {}

impl MetropolisHastings {
    pub fn new() -> Self {
        Self {}
    }
}

impl Acceptance for MetropolisHastings {
    fn acceptance(&mut self, x: &Point, x_prime: &Point, rng: &mut NestRng) -> bool {
        let log_acceptance_ratio = x_prime.logpdf - x.logpdf;
        log_acceptance_ratio >= 0.0 || log_acceptance_ratio >= rng.gen::<f64>().ln()
    }
}

#[derive(Debug)]
pub struct Threshold {
    pub threshold: f64,
}

impl Threshold {
    #[allow(dead_code)]
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Acceptance for Threshold {
    fn acceptance(&mut self, x: &Point, x_prime: &Point, rng: &mut NestRng) -> bool {
        if x_prime.ln_likelihood <= self.threshold {
            false
        } else {
            let log_acceptance_ratio = x_prime.ln_prior - x.ln_prior;
            log_acceptance_ratio >= rng.gen::<f64>().ln()
        }
    }
}
