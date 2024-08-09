use std::fmt::Debug;

use rand::Rng;

use super::point::Point;

pub trait Acceptance: Debug {
    fn acceptance(&mut self, x: &Point, x_prime: &Point) -> bool;
}

#[derive(Debug)]
pub struct MetropolisHastings {
    rng: rand::rngs::ThreadRng,
}

impl MetropolisHastings {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl Acceptance for MetropolisHastings {
    fn acceptance(&mut self, x: &Point, x_prime: &Point) -> bool {
        let log_acceptance_ratio = x_prime.logpdf - x.logpdf;
        log_acceptance_ratio >= 0.0 || log_acceptance_ratio >= self.rng.gen::<f64>().ln()
    }
}

#[derive(Debug)]
pub struct Threshold {
    pub threshold: f64,
    rng: rand::rngs::ThreadRng,
}

impl Threshold {
    #[allow(dead_code)]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            rng: rand::thread_rng(),
        }
    }
}

impl Acceptance for Threshold {
    fn acceptance(&mut self, x: &Point, x_prime: &Point) -> bool {
        if x_prime.ln_likelihood <= self.threshold {
            false
        } else {
            let log_acceptance_ratio = x_prime.ln_prior - x.ln_prior;
            log_acceptance_ratio >= self.rng.gen::<f64>().ln()
        }
    }
}
