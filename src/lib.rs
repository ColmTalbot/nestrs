use statrs::distribution::{FullContinuous, Normal, Uniform};
use std::sync::Arc;
use std::time::Duration;

pub mod acceptance;
pub mod likelihood;
pub mod mcmc;
pub mod nest;
pub mod point;
pub mod propose;

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

pub struct SamplerResult {
    pub posterior: Vec<Vec<f64>>,
    pub ln_evidence: f64,
    pub duration: Duration,
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
