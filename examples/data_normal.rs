use std::sync::Arc;

use nestrs::Result;
use nestrs::{
    acceptance,
    likelihood::Likelihood,
    mcmc,
    nest::{NestedSampler, ThreadedNestedSampler},
    propose,
};
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use statrs::distribution::{ContinuousCDF, Normal, Uniform};

#[derive(Clone)]
pub struct NormalLikelihood {
    data: Vec<Vec<f64>>,
}

impl NormalLikelihood {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self { data }
    }
}

impl Likelihood for NormalLikelihood {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (idx, &x) in x.iter().enumerate() {
            sum += self.data[idx]
                .iter()
                .map(|&y| -(y - x).powf(2.0) / 2.0)
                .sum::<f64>();
        }
        sum
    }
}

fn main() {
    let ndim: usize = 10;
    let rng = &mut ThreadRng::default();

    let base = Uniform::new(0.0, 1.0).unwrap();

    let means = base.sample_iter(rng.clone()).take(ndim);
    let stds = base.sample_iter(rng.clone()).take(ndim);

    let data: Vec<Vec<f64>> = means
        .zip(stds)
        .map(|(mean, std)| {
            Normal::new(mean, std)
                .unwrap()
                .sample_iter(rng.clone())
                .take(10000)
                .collect()
        })
        .collect();

    let prior = Arc::new(
        (0..ndim)
            .map(|_| {
                Box::new(Uniform::new(-5.0, 5.0).unwrap())
                    as Box<dyn ContinuousCDF<f64, f64> + Send + Sync>
            })
            .collect::<Vec<_>>(),
    );

    let likelihood = Arc::new(NormalLikelihood::new(data));

    let mut mcmc = mcmc::MCMC::new(
        likelihood.clone(),
        prior.clone(),
        Box::new(acceptance::MetropolisHastings::new()),
        "normal",
    );
    mcmc.sample(100, 100, &mut rng.clone(), propose::uniform_step);
    let result = mcmc.sample(1000, 100, &mut rng.clone(), propose::differential_step);
    result.print_summary();

    let mut nest = NestedSampler::new(likelihood, prior, 500, 1000, 40, "normal", true);
    let result = nest.sample(0.1, rng, 4);
    result.print_summary()
}
