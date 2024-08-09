use std::sync::Arc;

use nestrs::{
    acceptance,
    likelihood::Rosenbrock,
    mcmc,
    nest::{NestedSampler, ThreadedNestedSampler},
    propose, Result,
};
use rand::rngs;
use statrs::distribution::{ContinuousCDF, Uniform};

fn main() {
    let ndim = 2;
    let rng = &mut rngs::ThreadRng::default();

    let prior = Arc::new(
        (0..ndim)
            .map(|_| {
                Box::new(Uniform::new(-5.0, 5.0).unwrap())
                    as Box<dyn ContinuousCDF<f64, f64> + Send + Sync>
            })
            .collect::<Vec<_>>(),
    );

    let mut mcmc = mcmc::MCMC::new(
        Arc::new(Rosenbrock {}),
        prior.clone(),
        Box::new(acceptance::MetropolisHastings::new()),
        "uniform",
    );
    mcmc.sample(100, 100, &mut rng.clone(), propose::uniform_step);
    let result = mcmc.sample(10000, 200, &mut rng.clone(), propose::differential_step);
    result.print_summary();

    let mut nest = NestedSampler::new(
        Arc::new(Rosenbrock {}),
        prior.clone(),
        5000,
        1000,
        100 * ndim,
        "normal",
        true,
    );
    let result = nest.sample(0.1, rng, 16);
    result.print_summary();
}
