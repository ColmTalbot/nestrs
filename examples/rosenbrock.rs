use std::sync::Arc;

use nestrs::{
    acceptance,
    likelihood::Rosenbrock,
    mcmc,
    nest::{NestedSampler, ThreadedNestedSampler},
    propose, Result,
};
use nestrs::{NestRng, SeedableRng};
use statrs::distribution::{ContinuousCDF, Uniform};

fn main() {
    let ndim = 2;
    let rng = &mut NestRng::seed_from_u64(1234);

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
    mcmc.sample(100, 100, rng, propose::uniform_step);
    let result = mcmc.sample(10000, 200, rng, propose::differential_step);
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
    let result = nest.sample(0.1, 16);
    result.print_summary();
}
