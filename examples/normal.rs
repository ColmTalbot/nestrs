use std::f64::consts::{PI, SQRT_2};
use std::sync::Arc;

use nestrs::{
    acceptance,
    likelihood::Gaussian,
    mcmc,
    nest::{NestedSampler, ThreadedNestedSampler},
    propose, Result,
};
use rand::rngs;
use statrs::{
    distribution::{ContinuousCDF, Uniform},
    function::erf::erf,
};

fn main() {
    let ndim = 20;
    let rng = &mut rngs::ThreadRng::default();

    let prior = Arc::new(
        (0..ndim)
            .map(|_| {
                Box::new(Uniform::new(-5.0, 5.0).unwrap())
                    as Box<dyn ContinuousCDF<f64, f64> + Send + Sync>
            })
            .collect::<Vec<_>>(),
    );

    let likelihood = Gaussian::from_const_mean_std(0.5, 0.01, ndim);

    let mut mcmc = mcmc::MCMC::new(
        Arc::new(likelihood.clone()),
        prior.clone(),
        Box::new(acceptance::MetropolisHastings::new()),
        "uniform",
    );
    mcmc.sample(100, 100, &mut rng.clone(), propose::uniform_step);
    let result = mcmc.sample(1000, 100, &mut rng.clone(), propose::differential_step);
    result.print_summary();

    let mut nest = NestedSampler::new(Arc::new(likelihood), prior, 500, 1000, 400, "normal", true);
    let result = nest.sample(0.1, rng, 4);
    println!(
        "expected ln evidence: {:.2}",
        ndim as f64 * (PI.sqrt() * erf(25.0 * SQRT_2) / 5.0 / SQRT_2 / 10.0).ln(),
    );
    result.print_summary();
}
