use std::sync::Arc;

use nestrs::{
    acceptance::MetropolisHastings,
    likelihood::Null,
    mcmc::MCMC,
    nest::{NestedSampler, SingleThreadedNestedSampler},
    propose::{differential_step, uniform_step},
    Result,
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

    let mut mcmc = MCMC::new(
        Arc::new(Null {}),
        prior.clone(),
        Box::new(MetropolisHastings::new()),
        "uniform",
    );
    mcmc.sample(100, 100, &mut rng.clone(), uniform_step);
    let result = mcmc.sample(10000, 200, &mut rng.clone(), differential_step);
    result.print_summary();

    let mut nest = NestedSampler::new(
        Arc::new(Null {}),
        prior.clone(),
        500,
        1000,
        40,
        "normal",
        false,
    );
    let result = nest.sample_serial(0.1, rng);
    result.print_summary();
}
