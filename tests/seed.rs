use std::sync::Arc;

use nestrs::{
    likelihood::Gaussian,
    nest::{NestedSampler, SingleThreadedNestedSampler, ThreadedNestedSampler},
};
use statrs::distribution::{ContinuousCDF, Uniform};

type Prior = Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>;

fn sampler(seed: u64) -> NestedSampler {
    let ndim = 3;
    let prior: Prior = Arc::new(
        (0..ndim)
            .map(|_| {
                Box::new(Uniform::new(0.0, 1.0).unwrap())
                    as Box<dyn ContinuousCDF<f64, f64> + Send + Sync>
            })
            .collect(),
    );
    let likelihood = Gaussian::from_const_mean_std(0.5, 0.1, ndim);
    NestedSampler::new_seeded(
        Arc::new(likelihood),
        prior,
        50,
        20,
        10,
        "uniform",
        false,
        seed,
    )
}

#[test]
fn same_seed_same_serial_run() {
    let a = SingleThreadedNestedSampler::sample(&mut sampler(7), 1.0);
    let b = SingleThreadedNestedSampler::sample(&mut sampler(7), 1.0);
    assert_eq!(a.ln_evidence, b.ln_evidence);
    assert_eq!(a.posterior, b.posterior);
}

#[test]
fn same_seed_same_threaded_run() {
    let a = ThreadedNestedSampler::sample(&mut sampler(7), 1.0, 4);
    let b = ThreadedNestedSampler::sample(&mut sampler(7), 1.0, 4);
    assert_eq!(a.ln_evidence, b.ln_evidence);
    assert_eq!(a.nested_samples, b.nested_samples);
    assert!(a.ln_evidence_error.is_finite());
}

#[test]
fn different_seeds_differ() {
    let a = SingleThreadedNestedSampler::sample(&mut sampler(7), 1.0);
    let b = SingleThreadedNestedSampler::sample(&mut sampler(8), 1.0);
    assert_ne!(a.ln_evidence, b.ln_evidence);
}
