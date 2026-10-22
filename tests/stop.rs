use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nestrs::{
    likelihood::Likelihood,
    nest::{NestedSampler, SamplerOptions, SingleThreadedNestedSampler, ThreadedNestedSampler},
};
use statrs::distribution::{ContinuousCDF, Uniform};

type Prior = Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>;

/// A narrow Gaussian that takes a while to evaluate, so that a full run
/// would take far longer than the test allows.
#[derive(Clone)]
struct Slow;

impl Likelihood for Slow {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        std::thread::sleep(Duration::from_micros(200));
        -x.iter().map(|x| ((x - 0.5) / 1e-3).powi(2)).sum::<f64>() / 2.0
    }
}

fn sampler(stop: Arc<AtomicBool>) -> NestedSampler {
    let prior: Prior = Arc::new(
        (0..2)
            .map(|_| {
                Box::new(Uniform::new(0.0, 1.0).unwrap())
                    as Box<dyn ContinuousCDF<f64, f64> + Send + Sync>
            })
            .collect(),
    );
    NestedSampler::with_options(
        Arc::new(Slow),
        prior,
        SamplerOptions {
            nlive: 50,
            walks: 20,
            naccept: 10,
            seed: Some(1),
            stop: Some(stop),
            ..Default::default()
        },
    )
}

/// Sets `stop` after `delay`, from another thread.
fn stop_after(stop: &Arc<AtomicBool>, delay: Duration) -> std::thread::JoinHandle<()> {
    let stop = stop.clone();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        stop.store(true, Ordering::Relaxed);
    })
}

#[test]
fn stop_flag_ends_a_serial_run() {
    let stop = Arc::new(AtomicBool::new(false));
    let mut sampler = sampler(stop.clone());
    let setter = stop_after(&stop, Duration::from_millis(300));
    let start = Instant::now();
    let result = SingleThreadedNestedSampler::sample(&mut sampler, 0.1);
    setter.join().unwrap();
    assert!(result.interrupted);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
    assert!(!result.nested_samples.is_empty());
}

#[test]
fn stop_flag_ends_a_threaded_run() {
    let stop = Arc::new(AtomicBool::new(false));
    let mut sampler = sampler(stop.clone());
    let setter = stop_after(&stop, Duration::from_millis(300));
    let start = Instant::now();
    let result = ThreadedNestedSampler::sample(&mut sampler, 0.1, 4);
    setter.join().unwrap();
    assert!(result.interrupted);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn unset_flag_runs_to_completion() {
    let stop = Arc::new(AtomicBool::new(false));
    let prior: Prior = Arc::new(vec![Box::new(Uniform::new(0.0, 1.0).unwrap())]);
    let mut sampler = NestedSampler::with_options(
        Arc::new(nestrs::likelihood::Gaussian::from_const_mean_std(
            0.5, 0.1, 1,
        )),
        prior,
        SamplerOptions {
            nlive: 50,
            walks: 20,
            naccept: 10,
            seed: Some(2),
            stop: Some(stop),
            ..Default::default()
        },
    );
    let result = ThreadedNestedSampler::sample(&mut sampler, 0.1, 2);
    assert!(!result.interrupted);
    assert!(result.ln_evidence.is_finite());
}
