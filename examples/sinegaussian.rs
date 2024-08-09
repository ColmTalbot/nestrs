use std::f64::consts::PI;
use std::sync::Arc;

use nestrs::{
    acceptance,
    likelihood::Likelihood,
    mcmc,
    nest::{NestedSampler, ThreadedNestedSampler},
    propose, Result,
};
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use statrs::distribution::{ContinuousCDF, Normal, Uniform};

#[derive(Clone)]
pub struct TransientLikelihood {
    data: Vec<f64>,
    psd: Vec<f64>,
    model: fn(&[f64], usize) -> Vec<f64>,
}

impl Likelihood for TransientLikelihood {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        let signal = (self.model)(x, self.data.len());
        let mut sum = 0.0;
        let norm = 100.0 / signal.len() as f64;
        self.data
            .iter()
            .zip(signal)
            .zip(self.psd.iter())
            .for_each(|((y, x), p)| {
                sum += 2.0 * y * x / (2.0 * p);
                sum -= x.powf(2.0) / (2.0 * p);
            });
        sum * norm
    }
}

fn sine_gaussian(x: &[f64], len: usize) -> Vec<f64> {
    let frequency = x[0];
    let damping = x[1];
    let phase = x[2];
    let time = x[3];
    let amplitude = x[4];
    let mut signal = vec![0.0; len];
    let omega = 2.0 * PI * frequency;
    signal.iter_mut().enumerate().for_each(|(idx, x)| {
        let t = idx as f64 / len as f64 - time;
        *x = amplitude * (t * omega + phase).sin() * (-damping * t.powf(2.0) / 2.0).exp();
    });
    signal
}

fn main() {
    let rng = &mut ThreadRng::default();

    let truth = vec![50.0, 10.0, 0.6, 0.5, 1.0];

    let datalen = 4096;

    let noise: Vec<f64> = Normal::new(0.0, 0.1)
        .unwrap()
        .sample_iter(rng.clone())
        .take(datalen)
        .collect();
    let data = sine_gaussian(&truth, datalen);
    let data: Vec<f64> = data.iter().zip(noise).map(|(x, y)| x + y).collect();

    let prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>> = Arc::new(vec![
        Box::new(Uniform::new(0.0, 100.0).unwrap()),
        Box::new(Uniform::new(0.0, 20.0).unwrap()),
        Box::new(Uniform::new(-PI, PI).unwrap()),
        Box::new(Uniform::new(0.0, 1.0).unwrap()),
        Box::new(Uniform::new(0.0, 2.0).unwrap()),
    ]);

    let likelihood = Arc::new(TransientLikelihood {
        data,
        psd: vec![0.01; datalen],
        model: sine_gaussian,
    });

    let mut mcmc = mcmc::MCMC::new(
        likelihood.clone(),
        prior.clone(),
        Box::new(acceptance::MetropolisHastings::new()),
        "normal",
    );
    mcmc.sample(100, 100, &mut rng.clone(), propose::uniform_step);
    let result = mcmc.sample(1000, 100, &mut rng.clone(), propose::differential_step);
    result.print_summary();

    let mut nest = NestedSampler::new(
        likelihood.clone(),
        prior.clone(),
        250,
        100,
        60,
        "normal",
        true,
    );
    let result = nest.sample(0.1, rng, 5);
    result.print_summary();
}
