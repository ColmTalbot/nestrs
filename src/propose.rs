use rand::{rngs::ThreadRng, Rng};

use super::{mcmc::MCMC, point::Point};

fn validate_proposal(mcmc: &MCMC, point: &Point, x_prime: &Vec<f64>) -> Point {
    let ln_prior = mcmc.ln_prior(&x_prime);
    if ln_prior == -f64::INFINITY {
        point.clone()
    } else {
        let scaled = mcmc.rescale(&x_prime);
        Point::new(
            x_prime.clone(),
            scaled.clone(),
            mcmc.ln_prior(&x_prime),
            mcmc.likelihood.ln_likelihood(&scaled),
        )
    }
}

#[allow(dead_code)]
pub fn uniform_step(mcmc: &mut MCMC, point: &Point, rng: &mut ThreadRng) -> Point {
    let x_prime: Vec<f64> = point
        .x
        .iter()
        .map(|x| x + (rng.gen::<f64>() - 0.5) / 10.0)
        .collect();
    validate_proposal(mcmc, point, &x_prime)
}

#[allow(dead_code)]
pub fn differential_step(mcmc: &mut MCMC, point: &Point, rng: &mut ThreadRng) -> Point {
    let first = rng.gen_range(0..mcmc.samples.len());
    let second = rng.gen_range(0..mcmc.samples.len());
    let scale: f64 = 1.0 / (2.38 * point.x.len() as f64).sqrt();
    let delta: Vec<f64> = mcmc.samples[first]
        .x
        .iter()
        .zip(mcmc.samples[second].x.iter())
        .map(|(x, y)| (x - y))
        .collect();
    let x_prime: Vec<f64> = point
        .x
        .iter()
        .zip(delta.iter())
        .map(|(x, d)| x + d * scale)
        .collect();
    validate_proposal(mcmc, point, &x_prime)
}
