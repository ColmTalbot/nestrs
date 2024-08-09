use core::f64;
use std::f64::NAN;
use std::sync::Arc;
use std::time::Instant;

use lnexp::LnExp;
use logsumexp::LogAddExp;
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::FullContinuous;

use super::{acceptance::Threshold, mcmc::MCMC, point::Point};
use crate::{propose, SamplerResult, _get_sample_space, likelihood::Likelihood};

fn logsubexp(a: f64, b: f64) -> f64 {
    a + (b - a).ln_1m_exp()
}

#[derive(Debug)]
struct ProposalResult {
    new_point: Point,
    accepted: usize,
    rejected: usize,
    ln_ratio: f64,
}

pub struct NestedSampler {
    likelihood: Arc<dyn Likelihood + Send + Sync>,
    prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
    live_points: Vec<Point>,
    dead_points: Vec<Point>,
    threshold: f64,
    nlive: usize,
    pub ln_evidence: f64,
    min_ln_likelihood: f64,
    max_ln_likelihood: f64,
    walks: f64,
    naccept: usize,
    _sample_space: String,
    sample_space: Arc<Box<dyn FullContinuous<f64, f64> + Send + Sync>>,
    verbose: bool,
    dkl: f64,
    _dkl_square: f64,
    _dkl_counts: usize,
}

impl NestedSampler {
    pub fn new(
        likelihood: Arc<dyn Likelihood + Send + Sync>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        nlive: usize,
        walks: usize,
        naccept: usize,
        sample_space: &str,
        verbose: bool,
    ) -> Self {
        let sample_space = sample_space.to_string().to_lowercase();
        let mut new = Self {
            likelihood,
            prior,
            live_points: Vec::new(),
            dead_points: Vec::new(),
            threshold: -f64::INFINITY,
            nlive,
            ln_evidence: -f64::INFINITY,
            min_ln_likelihood: f64::INFINITY,
            max_ln_likelihood: -f64::INFINITY,
            walks: walks as f64,
            naccept,
            sample_space: _get_sample_space(&sample_space),
            _sample_space: sample_space,
            verbose,
            dkl: 0.0,
            _dkl_square: 0.0,
            _dkl_counts: 0,
        };
        for _ in 0..nlive {
            let point = new.sample_prior(&mut ThreadRng::default());
            new.max_ln_likelihood = new.max_ln_likelihood.max(point.ln_likelihood);
            new.live_points.push(point);
        }
        new
    }

    fn sample_prior(&self, rng: &mut ThreadRng) -> Point {
        let init: Vec<f64> = self
            .prior
            .iter()
            .map(|_prior| self.sample_space.inverse_cdf(rng.gen()))
            .collect();
        let scaled = self.rescale(&init);
        Point::new(
            init.clone(),
            scaled.clone(),
            self.ln_prior(&init),
            self.likelihood.ln_likelihood(&scaled),
        )
    }

    fn propose(&self, rng: &mut ThreadRng, threshold: f64) -> (Point, usize, usize, f64) {
        let mut point = self.sample_prior(rng);
        if point.ln_likelihood > threshold {
            return (point, 0, 0, NAN);
        }
        let mut mcmc = self.init_mcmc(threshold, rng);
        let start = mcmc.samples.last().unwrap().clone();
        let other = mcmc.samples.first().unwrap().clone();
        while point.ln_likelihood < threshold {
            mcmc.sample(
                1,
                self.walks.ceil() as usize,
                rng,
                propose::differential_step,
            );
            point = mcmc.samples.pop().unwrap();
        }
        let mut _dist = 0.0;
        start.x.iter().zip(point.x.iter()).for_each(|(a, b)| {
            _dist += (a - b).abs().powf(2.0);
        });
        let mut _alt_dist = 0.0;
        start.x.iter().zip(other.x.iter()).for_each(|(a, b)| {
            _alt_dist += (a - b).abs().powf(2.0);
        });
        (
            point,
            mcmc.accepted,
            mcmc.rejected,
            _alt_dist.ln() - _dist.ln(),
        )
    }

    fn init_mcmc(&self, threshold: f64, rng: &mut ThreadRng) -> MCMC {
        let mut mcmc = MCMC::new(
            self.likelihood.clone(),
            self.prior.clone(),
            Box::new(Threshold::new(threshold)),
            &self._sample_space,
        );
        mcmc.samples = self.live_points.clone();
        mcmc.samples.shuffle(rng);
        mcmc
    }

    fn update_nmcmc(&self, accepted: usize, rejected: usize) -> f64 {
        if accepted > 0 {
            let accept_prob = (accepted as f64).max(0.5) / (accepted + rejected) as f64;
            let delay = self.nlive as f64 / 10.0 - 1.0;
            let n_target = self.naccept;
            (self.walks * delay + n_target as f64 / accept_prob) / (delay + 1.0)
        } else {
            self.walks
        }
    }

    fn ln_prior_volume(&self, idx: usize) -> f64 {
        -(idx as f64) / self.nlive as f64
    }

    fn ln_prior_weight(&self, idx: usize) -> f64 {
        logsubexp(self.ln_prior_volume(idx), self.ln_prior_volume(idx + 1))
    }

    fn get_worst(&self) -> (usize, f64) {
        let (min_idx, worst) = self
            .live_points
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.ln_likelihood.partial_cmp(&b.1.ln_likelihood).unwrap())
            .unwrap();
        (min_idx, worst.ln_likelihood)
    }

    fn iterate(&self, threshold: f64) -> ProposalResult {
        let mut rng = ThreadRng::default();
        let (new_point, accepted, rejected, ln_ratio) = self.propose(&mut rng, threshold);
        ProposalResult {
            new_point,
            accepted,
            rejected,
            ln_ratio,
        }
    }

    fn add_point(&mut self, worst_idx: usize, threshold: f64, new_point: Point) {
        if new_point.ln_likelihood <= threshold {
            return;
        }
        let mut dead_point = self.live_points.swap_remove(worst_idx);
        dead_point.ln_prior = self.ln_prior_weight(self.dead_points.len());
        self.ln_evidence = self
            .ln_evidence
            .ln_add_exp(dead_point.ln_prior + dead_point.ln_likelihood);
        self.threshold = dead_point.ln_likelihood;
        self.max_ln_likelihood = self.max_ln_likelihood.max(new_point.ln_likelihood);
        self.live_points.push(new_point);
        self.dead_points.push(dead_point);
    }

    fn ln_posterior_weights(&self) -> Vec<f64> {
        self.dead_points
            .iter()
            .map(|point| point.ln_likelihood + point.ln_prior)
            .collect()
    }

    pub fn posterior_samples(&self, rng: &mut ThreadRng) -> Vec<Vec<f64>> {
        let ln_weights = self.ln_posterior_weights();
        let max_weight = ln_weights
            .iter()
            .max_by(|&a, &b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut output = Vec::new();
        for (point, &weight) in self.dead_points.iter().zip(ln_weights.iter()) {
            if weight - max_weight > rng.gen::<f64>().ln() {
                output.push(point.scaled.clone());
            }
        }
        output
    }

    fn finalize(&mut self, start: &Instant, rng: &mut ThreadRng) -> SamplerResult {
        self.add_final_live();
        self._print_final();
        SamplerResult {
            ln_evidence: self.ln_evidence,
            posterior: self.posterior_samples(rng),
            duration: start.elapsed(),
        }
    }

    fn _print_update(&self, accepted: f64) {
        println!(
            "iter: {}, ln Z {:.2}, min ln L {:.2}, max ln L {:.2}, Fraction remaining: {:.2}, Walks: {:.1}, Accepted: {}, DKL: {:.2e}",
            self.dead_points.len(),
            self.ln_evidence,
            self.threshold,
            self.max_ln_likelihood,
            self.fraction_remaining(),
            self.walks,
            accepted,
            self.dkl,
        );
    }

    fn _print_final(&self) {
        println!(
            "iter: {}, ln Z {:.2}, DKL: {:.2e} +/- {:.2e}",
            self.dead_points.len(),
            self.ln_evidence,
            self.dkl,
            (self._dkl_square - self.dkl.powf(2.0)).powf(0.5),
        );
    }

    fn fraction_remaining(&self) -> f64 {
        if self.max_ln_likelihood == self.min_ln_likelihood {
            println!("All live points have equal likelihood, exiting.");
            0.0
        } else if self.dead_points.is_empty() {
            f64::INFINITY
        } else {
            (self.ln_prior_volume(self.dead_points.len()) + self.max_ln_likelihood
                - self.ln_evidence)
                .ln_1p_exp()
        }
    }

    fn add_final_live(&mut self) {
        let ln_prior = self.ln_prior_volume(self.dead_points.len()) - (self.nlive as f64).ln();
        for point in self.live_points.iter() {
            self.ln_evidence = self.ln_evidence.ln_add_exp(ln_prior + point.ln_likelihood);
            self.dead_points.push(Point {
                ln_prior,
                ..point.clone()
            });
        }
    }

    fn increment_dkl(&mut self, result: &ProposalResult) {
        if result.ln_ratio.is_finite() {
            self.dkl = (self.dkl * self._dkl_counts as f64 + result.ln_ratio)
                / (self._dkl_counts as f64 + 1.0);
            self._dkl_square = (self._dkl_square * self._dkl_counts as f64
                + result.ln_ratio.powf(2.0))
                / (self._dkl_counts as f64 + 1.0);
            self._dkl_counts += 1;
        }
    }

    fn rescale(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.prior.iter())
            .map(|(&x, prior)| {
                let x = self.sample_space.cdf(x);
                prior.inverse_cdf(x)
            })
            .collect()
    }

    fn ln_prior(&self, x: &[f64]) -> f64 {
        x.iter().map(|&x| self.sample_space.ln_pdf(x)).sum()
    }
}

pub trait SingleThreadedNestedSampler {
    fn sample_serial(&mut self, dlogz: f64, rng: &mut ThreadRng) -> SamplerResult;

    fn sample(&mut self, dlogz: f64, rng: &mut ThreadRng) -> SamplerResult {
        self.sample_serial(dlogz, rng)
    }
}

impl SingleThreadedNestedSampler for NestedSampler {
    fn sample_serial(&mut self, dlogz: f64, rng: &mut ThreadRng) -> SamplerResult {
        let start = Instant::now();
        while self.fraction_remaining() > dlogz {
            let replace = self.get_worst();
            self.min_ln_likelihood = replace.1;
            let result = self.iterate(replace.1);
            if result.new_point.ln_likelihood > replace.1 {
                self.increment_dkl(&result);
                self.add_point(replace.0, replace.1, result.new_point);
            }
            self.walks = self.update_nmcmc(result.accepted, result.rejected);
            if self.dead_points.len() % 100 == 0 && self.verbose {
                self._print_update(result.accepted as f64);
            }
        }
        self.finalize(&start, rng)
    }
}

pub trait ThreadedNestedSampler {
    fn sample_threaded(&mut self, dlogz: f64, rng: &mut ThreadRng, npool: usize) -> SamplerResult;

    fn sample(&mut self, dlogz: f64, rng: &mut ThreadRng, npool: usize) -> SamplerResult {
        self.sample_threaded(dlogz, rng, npool)
    }
}

impl ThreadedNestedSampler for NestedSampler {
    fn sample_threaded(&mut self, dlogz: f64, rng: &mut ThreadRng, npool: usize) -> SamplerResult {
        let start = Instant::now();
        while self.fraction_remaining() > dlogz {
            let mut replace = self.get_worst();
            self.min_ln_likelihood = replace.1;
            let results: Vec<ProposalResult> = (0..npool)
                .into_par_iter()
                .map(|_| self.iterate(replace.1))
                .collect();
            let mut accepted: usize = 0;
            let mut rejected: usize = 0;
            for result in results {
                if result.new_point.ln_likelihood > replace.1 {
                    self.increment_dkl(&result);
                    self.add_point(replace.0, replace.1, result.new_point);
                    accepted += result.accepted;
                    rejected += result.rejected;
                    replace = self.get_worst();
                    self.min_ln_likelihood = replace.1;
                    if self.dead_points.len() % 100 == 0 && self.verbose {
                        self._print_update(accepted as f64 / npool as f64);
                    }
                }
            }
            self.walks = self.update_nmcmc(accepted, rejected);
        }
        self.finalize(&start, rng)
    }
}
