use core::f64;
use std::f64::NAN;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use lnexp::LnExp;
use logsumexp::LogAddExp;
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;

use super::{acceptance::Threshold, mcmc::MCMC, point::Point};
use crate::{likelihood::Likelihood, propose, NestRng, SampleSpace, SamplerResult, SeedableRng};

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

/// The settings of a [`NestedSampler`].
#[derive(Clone, Debug)]
pub struct SamplerOptions {
    pub nlive: usize,
    /// Initial number of MCMC steps per new point (adapted during the run).
    pub walks: usize,
    /// Target number of accepted MCMC steps per new point.
    pub naccept: usize,
    /// The base space of the MCMC, `"normal"` or `"uniform"`.
    pub sample_space: String,
    pub verbose: bool,
    /// `None` seeds from the operating system's entropy source.
    pub seed: Option<u64>,
    /// Which parameters are periodic (missing entries are not): their MCMC
    /// steps wrap around the ends of the prior.
    pub periodic: Vec<bool>,
    /// Setting this flag (from another thread) stops the run after the
    /// current iteration; the result is then marked `interrupted`.
    pub stop: Option<Arc<AtomicBool>>,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            nlive: 500,
            walks: 100,
            naccept: 60,
            sample_space: "normal".into(),
            verbose: false,
            seed: None,
            periodic: Vec::new(),
            stop: None,
        }
    }
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
    space: SampleSpace,
    verbose: bool,
    dkl: f64,
    _dkl_square: f64,
    _dkl_counts: usize,
    rng: NestRng,
    stop: Option<Arc<AtomicBool>>,
}

impl NestedSampler {
    /// A sampler seeded from the operating system's entropy source.
    pub fn new(
        likelihood: Arc<dyn Likelihood + Send + Sync>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        nlive: usize,
        walks: usize,
        naccept: usize,
        sample_space: &str,
        verbose: bool,
    ) -> Self {
        let seed = NestRng::from_entropy().gen();
        Self::new_seeded(
            likelihood,
            prior,
            nlive,
            walks,
            naccept,
            sample_space,
            verbose,
            seed,
        )
    }

    /// A reproducible sampler: the same seed gives the same run, whatever the
    /// number of threads used to evaluate the likelihood.
    #[allow(clippy::too_many_arguments)]
    pub fn new_seeded(
        likelihood: Arc<dyn Likelihood + Send + Sync>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        nlive: usize,
        walks: usize,
        naccept: usize,
        sample_space: &str,
        verbose: bool,
        seed: u64,
    ) -> Self {
        Self::with_options(
            likelihood,
            prior,
            SamplerOptions {
                nlive,
                walks,
                naccept,
                sample_space: sample_space.to_string(),
                verbose,
                seed: Some(seed),
                periodic: Vec::new(),
                stop: None,
            },
        )
    }

    /// A sampler with all of its settings, including periodic parameters.
    pub fn with_options(
        likelihood: Arc<dyn Likelihood + Send + Sync>,
        prior: Arc<Vec<Box<dyn ContinuousCDF<f64, f64> + Send + Sync>>>,
        options: SamplerOptions,
    ) -> Self {
        let SamplerOptions {
            nlive,
            walks,
            naccept,
            sample_space,
            verbose,
            seed,
            periodic,
            stop,
        } = options;
        let seed = seed.unwrap_or_else(|| NestRng::from_entropy().gen());
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
            space: SampleSpace::new(&sample_space, periodic),
            verbose,
            dkl: 0.0,
            _dkl_square: 0.0,
            _dkl_counts: 0,
            rng: NestRng::seed_from_u64(seed),
            stop,
        };
        // Draw the initial points serially, so they depend only on the seed,
        // and evaluate their likelihoods in parallel.
        let mut rng = new.rng.clone();
        let draws: Vec<Vec<f64>> = (0..nlive).map(|_| new.draw_prior(&mut rng)).collect();
        new.rng = rng;
        let points: Vec<Point> = draws
            .into_par_iter()
            .map(|init| new.point_from_unit(init))
            .collect();
        for point in points {
            new.max_ln_likelihood = new.max_ln_likelihood.max(point.ln_likelihood);
            new.live_points.push(point);
        }
        new
    }

    fn draw_prior(&self, rng: &mut NestRng) -> Vec<f64> {
        self.prior
            .iter()
            .enumerate()
            .map(|(i, _prior)| self.space.from_uniform(i, rng.gen()))
            .collect()
    }

    fn sample_prior(&self, rng: &mut NestRng) -> Point {
        let init = self.draw_prior(rng);
        self.point_from_unit(init)
    }

    fn point_from_unit(&self, init: Vec<f64>) -> Point {
        let scaled = self.rescale(&init);
        Point::new(
            init.clone(),
            scaled.clone(),
            self.ln_prior(&init),
            self.likelihood.ln_likelihood(&scaled),
        )
    }

    fn propose(&self, rng: &mut NestRng, threshold: f64) -> (Point, usize, usize, f64) {
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

    fn init_mcmc(&self, threshold: f64, rng: &mut NestRng) -> MCMC {
        let mut mcmc = MCMC::with_space(
            self.likelihood.clone(),
            self.prior.clone(),
            Box::new(Threshold::new(threshold)),
            self.space.clone(),
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

    fn iterate(&self, threshold: f64, seed: u64) -> ProposalResult {
        let mut rng = NestRng::seed_from_u64(seed);
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

    /// Equally weighted posterior samples by rejection sampling of the dead
    /// points, with the log likelihood of each.
    pub fn posterior_samples(&mut self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let ln_weights = self.ln_posterior_weights();
        let max_weight = ln_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut output = Vec::new();
        let mut ln_likelihood = Vec::new();
        for (point, &weight) in self.dead_points.iter().zip(ln_weights.iter()) {
            if weight - max_weight > self.rng.gen::<f64>().ln() {
                output.push(point.scaled.clone());
                ln_likelihood.push(point.ln_likelihood);
            }
        }
        (output, ln_likelihood)
    }

    /// Whether the run has been asked to stop.
    fn stopped(&self) -> bool {
        self.stop
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Relaxed))
    }

    fn finalize(&mut self, start: &Instant) -> SamplerResult {
        let interrupted = self.stopped();
        self.add_final_live();
        self._print_final();
        let (posterior, posterior_ln_likelihood) = self.posterior_samples();
        let nested_ln_weights: Vec<f64> = self
            .ln_posterior_weights()
            .iter()
            .map(|w| w - self.ln_evidence)
            .collect();
        // Information H = sum_i w_i ln(L_i / Z), and sigma(ln Z) ~ sqrt(H / nlive).
        let information: f64 = self
            .dead_points
            .iter()
            .zip(nested_ln_weights.iter())
            .map(|(point, &w)| {
                if w.is_finite() {
                    w.exp() * (point.ln_likelihood - self.ln_evidence)
                } else {
                    0.0
                }
            })
            .sum();
        SamplerResult {
            posterior,
            posterior_ln_likelihood,
            nested_samples: self.dead_points.iter().map(|p| p.scaled.clone()).collect(),
            nested_ln_likelihood: self.dead_points.iter().map(|p| p.ln_likelihood).collect(),
            nested_ln_weights,
            ln_evidence: self.ln_evidence,
            ln_evidence_error: (information.max(0.0) / self.nlive as f64).sqrt(),
            duration: start.elapsed(),
            interrupted,
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
            .enumerate()
            .map(|(i, (&x, prior))| prior.inverse_cdf(self.space.to_unit(i, x)))
            .collect()
    }

    fn ln_prior(&self, x: &[f64]) -> f64 {
        self.space.ln_prior(x)
    }
}

pub trait SingleThreadedNestedSampler {
    fn sample_serial(&mut self, dlogz: f64) -> SamplerResult;

    fn sample(&mut self, dlogz: f64) -> SamplerResult {
        self.sample_serial(dlogz)
    }
}

impl SingleThreadedNestedSampler for NestedSampler {
    fn sample_serial(&mut self, dlogz: f64) -> SamplerResult {
        let start = Instant::now();
        while self.fraction_remaining() > dlogz && !self.stopped() {
            let replace = self.get_worst();
            self.min_ln_likelihood = replace.1;
            let seed = self.rng.gen();
            let result = self.iterate(replace.1, seed);
            if result.new_point.ln_likelihood > replace.1 {
                self.increment_dkl(&result);
                self.add_point(replace.0, replace.1, result.new_point);
            }
            self.walks = self.update_nmcmc(result.accepted, result.rejected);
            if self.dead_points.len() % 100 == 0 && self.verbose {
                self._print_update(result.accepted as f64);
            }
        }
        self.finalize(&start)
    }
}

pub trait ThreadedNestedSampler {
    fn sample_threaded(&mut self, dlogz: f64, npool: usize) -> SamplerResult;

    fn sample(&mut self, dlogz: f64, npool: usize) -> SamplerResult {
        self.sample_threaded(dlogz, npool)
    }
}

impl ThreadedNestedSampler for NestedSampler {
    fn sample_threaded(&mut self, dlogz: f64, npool: usize) -> SamplerResult {
        let start = Instant::now();
        while self.fraction_remaining() > dlogz && !self.stopped() {
            let mut replace = self.get_worst();
            self.min_ln_likelihood = replace.1;
            // One seed per chain, drawn serially, so the run does not depend on
            // how rayon schedules the chains.
            let seeds: Vec<u64> = (0..npool).map(|_| self.rng.gen()).collect();
            let results: Vec<ProposalResult> = seeds
                .into_par_iter()
                .map(|seed| self.iterate(replace.1, seed))
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
        self.finalize(&start)
    }
}
