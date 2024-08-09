pub trait Likelihood: _LikelihoodClone {
    fn ln_likelihood(&self, x: &[f64]) -> f64;
}

pub trait _LikelihoodClone {
    fn clone_box(&self) -> Box<dyn Likelihood>;
}

impl Clone for Box<dyn Likelihood> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl<T> _LikelihoodClone for T
where
    T: 'static + Likelihood + Clone,
{
    fn clone_box(&self) -> Box<dyn Likelihood> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Rosenbrock {}

impl Likelihood for Rosenbrock {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powf(2.0)).powf(2.0) + (1.0 - x[i]).powf(2.0);
        }
        -sum
    }
}

#[derive(Clone)]
pub struct Gaussian {
    pub mean: Vec<f64>,
    pub prec: Vec<Vec<f64>>,
}

impl Gaussian {
    pub fn new(mean: Vec<f64>, prec: Vec<Vec<f64>>) -> Self {
        Self { mean, prec }
    }

    pub fn from_const_mean_std(mean: f64, std: f64, ndim: usize) -> Self {
        let _prec = std.powf(-2.0);
        let mut prec = vec![vec![_prec; ndim]; ndim];
        for i in 0..ndim {
            prec[i][i] = _prec;
        }
        let mean = vec![mean; ndim];
        Self { mean, prec }
    }
}

impl Likelihood for Gaussian {
    fn ln_likelihood(&self, x: &[f64]) -> f64 {
        x.iter().map(|&x| -(x - 0.5).powf(2.0) / (2.0 * 0.01)).sum()
    }
}

#[derive(Clone)]
pub struct Null {}

impl Likelihood for Null {
    fn ln_likelihood(&self, _x: &[f64]) -> f64 {
        0.0
    }
}
