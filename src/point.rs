#[derive(Debug)]
pub struct Point {
    pub x: Vec<f64>,
    pub scaled: Vec<f64>,
    pub logpdf: f64,
    pub ln_prior: f64,
    pub ln_likelihood: f64,
}

impl Clone for Point {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            scaled: self.scaled.clone(),
            logpdf: self.logpdf,
            ln_prior: self.ln_prior,
            ln_likelihood: self.ln_likelihood,
        }
    }
}

impl Point {
    pub fn new(x: Vec<f64>, scaled: Vec<f64>, ln_prior: f64, ln_likelihood: f64) -> Self {
        Self {
            x,
            scaled,
            logpdf: ln_prior + ln_likelihood,
            ln_prior,
            ln_likelihood,
        }
    }
}
