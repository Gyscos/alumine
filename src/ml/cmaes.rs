use std::marker::PhantomData;

use rand::distributions::IndependentSample;
use rand::distributions::Normal;
use rand::{thread_rng,Rng};
use num::Float;

use alg::{Vector,Matrix};
use ml::Optimizer;

/// Covariance Matrix Adaptation Evolution Strategy is a general purpose
/// black-box optimization algorithm.
pub struct CmaEs<T: Float> {
    // population size
    pop: usize,
    // Dimension of the input vector
    n: usize,
    // Steps
    t: usize,

    phantom: PhantomData<T>,
}

impl <T: Float> CmaEs<T> {
    /// Creates a new CMA-ES optimizer.
    ///
    /// - `n`: dimension of the function input vector
    /// - `pop`: size of the population.
    /// - `t`: number of generations to simulate.
    pub fn new(n: usize, pop: usize, t: usize) -> Self {
        CmaEs {
            n: n,
            pop: pop,
            t: t,
            phantom: PhantomData,
        }
    }
}

impl <T: Float> Optimizer for CmaEs<T> {
    type Input = Vector<T>;
    type Score = T;

    fn optimize<F>(&self, f: F) -> Vector<T>
        where F: Fn(Vector<T>) -> T
    {
        CmaEsSlave::new(self.n, self.pop).optimize(f, self.t)
    }
}

struct CmaEsSlave<T: Float> {
    // These are stable
    n: usize,
    pop: usize,
    weights: Vec<T>,

    // These vary
    population: Matrix<T>,
    covariance: Matrix<T>,
    mean: Vector<T>,
}

impl <T: Float> CmaEsSlave<T> {

    fn new(n: usize, pop: usize) -> Self {
        CmaEsSlave {
            n: n,
            pop: pop,
            weights: Vec::new(),

            population: Matrix::zero(pop, n),
            covariance: Matrix::zero(n,n),
            mean: Vector::zero(n),
        }
    }

    fn optimize<F>(mut self, f: F, t: usize) -> Vector<T>
        where F: Fn(Vector<T>) -> T
    {
        self.compute_weights();

        for _ in 0..t {
            // Generate population
            self.generate_population();

            // Adapt covariance and mean
            self.adapt_covariance(&f);
        }

        self.mean
    }

    fn compute_weights(&mut self) {
        let mu = self.n / 2;
        self.weights.reserve(mu);

        let mut sum = T::zero();
        let logmu = T::from(mu as f64 + 0.5).unwrap().log10();
        for i in 0..mu {
            let w = logmu - T::from(i+1).unwrap().log10();
            self.weights.push(w);
            sum = sum + w;
        }

        for w in self.weights.iter_mut() {
            *w = *w / sum;
        }
    }

    fn adapt_covariance<F>(&mut self, f: &F)
        where F: Fn(Vector<T>) -> T
    {
        let scores: Vec<T> = (0..self.pop).map(|i| self.population.row(i)).map(|sample| f(sample)).collect();
        // Maybe sort the population by its fitness?

        // The mean is easy...
        let mean = (0..self.pop)
            .map(|i| self.population.row(i))
            .zip(self.weights.iter())
            .fold(Vector::zero(self.n), |a,(b,&w)| a+b*w);
        let mean = mean / T::from(self.pop).unwrap();
        self.mean = mean;

        // Now, for the covariance...
    }

    fn generate_population(&mut self) {
        // Lower triangular cholesky decomposition
        let L = self.covariance.cholesky();

        let mut r = thread_rng();
        let d = Normal::new(0f64, 1f64);

        for i in 0..self.pop {
            self.population.set_col(i, CmaEsSlave::generate_sample(&self.mean, &L, &mut r, &d));
        }
    }

    fn generate_sample<R: Rng, D: IndependentSample<f64>>(mean: &Vector<T>, L: &Matrix<T>, r: &mut R, d: &D) -> Vector<T> {
        // Multivariate sampling from covariance matrix
        let normal = Vector::new(mean.dim(), |_| T::from(d.ind_sample(r)).unwrap());
        mean + L * normal
    }
}
