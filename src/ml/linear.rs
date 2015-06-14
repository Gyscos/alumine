use std::fmt::Debug;

use num::Num;

use ml::Classifier;
use alg::{Vector,Matrix};

pub struct LinearRegression<T> {
    model: Vector<T>,
}

impl <T : Num> LinearRegression<T> {
    pub fn new(n: usize) -> Self {
        LinearRegression {
            model: Vector::zero(n),
        }
    }
}

impl <T: Clone + Num + Debug> Classifier for LinearRegression<T> {
    type Input = Vector<T>;
    type Label = T;

    fn train(&mut self, samples: &[Vector<T>], labels: &[T]) {

        let x = Matrix::from_rows(samples);
        let labels = Vector::from(Vec::from(labels));

        let tx = x.transpose();

        let inv_txx = match (&tx * &x).invert_inplace() {
            None => return,
            Some(m) => m,
        };
        self.model = &(&inv_txx * &tx) * &labels;
    }

    fn classify(&self, input: &Vector<T>) -> T {
        self.model.dot(input)
    }
}
