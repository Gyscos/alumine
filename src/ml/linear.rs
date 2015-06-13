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

impl <T: Clone + Num> Classifier for LinearRegression<T> {
    type Input = Vector<T>;
    type Label = T;

    fn train(&mut self, samples: &[Vector<T>], labels: &[T]) {

        let x = Matrix::from_rows(samples);
        let labels = Vector::from_vec(Vec::from(labels));

        let tx = x.transpose();

        self.model = &(&(&tx * &x).invert_inplace() * &tx) * &labels;
    }

    fn classify(&self, input:&Vector<T>) -> T {
        self.model.dot(input)
    }
}
