use num::Float;

use ml::Classifier;
use alg::{Vector,Matrix};

pub struct MultiLayerPerceptron<T> {
    layers: Vec<Matrix<T>>,
}

impl <T: Float> MultiLayerPerceptron<T> {
    pub fn new(layer_sizes: &[usize]) -> Self {

        // For each pair of layers, build the edge matrix
        let layers = layer_sizes.windows(2)
            .map(|pair| Matrix::zero(pair[0], pair[1]))
            .collect();

        MultiLayerPerceptron {
            layers: layers,
        }
    }
}

fn sigmoid<T: Float>(t: T) -> T {
    T::one() / (T::one() + (-t).exp())
}

impl <T: Clone + Float> Classifier for MultiLayerPerceptron<T> {
    type Input = Vector<T>;
    type Label = Vector<T>;

    fn train(&mut self, samples: &[Vector<T>], labels: &[Vector<T>]) {
        // TODO: actually train the perceptron
    }

    fn classify(&self, input: &Vector<T>) -> Vector<T> {
        self.layers.iter().fold(input.clone(), |a,b| (b*a).chain_apply(sigmoid))
    }
}

#[test]
fn test_mlp() {
    let mlp = MultiLayerPerceptron::<f64>::new(&[5,1]);
}
