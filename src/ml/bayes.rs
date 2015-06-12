use alg::Vector;
use ml::Classifier;

pub struct NaiveBayes {
    k: usize,
}

pub enum Value {
    Double(f32),
    Integer(i32),
    Boolean(bool),
}

impl NaiveBayes {
    pub fn new(k: usize) -> Self {
        NaiveBayes {
            k: k,
        }
    }
}

impl Classifier for NaiveBayes {
    type Input = Vector<Value>;
    type Label = usize;

    fn train(&mut self, samples: &[Vector<Value>], labels: &[usize]) {
    }

    fn classify(&self, input: &Vector<Value>) -> usize {
        0
    }
}
