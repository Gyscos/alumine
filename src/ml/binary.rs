use ml::Classifier;

use num::Float;

pub struct Binary<T:Float, C: Classifier<Label=T>>
{
    inner: C,
}

impl <T: Float, C: Classifier<Label=T>> Binary<T,C> {
    pub fn wrap(threshold: T, classifier: C) -> Self {
        Binary {
            inner: classifier,
        }
    }
}

impl <I, T: Float, C: Classifier<Input=I,Label=T>> Classifier for Binary<T,C> {
    type Input = I;
    type Label = bool;

    fn train(&mut self, samples: &[I], labels: &[bool]) {
        let labels: Vec<T> = labels.iter().map(|&b| if b { T::one() } else { T::zero() }).collect();

        self.inner.train(samples, &labels);
    }

    fn classify(&self, input: &I) -> bool {
        let threshold = T::one() / (T::one() + T::one());
        self.inner.classify(input).clone() >= threshold
    }
}
