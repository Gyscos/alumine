use ml::Classifier;

pub struct Binary<C: Classifier<Label=f64>>
{
    inner: C,
}

impl <C: Classifier<Label=f64>> Binary<C> {
    pub fn wrap(classifier: C) -> Self {
        Binary {
            inner: classifier,
        }
    }
}

impl <I,C: Classifier<Input=I,Label=f64>> Classifier for Binary<C> {
    type Input = I;
    type Label = bool;

    fn train(&mut self, samples: &[I], labels: &[bool]) {
        let labels: Vec<f64> = labels.iter().map(|&b| if b { 1f64 } else { 0f64}).collect();

        self.inner.train(samples, &labels);
    }

    fn classify(&self, input: &I) -> bool {
        self.inner.classify(input) > 0.5
    }
}
