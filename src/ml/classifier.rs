pub trait Classifier {
    type Input;
    type Label;

    fn train(&mut self, samples: &[Self::Input], labels: &[Self::Label]);
    fn classify(&self, input: &Self::Input) -> Self::Label;
}
