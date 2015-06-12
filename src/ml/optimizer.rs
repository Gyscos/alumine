pub trait Optimizer {
    type Input;
    type Score;

    fn optimize<F>(&self, f: F) -> Self::Input
        where F: Fn(Self::Input) -> Self::Score;
}
