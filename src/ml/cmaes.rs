use alg::Vector;
use ml::Optimizer;

struct CmaEs {
    // population size
    n: usize,
}

impl CmaEs {
    fn new(n: usize) -> Self {
        CmaEs {
            n: n,
        }
    }
}

impl Optimizer for CmaEs {
    type Input = Vector<f64>;
    type Score = f64;

    fn optimize<F>(&self, f: F) -> Vector<f64>
        where F: Fn(Vector<f64>) -> f64
    {
        Vector::from_copies(0,0f64)
    }
}
