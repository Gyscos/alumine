//! Machine-learning module
//!
//! The main traits are the `Classifier` and the `Optimizer`. Various implementations are provided.
mod classifier;
mod optimizer;

pub mod binary;
pub mod linear;
pub mod bayes;
pub mod cmaes;
pub mod mlp;

pub use self::classifier::Classifier;
pub use self::optimizer::Optimizer;
