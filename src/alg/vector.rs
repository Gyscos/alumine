use num::Zero;
use std::vec;
use std::ops::{Add,Sub,Mul,Div,Index};

use alg::Matrix;

/// Represents a `N`-dimensional vector.
#[derive(Clone,PartialEq,Debug)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl <T> Vector<T> {
    /// Creates a new vector, filling the values with the given functor.
    pub fn new<F>(n: usize, mut f: F) -> Self
        where F: FnMut(usize) -> T
    {
        let data = (0..n).map(|i| f(i)).collect();
        Vector::from_vec(data)
    }

    /// Returns a 0-dimensional empty vector.
    pub fn dummy() -> Self {
        Vector::from_vec(Vec::new())
    }

    /// Creates a vector directly from the given vec.
    pub fn from_vec(data: Vec<T>) -> Self {
        Vector {
            data: data,
        }
    }

    /// Return the dimension of the vector: the number of values.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn into_iter(self) -> vec::IntoIter<T> {
        self.data.into_iter()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
}

impl <T> From<Vec<T>> for Vector<T> {
    // Creates a vector directly from the given vec.
    fn from(data: Vec<T>) -> Self {
        Vector::from_vec(data)
    }
}

impl <'a,T: Clone> From<&'a [T]> for Vector<T> {
    fn from(data: &'a [T]) -> Self {
        Vector::from_slice(data)
    }
}

impl <T: Zero> Vector<T> {
    /// Creates a zero vector of the given dimension.
    pub fn zero(n: usize) -> Self {
        Vector::new(n, |_| T::zero())
    }
}

impl <T: Clone> Vector<T> {
    /// Creates a vector with all values cloned from the given value.
    pub fn from_copies(n: usize, model: T) -> Self {
        Vector::new(n, |_| model.clone())
    }

    pub fn from_slice(data: &[T]) -> Self {
        Vector::from_vec(Vec::from(data))
    }

}

impl <T: Clone + Mul<Output=T> + Add<Output=T> + Zero> Vector<T> {
    /// Returns the dot product between two vectors.
    pub fn dot(&self, other: &Vector<T>) -> T {
        self.data.iter().zip(other.data.iter()).map(|(a,b)| a.clone()*b.clone()).fold(T::zero(), |a,b| a+b)
    }

    /// Returns the squared norm of this vector.
    pub fn norm_sq(&self) -> T {
        self.dot(self)
    }

    pub fn outer_product(&self, other: &Vector<T>) -> Matrix<T> {
        Matrix::from_col(self) * Matrix::from_row(other)
    }
}

impl <T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl <T: Clone + Add<Output=T>> Vector<T> {
    pub fn add_in_place(&mut self, other: &Vector<T>) {
        for (s,o) in self.data.iter_mut().zip(other.data.iter()) {
            *s = s.clone() + o.clone();
        }
    }
}

impl <T: Clone + Add<Output=T>> Add for Vector<T> {
    type Output = Vector<T>;

    fn add(mut self, other: Vector<T>) -> Vector<T> {
        self.add_in_place(&other);
        self
    }
}

impl <'a,T: Clone + Add<Output=T>> Add<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, other: Vector<T>) -> Vector<T> {
        self + &other
    }
}

impl <'a,'b,T: Clone + Add<Output=T>> Add<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, other: &'b Vector<T>) -> Vector<T> {
        let data = self.data.iter().zip(other.data.iter())
            .map(|(a,b)| a.clone()+b.clone()).collect();

        Vector {
            data: data,
        }
    }
}

impl <'a, T: Clone + Sub<Output=T>> Sub for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, other: &'a Vector<T>) -> Vector<T> {
        let data = self.data.iter().zip(other.data.iter())
            .map(|(a,b)| a.clone()-b.clone()).collect();

        Vector {
            data: data,
        }
    }
}

impl <'a, T: Mul<Output=T> + Clone> Mul<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, other: T) -> Vector<T> {
        let data = self.data.iter().map(|a| a.clone()*other.clone()).collect();

        Vector {
            data: data,
        }
    }
}

impl <T: Div<Output=T> + Clone> Div<T> for Vector<T> {
    type Output = Vector<T>;

    fn div(mut self, other: T) -> Vector<T> {
        for s in self.data.iter_mut() {
            *s = s.clone() / other.clone();
        }
        self
    }
}

impl <'a, T: Div<Output=T> + Clone> Div<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn div(self, other: T) -> Vector<T> {
        let data = self.data.iter().map(|a| a.clone()/other.clone()).collect();

        Vector {
            data: data,
        }
    }
}

#[test]
fn test_dot() {
    let a = Vector::from_copies(5, 1);
    let b = Vector::from_copies(5, 2);

    let norm = a.dot(&b);

    assert_eq!(norm, 10);
}
