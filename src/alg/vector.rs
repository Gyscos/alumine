use num::Zero;
use std::ops::{Add,Sub,Mul,Div,Index};

#[derive(Clone,PartialEq,Debug)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl <T> Vector<T> {
    pub fn new<F>(n: usize, f: F) -> Self
        where F: Fn(usize) -> T
    {
        let data = (0..n).map(|i| f(i)).collect();
        Vector {
            data: data,
        }
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        Vector {
            data: data,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl <T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl <T: Zero> Vector<T> {
    pub fn zero(n: usize) -> Self {
        Vector::new(n, |_| T::zero())
    }
}

impl <T: Clone> Vector<T> {
    pub fn from_copies(n: usize, model: T) -> Self {
        Vector::new(n, |_| model.clone())
    }
}

impl <'a,T: Clone + Add<Output=T>> Add for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, other: &'a Vector<T>) -> Vector<T> {
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

impl <T: Clone + Mul<Output=T> + Add<Output=T> + Zero> Vector<T> {
    pub fn dot(&self, other: &Vector<T>) -> T {
        self.data.iter().zip(other.data.iter()).map(|(a,b)| a.clone()*b.clone()).fold(T::zero(), |a,b| a+b)
    }

    pub fn norm_sq(&self) -> T {
        self.dot(self)
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
