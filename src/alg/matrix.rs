use num::Zero;
use std::fmt;
use std::ops::{Index,Add,Mul,Div,Sub};

use alg::Vector;

#[derive(Clone,PartialEq,Debug)]
pub struct Matrix<T> {
    /// Number of rows (max Y)
    pub m: usize,
    /// Number of columns (max X)
    pub n: usize,

    data: Vec<T>,
}

impl <T> Matrix<T> {
    pub fn new<F>(n: usize, m: usize, f: F) -> Self
        where F: Fn(usize,usize) -> T
    {
        let data = (0..n*m).map(|i| (i%n,i/n)).map(|(x,y)| f(x,y)).collect();
        Matrix {
            m: m,
            n: n,
            data: data,
        }
    }
}

impl <T: Zero> Matrix<T> {
    pub fn zero(n: usize, m: usize) -> Self {
        Matrix::new(n,m, |_,_| T::zero())
    }
}

impl <T: Clone> Matrix<T> {

    pub fn from_col(v: &Vector<T>) -> Self {
        Matrix::new(1, v.len(), |_,y| v[y].clone())
    }

    pub fn from_row(v: &Vector<T>) -> Self {
        Matrix::new(v.len(), 1, |x,_| v[x].clone())
    }

    pub fn transpose(&self) -> Matrix<T> {
        Matrix::new(self.m, self.n, |x,y| self[(y,x)].clone())
    }

    pub fn to_vector(&self) -> Vector<T> {
        if self.n == 1 || self.m == 1 {
            Vector::from_vec(self.data.clone())
        } else {
            panic!("Matrix is not single-row or single-column.");
        }
    }
}

impl <'a, T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: &'a Matrix<T>) -> Matrix<T> {
        // TODO: might want to make this faster. Parallel? Skip zero values?
        Matrix::new(other.n, self.m, |x,y| (0..self.n).map(|i| self[(i,y)].clone() * other[(x,i)].clone()).fold(T::zero(), |a,b|a+b))
    }
}

impl <'a, T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul<&'a Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, other: &'a Vector<T>) -> Vector<T> {
        Vector::new(self.m, |i| (0..self.n).map(|j| self[(j,i)].clone() * other[j].clone()).fold(T::zero(), |a,b| a+b))
    }
}

impl <'a, T: Add<Output=T> + Clone> Add for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: &'a Matrix<T>) -> Matrix<T> {
        Matrix::new(self.n, self.m, |x,y| self[(x,y)].clone() + other[(x,y)].clone())
    }
}

impl <'a, T: Sub<Output=T> + Clone> Sub for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: &'a Matrix<T>) -> Matrix<T> {
        Matrix::new(self.n, self.m, |x,y| self[(x,y)].clone() - other[(x,y)].clone())
    }
}

impl <'a, T: Mul<Output=T> + Clone> Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: T) -> Matrix<T> {
        Matrix::new(self.n, self.m, |x,y| self[(x,y)].clone() * other.clone())
    }
}

impl <'a, T: Div<Output=T> + Clone> Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, other: T) -> Matrix<T> {
        Matrix::new(self.n, self.m, |x,y| self[(x,y)].clone() / other.clone())
    }
}

impl <T: Clone + Zero> Matrix<T> {
    pub fn diagonal<F>(n: usize, f: F) -> Self
        where F: Fn(usize) -> T
    {
        Matrix::new(n, n, |x,y| if x == y { f(x) } else { T::zero() })
    }

    pub fn identity(n: usize, value: T) -> Self {
        Matrix::diagonal(n, |_| value.clone())
    }
}

impl <T> Index<(usize,usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (x,y): (usize,usize)) -> &T {
        let i = x + y * self.n;
        &self.data[i]
    }
}

impl <T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(),fmt::Error> {
        try!(fmt.write_str(&format!("[{} x {}]", self.n, self.m)));
        for y in 0..self.m {
            try!(fmt.write_str("["));
            for x in 0..self.n {
                try!(self[(x,y)].fmt(fmt));
                try!(fmt.write_str(", "));
            }
            try!(fmt.write_str("]\n"));
        }
        Ok(())
    }
}

#[test]
fn test_i3() {
    let i3 = Matrix::identity(3, 1);
    assert_eq!(i3.n, 3);
    assert_eq!(i3.m, 3);
    assert_eq!(i3[(0,0)], 1);
    assert_eq!(i3[(1,1)], 1);
    assert_eq!(i3[(2,2)], 1);
}

#[test]
fn test_add() {
    let i3 = Matrix::identity(3, 1);
    let double = &i3 + &i3;
    assert_eq!(double, Matrix::identity(3,2));
}

#[test]
fn test_transpose() {
    let v = Vector::new(4, |i| 1+i);

    assert_eq!(Matrix::from_row(&v), Matrix::from_col(&v).transpose());

    let m = Matrix::new(4,3, |x,y| x+4*y);
    let n = Matrix::new(3,4, |x,y| y+4*x);

    assert!(m != n);
    assert_eq!(m.transpose().transpose(), m);
    assert_eq!(m.transpose(), n);
    assert_eq!(n.transpose(), m);
}

#[test]
fn test_transpose_commutation() {
    let a = Matrix::new(5,3, |x,y| x+5*y);
    let b = Matrix::new(4,5, |x,y| x+y + 2*(x+y)%2);

    assert_eq!((&a * &b).transpose(), &b.transpose() * &a.transpose());
}

#[test]
fn test_from_row() {
    let v = Vector::new(4, |i| 1+i);
    let m = Matrix::new(4,1, |x,_| 1+x);

    assert_eq!(Matrix::from_row(&v), m);
    assert_eq!(m.to_vector(), v);
}

#[test]
fn test_sub() {
    let a = Matrix::new(4,3, |x,y| x+y);
    assert_eq!(&a - &a, Matrix::zero(4,3));
}

#[test]
fn test_mul() {
    let m = Matrix::new(4,4, |x,y| x+y);
    let v = Vector::new(4, |i| i+1);

    assert_eq!(Matrix::from_col(&(&m * &v)), &m * &Matrix::from_col(&v));
}
