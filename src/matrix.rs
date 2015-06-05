use num::Zero;
use std::fmt;
use std::ops::{Index,Add,Mul,Div,Sub};

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

impl <'a, T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: &'a Matrix<T>) -> Matrix<T> {
        Matrix::new(other.n, self.m, |x,y| (0..self.n).map(|i| self[(i,y)].clone() * other[(x,i)].clone()).fold(T::zero(), |a,b|a+b))
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
    pub fn diagonal(n: usize, value: T) -> Self {
        Matrix::new(n, n, |x,y| if x == y { value.clone() } else { T::zero() })
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
    let i3 = Matrix::diagonal(3, 1);
    assert_eq!(i3.n, 3);
    assert_eq!(i3.m, 3);
    assert_eq!(i3[(0,0)], 1);
    assert_eq!(i3[(1,1)], 1);
    assert_eq!(i3[(2,2)], 1);
}

#[test]
fn test_add() {
    let i3 = Matrix::diagonal(3, 1);
    let double = &i3 + &i3;
    assert_eq!(double, Matrix::diagonal(3,2));
}
