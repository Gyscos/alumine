use num::{Zero,One};
use num::{Num,Float};
use std::fmt;
use std::ops::{Index,IndexMut,Add,Mul,Div,Sub,Range};

use alg::Vector;

/// Represents a simple `NxM` matrix.
#[derive(Clone,PartialEq,Debug)]
pub struct Matrix<T> {
    /// Number of rows (max Y)
    pub m: usize,
    /// Number of columns (max X)
    pub n: usize,

    // Inner data. Cols concatenated.
    data: Vec<T>,
}

impl <T> Matrix<T> {
    /// Creates a new matrix with the given dimensions,
    /// initializing each cell with the given functor.
    ///
    /// * `m` is the number of lines (the Y size)
    /// * `n` is the number of columns (the X size)
    pub fn new<F>(n: usize, m: usize, f: F) -> Self
        where F: Fn(usize,usize) -> T
    {
        let data = (0..n*m).map(|i| (i/m,i%m)).map(|(x,y)| f(x,y)).collect();
        Matrix {
            m: m,
            n: n,
            data: data,
        }
    }

    pub fn keep_cols(&mut self, cols: Range<usize>) {
        let n_cols = cols.end - cols.start;
        let start = cols.start * self.m;
        let end = cols.end * self.m;

        self.data.truncate(end);

        if start > 0 {
            self.data = self.data.split_off(start);
        }

        self.n = n_cols;
    }

    pub fn append_cols(&mut self, mut other: Matrix<T>) {
        if self.m != other.m {
            panic!("Matrices don't have the same height.");
        }

        self.n += other.n;
        // append is currently unstable. See ../lib.rs
        self.data.append(&mut other.data);
    }

    fn get_index(&self, (x,y): (usize,usize)) -> usize {
        y + x * self.m
    }

    /// Creates a dummy empty matrix.
    pub fn dummy() -> Self {
        Matrix {
            m: 0,
            n: 0,
            data: Vec::new(),
        }
    }

    pub fn is_squared(&self) -> bool {
        return self.m == self.n
    }

    pub fn swap(&mut self, a: (usize,usize), b: (usize,usize)) {
        let ia = self.get_index(a);
        let ib = self.get_index(b);
        self.data.swap(ia, ib);
    }

    pub fn swap_cols(&mut self, xa: usize, xb: usize) {
        if xa == xb { return; }

        for y in 0..self.n {
            self.swap((xa,y), (xb,y));
        }
    }

    pub fn swap_rows(&mut self, ya: usize, yb: usize) {
        if ya == yb { return; }

        for x in 0..self.n {
            self.swap((x,ya),(x,yb));
        }
    }
}

impl <T: Zero> Matrix<T> {
    /// Creates a new matrix initialized to zero.
    pub fn zero(n: usize, m: usize) -> Self {
        Matrix::new(n,m, |_,_| T::zero())
    }
}

impl <T: Clone> Matrix<T> {

    pub fn row(&self, y: usize) -> Vector<T> {
        Vector::new(self.n, |i| self[(i,y)].clone())
    }

    pub fn col(&self, x: usize) -> Vector<T> {
        Vector::new(self.m, |i| self[(x,i)].clone())
    }

    pub fn set_col(&mut self, x: usize, col: Vector<T>) {
        for (i,v) in col.into_iter().enumerate() {
            self[(x,i)] = v;
        }
    }

    pub fn set_row(&mut self, y: usize, row: Vector<T>) {
        for (i,v) in row.into_iter().enumerate() {
            self[(i,y)] = v;
        }
    }

    /// Create a single-column matrix from the given Vector.
    pub fn from_col(v: &Vector<T>) -> Self {
        Matrix::new(1, v.dim(), |_,y| v[y].clone())
    }

    /// Make a matrix from the given vectors, to treat as columns.
    pub fn from_cols(cols: &[Vector<T>]) -> Self {
        if cols.is_empty() {
            Matrix::dummy()
        } else {
            let n = cols.len();
            let m = cols.first().unwrap().dim();
            Matrix::new(n, m, |x,y| cols[x][y].clone())
        }
    }

    /// Create a single-row matrix from the given Vector.
    pub fn from_row(v: &Vector<T>) -> Self {
        Matrix::new(v.dim(), 1, |x,_| v[x].clone())
    }

    /// Make a matrix from the given vectors, to treas as rows.
    pub fn from_rows(rows: &[Vector<T>]) -> Self {
        if rows.is_empty() {
            Matrix::dummy()
        } else {
            let m = rows.len();
            let n = rows.first().unwrap().dim();
            Matrix::new(n, m, |x,y| rows[y][x].clone())
        }
    }

    /// Returns the transposed matrix.
    pub fn transpose(&self) -> Matrix<T> {
        Matrix::new(self.m, self.n, |x,y| self[(y,x)].clone())
    }

    /// If this matrix is single-row or single-column,
    /// transforms this into a Vector.
    pub fn to_vector(self) -> Vector<T> {
        if self.n == 1 || self.m == 1 {
            Vector::from(self.data)
        } else {
            panic!("Matrix is not single-row or single-column.");
        }
    }
}

impl <T: Clone + Zero> Matrix<T> {
    /// Creates a new diagonal matrix, with the given vector serving as the diagonal.
    pub fn diagonal(vector: Vector<T>) -> Self {
        let n = vector.dim();
        let mut m = Matrix::zero(n,n);
        for (i,x) in vector.into_iter().enumerate() {
            m[(i,i)] = x;
        }
        m
    }

    /// Creates a new scalar matrix with the given value.
    pub fn scalar(n: usize, value: T) -> Self {
        Matrix::diagonal(Vector::from_copies(n, value))
    }
}

impl <T: Clone + Zero + One> Matrix<T> {
    pub fn identity(n: usize) -> Self {
        Matrix::scalar(n, T::one())
    }
}

impl <T: Clone + Num> Matrix<T> {
    pub fn norm(&self) -> T {
        self.data.iter().map(|a| a.clone() * a.clone()).fold(T::zero(), |a,b| a+b)
    }

    pub fn determinant(&self) -> T {
        if self.m != self.n { return T::zero(); }

        let mut sum = T::zero();

        let range: Vec<usize> = (0..self.n).collect();

        for sigma in range.permutations() {
            sum = sum + sigma.into_iter().enumerate().map(|(i,j)| self[(i,j)].clone()).fold(T::one(), |a,b| a*b);
        }

        sum
    }

    pub fn inverse(&self) -> Option<Self> {
        self.clone().invert_in_place()
    }

    pub fn invert_in_place(mut self) -> Option<Self> {
        if !self.is_squared() {
            panic!("Attempting to invert a non-square matrix.");
        }

        let n = self.n;

        // Waaa matrix inversion is a complex business.
        // Let's keep it `simple` with a Gauss-Jordan elimination...
        // The idea is: append an Identity matrix to the right (so we have a 2x1 aspect ratio)
        // Apply simple linear row operations (permutation, multiplication, addition) until the
        // first half is an identity matrix.
        // At this point, the second half should be the inverse matrix.
        // (Since we apply the inverse of the first half to an identity matrix.)

        self.append_cols(Matrix::scalar(n, T::one()));

        // For each (original) column...
        for k in 0..n {
            // Make sure the column is C[i] = i==k ? 1 : 0

            // Find the perfect candidate: a non-zero element
            let j = match (k..n).find(|&i| self[(k,i)] != T::zero()) {
                None => return None,
                Some(j) => j,
            };

            self.swap_rows(j, k);

            // Now divide the row by the diagonal value
            let pivot = self[(k,k)].clone();
            // No need to divide the first k values, they should be zeroes
            for x in k..self.n {
                self[(x,k)] = self[(x,k)].clone() / pivot.clone();
            }

            // Finally, zero all other rows
            for y in (0..n).filter(|&i| i != k) {
                let value = self[(k,y)].clone();
                for x in k..self.n {
                    self[(x,y)] = self[(x,y)].clone() - value.clone() * self[(x,k)].clone();
                }
            }
        }

        // And remove the first half
        self.keep_cols(n..2*n);

        Some(self)
    }

}

impl <T: Clone + Float> Matrix<T> {

    pub fn cholesky(&self) -> Self {
        self.clone().cholesky_in_place()
    }

    /// Returns a matrix `L` such that `L * L.transpose() == self`. Assumes that self is symmetric.
    pub fn cholesky_in_place(mut self) -> Self {
        let n = self.n;
        for x in 0..n {
            for y in 0..x { self[(x,y)] = T::zero(); }

            let sum = (0..x).map(|i| self[(i,x)]).map(|v| v*v).fold(T::zero(), |a,b| a+b);
            let ljj = (self[(x,x)] - sum).sqrt();
            self[(x,x)] = ljj;
            let iv = ljj.recip();

            for y in x+1..n {
                let sum = (0..x).map(|i| self[(i,x)] * self[(i,y)]).fold(T::zero(), |a,b| a+b);
                let lij = (self[(x,y)] - sum) * iv;
                self[(x,y)] = lij;
            }
        }

        self
    }
}

impl <T: Clone + Mul<Output=T> + Add<Output=T> + Zero> Mul for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        &self * &other
    }
}

impl <'a,'b, T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: &'b Matrix<T>) -> Matrix<T> {
        // TODO: might want to make this faster. Parallel? Skip zero values?
        Matrix::new(other.n, self.m, |x,y| (0..self.n).map(|i| self[(i,y)].clone() * other[(x,i)].clone()).fold(T::zero(), |a,b|a+b))
    }
}

impl <T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, other: Vector<T>) -> Vector<T> {
        &self * &other
    }
}

impl <'a, T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul<Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, other: Vector<T>) -> Vector<T> {
        self * &other
    }
}

impl <'a,'b, T: Add<Output=T> + Mul<Output=T> + Zero + Clone> Mul<&'b Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::new(self.m, |i| (0..self.n).map(|j| self[(j,i)].clone() * other[j].clone()).fold(T::zero(), |a,b| a+b))
    }
}

impl <'a, T: Add<Output=T> + Clone> Add for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: &'a Matrix<T>) -> Matrix<T> {
        Matrix::new(self.n, self.m, |x,y| self[(x,y)].clone() + other[(x,y)].clone())
    }
}

impl <T: Sub<Output=T> + Clone> Sub for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(mut self, other: Matrix<T>) -> Matrix<T> {
        for (s,o) in self.data.iter_mut().zip(other.data.into_iter()) {
            *s = s.clone() - o;
        }

        self
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

impl <T> Index<(usize,usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (x,y): (usize,usize)) -> &T {
        let i = self.get_index((x,y));
        &self.data[i]
    }
}

impl <T> IndexMut<(usize,usize)> for Matrix<T> {
    fn index_mut(&mut self, (x,y): (usize,usize)) -> &mut T {
        let i = self.get_index((x,y));
        &mut self.data[i]
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
    let i3 = Matrix::scalar(3, 1);
    assert_eq!(i3.n, 3);
    assert_eq!(i3.m, 3);
    assert_eq!(i3[(0,0)], 1);
    assert_eq!(i3[(1,1)], 1);
    assert_eq!(i3[(2,2)], 1);
}

#[test]
fn test_add() {
    let i3 = Matrix::scalar(3, 1);
    let double = &i3 + &i3;
    assert_eq!(double, Matrix::scalar(3,2));
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
fn test_cols() {
    let a = Matrix::new(4,4, |x,y| x+y);
    let b = Matrix::new(4,4, |x,y| 8 - (x+y));

    let mut c = a.clone();
    c.append_cols(b.clone());
    c.keep_cols(4..8);
    assert_eq!(b, c);

    let mut d = b.clone();
    d.append_cols(a.clone());
    d.keep_cols(4..8);
    assert_eq!(a, d);
}

#[test]
fn test_determinant() {
    let i5 = Matrix::<f64>::identity(5);
    assert_eq!(i5.determinant(), 1f64);
}

#[test]
fn test_inverse() {
    let a = Matrix::new(2,2, |x,y| (x+y) as f64);
    let b = a.inverse().unwrap();

    assert_eq!((&a * &b), Matrix::identity(2));
}

#[test]
fn test_mul() {
    let m = Matrix::new(4,4, |x,y| x+y);
    let v = Vector::new(4, |i| i+1);

    assert_eq!(Matrix::from_col(&(&m * &v)), &m * &Matrix::from_col(&v));
}
