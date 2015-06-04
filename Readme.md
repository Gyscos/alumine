Alumine
=======

Components :

Vector<T>
---------

Simple n-dimensionnal vector. Implements:
* Add,Sub(Vector<T>, Vector<T>) -> Vector<T>
* Mul,Div(Vector<T>, T) -> Vector<T>
* DotProduct(Vector<T>, Vector<T>) -> T
* Norm(Vector<T>) -> T

Vector length mismatch in an operation may panic.

Matrix<T>
---------

N x M matrix.
* Add,Sub(M,M)
* Mul,Div(M,T)
* Mul(M,M)
* Mul(M,V)
* Determinant
* Inverse
