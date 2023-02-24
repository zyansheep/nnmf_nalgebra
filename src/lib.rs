use std::{
    iter::Sum,
    ops::{Div, Sub, AddAssign, MulAssign},
};

use nalgebra::{
    allocator::Allocator, ComplexField, Const, DMatrix, DefaultAllocator, Dim, Dyn, Matrix,
    SMatrix, Scalar,
};
use num::{Zero, One, Num};
use rand::{distributions::Standard, prelude::Distribution};

/// Does non-negative matrix factorization using multiplicative static update rule as defined on the [wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms) page. Is generic over the allocation strategy of the Matrix.
pub fn non_negative_matrix_factorization_generic<T, R: Dim, C: Dim, K: Dim>(
    matrix: &Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>,
    max_iter: usize,
    tolerance: T,
    mut init_fn: impl FnMut(usize, usize) -> T,
    nrows: R,
    ncols: C,
    k: K,
) -> (
    Matrix<T, R, K, <DefaultAllocator as Allocator<T, R, K>>::Buffer>,
    Matrix<T, K, C, <DefaultAllocator as Allocator<T, K, C>>::Buffer>,
)
where
    T: Scalar
        + Num
        + Clone
        + Copy + Sum<T> + AddAssign<T> + MulAssign<T> + PartialOrd,
    Standard: Distribution<T>,
    DefaultAllocator: Allocator<T, R, C>,
    DefaultAllocator: Allocator<T, R, K>,
    DefaultAllocator: Allocator<T, K, C>,
    DefaultAllocator: Allocator<T, K, R>,
    DefaultAllocator: Allocator<T, C, K>,
{
    // Two reduced-dimension vectors we are trying to calculate, each field is initialized to [0, 1)
    let mut w: Matrix<T, R, K, _> = Matrix::from_fn_generic(nrows, k, |a,b|init_fn(a,b));
    let mut h: Matrix<T, K, C, _> = Matrix::from_fn_generic(k, ncols, |a,b|init_fn(a,b));

    let mut w_transpose: Matrix<T, K, R, _> = Matrix::zeros_generic(k, nrows);
    let mut h_transpose: Matrix<T, C, K, _> = Matrix::zeros_generic(ncols, k);

    let mut wh: Matrix<T, R, C, _> = &w * &h;

    let mut wt_v: Matrix<T, K, C, _> = Matrix::zeros_generic(k, ncols);
    let mut wt_w_h: Matrix<T, K, C, _> = Matrix::zeros_generic(k, ncols);

    let mut v_ht: Matrix<T, R, K, _> = Matrix::zeros_generic(nrows, k);
    let mut w_h_ht: Matrix<T, R, K, _> = Matrix::zeros_generic(nrows, k);

    // Repeat until convergence
    for _ in 0..max_iter {
        // Return if cost is less than tolerance
        let cost = matrix
            .iter()
            .zip(wh.iter())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff * diff
            })
            .sum::<T>();

        if cost < tolerance {
            break;
        }

        // Calculate W^T
        w.transpose_to(&mut w_transpose);

        // let wt_v: Matrix<T, K, C, _> = &w_transpose * matrix;
        // Numerator = W^T * V
        w_transpose.mul_to(matrix, &mut wt_v);
        // Denominator = W^T * W * H
        w_transpose.mul_to(&wh, &mut wt_w_h);

        // Component-wise update of h
        h.iter_mut()
            .zip(wt_v.iter().zip(wt_w_h.iter()))
            .map(|(h_old, (num, den))| *h_old *= *num / *den)
            .last()
            .unwrap();

        // Calculate H^T
        h.transpose_to(&mut h_transpose);

        // WH = W * H
        w.mul_to(&h, &mut wh);

        // Numerator = V * H^T
        matrix.mul_to(&h_transpose, &mut v_ht);
        // Denominator = W * H * H^T
        wh.mul_to(&h_transpose, &mut w_h_ht);

        // Component-wise update of w
        w.iter_mut()
            .zip(v_ht.iter().zip(w_h_ht.iter()))
            .map(|(w_old, (num, den))| *w_old *= *num / *den)
            .last()
            .unwrap();

        w.mul_to(&h, &mut wh);
    }

    (w, h)
}

/// Does non-negative matrix factorization on a statically-sized matrix (SMatrix)
pub fn non_negative_matrix_factorization_static<T, const R: usize, const C: usize, const K: usize>(
    matrix: &SMatrix<T, R, C>,
    max_iter: usize,
    tolerance: T,
    init_fn: impl FnMut(usize, usize) -> T,
) -> (SMatrix<T, R, K>, SMatrix<T, K, C>)
where
    T: Scalar
        + Num
        + Clone
        + Copy
        + Sum<T>
        + PartialOrd
        + AddAssign<T>
        + MulAssign<T>,
    Standard: Distribution<T>,
{
    non_negative_matrix_factorization_generic(
        matrix, max_iter, tolerance, init_fn, Const::<R>, Const::<C>, Const::<K>,
    )
}

/// Does non-negative matrix factorization on a dynamically-sized matrix (DMatrix)
pub fn non_negative_matrix_factorization_dyn<T>(
    matrix: &DMatrix<T>,
    max_iter: usize,
    tolerance: T,
    k: usize,
    init_fn: impl FnMut(usize, usize) -> T
) -> (DMatrix<T>, DMatrix<T>)
where
    T: Scalar
        + ComplexField<RealField = T>
        + Sub<T>
        + Clone
        + Copy
        + Sum<T>
        + PartialOrd
        + Div<T, Output = T>,
    Standard: Distribution<T>,
{
    let (nrows, ncols) = matrix.shape();
    non_negative_matrix_factorization_generic(
        matrix,
        max_iter,
        tolerance,
        init_fn,
        Dyn(nrows),
        Dyn(ncols),
        Dyn(k),
    )
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, matrix, SMatrix};
    use rand::{Rng, thread_rng};

    use crate::*;

    #[test]
    fn test_static_and_dyn() {
        let thread_rng = &mut thread_rng();
        let mut init_fn = |_,_| thread_rng.gen();

        let matrix: SMatrix<f64, 4, 5> = matrix![
            1.0, 2.0, 0.0, 30.0, 1.5;
            0.0, 3.0, 1.0, 30.0, 1.6;
            5.0, 1.0, 10.0, 30.0, 1.7;
            3.0, 1.0, 10.0, 30.0, 1.7
        ];

        let (w, h) = non_negative_matrix_factorization_static::<_, 4, 5, 3>(&matrix, 1000, 0.01, &mut init_fn);
        println!("{w}\n{h}");
        let prediction = w * h;

        println!("Matrix: {}\n Predic: {}", matrix, prediction);
        assert!(matrix.relative_eq(&(w * h), 0.5, 0.5));

        let d_matrix: DMatrix<f64> = dmatrix![
            1.0, 2.0, 0.0, 30.0, 1.5;
            0.0, 3.0, 1.0, 30.0, 1.6;
            5.0, 1.0, 10.0, 30.0, 1.7;
            3.0, 1.0, 10.0, 30.0, 1.7
        ];
        let (w_dyn, h_dyn) = non_negative_matrix_factorization_dyn::<f64>(&d_matrix, 1000, 0.01, 3, &mut init_fn);
        println!("{w}\n{h}");
        let prediction = w * h;

        println!("Matrix: {}\n Predic: {}", matrix, prediction);
        assert!(d_matrix.relative_eq(&(w_dyn.clone() * h_dyn.clone()), 0.5, 0.5));
    }

    #[test]
    fn test_integer() {
        let rng = &mut thread_rng();

        let matrix: SMatrix<i64, 5, 5> = matrix![
            0000, 0200, 5100, 3000, 5000;
            0300, 0000, 3000, 2100, 3100;
            5000, 0100, 0000, 0100, 1900;
            3000, 2000, 0200, 0000, 0300;
            5000, 3000, 2000, 0200, 0000
        ];

        let (w, h) = non_negative_matrix_factorization_static::<_, 5, 5, 3>(&matrix, 1000, 10, &mut |_,_| rng.gen_range(10..1000));
        println!("{w}\n{h}");
        let prediction = w * h;

        println!("Matrix: {}\n Predic: {}", matrix, prediction);

        let cost = matrix
            .iter()
            .zip(prediction.iter())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff * diff
            })
            .sum::<i64>();
        assert!(cost < 100);
    }
}
