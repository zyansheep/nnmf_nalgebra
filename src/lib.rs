use std::{ops::{Sub, Div}, iter::Sum};

use nalgebra::{DMatrix, SMatrix, ComplexField, Scalar, Dim, Matrix, DefaultAllocator, allocator::Allocator, Const, Dyn};
use rand::{prelude::Distribution, distributions::Standard};

/// Does non-negative matrix factorization using multiplicative static update rule as defined on the [wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms) page. Is generic over the allocation strategy of the Matrix.
pub fn non_negative_matrix_factorization_generic<T, R: Dim, C: Dim, K: Dim> (
	matrix: &Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>,
	max_iter: usize,
	tolerance: T,
	nrows: R,
	ncols: C,
	k: K,
) -> (Matrix<T, R, K, <DefaultAllocator as Allocator<T, R, K>>::Buffer>, Matrix<T, K, C, <DefaultAllocator as Allocator<T, K, C>>::Buffer>)
where
	T: Scalar + ComplexField<RealField = T> + Sub<T> + Clone + Copy + Sum<T> + PartialOrd + Div<T, Output = T>,
	Standard: Distribution<T>,
	DefaultAllocator: Allocator<T, R, C>,
	DefaultAllocator: Allocator<T, R, K>,
	DefaultAllocator: Allocator<T, K, C>,
	DefaultAllocator: Allocator<T, K, R>,
	DefaultAllocator: Allocator<T, C, K>,
{
	// Two reduced-dimension vectors we are trying to calculate, each field is initialized to [0, 1)
	let mut w: Matrix<T, R, K, _> = Matrix::new_random_generic(nrows, k);
	let mut h: Matrix<T, K, C, _> = Matrix::new_random_generic(k, ncols);

	let mut w_transpose: Matrix<T, K, R, _> = Matrix::zeros_generic(k, nrows);
	let mut h_transpose: Matrix<T, C, K, _> = Matrix::zeros_generic(ncols, k);

	let mut wh: Matrix<T, R, C, _> = &w * &h;

	// Repeat until convergence
    for _ in 0..max_iter {

		// Return if cost is less than tolerance
		let cost = matrix
			.iter()
			.zip(wh.iter())
			.map(|(a, b)| (*a - *b).powi(2))
			.sum::<T>();

		if cost < tolerance { break; }

		// Numerator & denominator for h update
		w.transpose_to(&mut w_transpose);

		let wt_v: Matrix<T, K, C, _> = &w_transpose * matrix;
		let wt_w_h: Matrix<T, K, C, _> = &w_transpose * wh;

		// Component-wise update of h
		h.iter_mut().zip(wt_v.iter().zip(wt_w_h.iter())).map(|(h_old, (num, den))| *h_old = *h_old * (*num / *den)).last().unwrap();

		// Numerator & denominator for w update
		h.transpose_to(&mut h_transpose);

		let v_ht: Matrix<T, R, K, _> = matrix * &h_transpose;
		let w_h_ht: Matrix<T, R, K, _> = &w * &h * &h_transpose;

		// Component-wise update of w
		w.iter_mut().zip(v_ht.iter().zip(w_h_ht.iter())).map(|(w_old, (num, den))| *w_old = *w_old * (*num / *den)).last().unwrap();
		
		wh = &w * &h;
	}

	(w, h)
}

/// Does non-negative matrix factorization on a statically-sized matrix (SMatrix)
pub fn non_negative_matrix_factorization_static<T, const R: usize, const C: usize, const K: usize>(
	matrix: &SMatrix<T, R, C>,
	max_iter: usize,
	tolerance: T
) -> (SMatrix<T, R, K>, SMatrix<T, K, C>)
where
T: Scalar + ComplexField<RealField = T> + Sub<T> + Clone + Copy + Sum<T> + PartialOrd + Div<T, Output = T>,
	Standard: Distribution<T>,
{
	non_negative_matrix_factorization_generic(matrix, max_iter, tolerance, Const::<R>, Const::<C>, Const::<K>)
}

/// Does non-negative matrix factorization on a dynamically-sized matrix (DMatrix)
pub fn non_negative_matrix_factorization_dyn<T>(
	matrix: &DMatrix<T>,
	max_iter: usize,
	tolerance: T,
	k: usize,
) -> (DMatrix<T>, DMatrix<T>)
where
T: Scalar + ComplexField<RealField = T> + Sub<T> + Clone + Copy + Sum<T> + PartialOrd + Div<T, Output = T>,
	Standard: Distribution<T>,
{
	let (nrows, ncols) = matrix.shape();
	non_negative_matrix_factorization_generic(matrix, max_iter, tolerance, Dyn(nrows), Dyn(ncols), Dyn(k))
}

#[cfg(test)]
mod tests {
    use nalgebra::{matrix, SMatrix, dmatrix};

    use crate::*;

	#[test]
	fn test_static_and_dyn() {
		let matrix: SMatrix<f64, 4, 5> = matrix![
			1.0, 2.0, 0.0, 30.0, 1.5;
			0.0, 3.0, 1.0, 30.0, 1.6;
			5.0, 1.0, 10.0, 30.0, 1.7;
			3.0, 1.0, 10.0, 30.0, 1.7
		];
		
		let (w, h) = non_negative_matrix_factorization_static::<_, 4, 5, 3>(&matrix, 1000, 0.01);
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
		let (w_dyn, h_dyn) = non_negative_matrix_factorization_dyn::<f64>(&d_matrix, 1000, 0.01, 3);
		println!("{w}\n{h}");
		let prediction = w * h;

		println!("Matrix: {}\n Predic: {}", matrix, prediction);
		assert!(d_matrix.relative_eq(&(w_dyn.clone() * h_dyn.clone()), 0.5, 0.5));

		/* assert_eq!(w, w_dyn);
		assert_eq!(h, h_dyn); */

		/* let mut nmf = FixedTemplateNmf::new(templates, activation_coef, &input, 0.5);

		for _ in 1..5 {
			nmf.update_activation_coef();
		}

		let activation = nmf.get_activation_coef();
		let result = max_activation_vector(&activation);

		// Note that the max here is index 1 (2.0, 3.0, 1.0)
		assert_eq!(result, Vector::new(vec![0.0038612997, 0.113134526, 0.003651987, 0.057484213, 0.054535303])); */
	}
}