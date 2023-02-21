use std::{ops::{Sub, Div}, iter::Sum};

use nalgebra::{DMatrix, SMatrix, ComplexField, Scalar};
use rand::{prelude::Distribution, distributions::Standard};

/// Find the N by K and K by C matrix factors of a staticly-sized matrix R by C.s
pub fn non_negative_matrix_factorization<T, const R: usize, const C: usize, const K: usize> (
	matrix: &SMatrix<T, R, C>,
	max_iter: usize,
	tolerance: T,
) -> (SMatrix<T, R, K>, SMatrix<T, K, C>)
where
	T: Scalar + ComplexField<RealField = T> + Sub<T> + Clone + Copy + Sum<T> + PartialOrd + Div<T, Output = T>,
	Standard: Distribution<T>,
{
	// Two reduced-dimension vectors we are trying to calculate, each field is initialized to [0, 1)
	let mut w: SMatrix<T, R, K> = SMatrix::new_random();
	let mut h: SMatrix<T, K, C> = SMatrix::new_random();

	let mut w_transpose: SMatrix<T, K, R> = SMatrix::zeros();
	let mut h_transpose: SMatrix<T, C, K> = SMatrix::zeros();

	// Repeat until convergence
    for _ in 0..max_iter {
		let wh = w * h;

		// Return if cost is less than tolerance
		let cost = matrix
			.iter()
			.zip(wh.iter())
			.map(|(a, b)| (*a - *b).powi(2))
			.sum::<T>();

		if cost < tolerance { break; }

		// Numerator & denominator for h update
		w.transpose_to(&mut w_transpose);

		let wt_v: SMatrix<T, K, C> = w_transpose * matrix;
		let wt_w_h: SMatrix<T, K, C> = w_transpose * wh;

		// Component-wise update of h
		h.iter_mut().zip(wt_v.iter().zip(wt_w_h.iter())).map(|(h_old, (num, den))| *h_old = *h_old * (*num / *den)).last().unwrap();

		// Numerator & denominator for w update
		h.transpose_to(&mut h_transpose);

		let v_ht: SMatrix<T, R, K> = matrix * h_transpose;
		let w_h_ht: SMatrix<T, R, K> = w * h * h_transpose;

		// Component-wise update of w
		w.iter_mut().zip(v_ht.iter().zip(w_h_ht.iter())).map(|(w_old, (num, den))| *w_old = *w_old * (*num / *den)).last().unwrap();
    }

	(w, h)
}

/// Does the same thing as non_negative_matrix_factorization but using heap-allocated matricies. (Oh how I wish statics and dynamics could be abstracted over...)
pub fn non_negative_matrix_factorization_dyn<T>(
	matrix: &DMatrix<T>,
	k: usize,
	max_iter: usize,
	tolerance: T,
) -> (DMatrix<T>, DMatrix<T>)
where
	T: Scalar + ComplexField<RealField = T> + Sub<T> + Clone + Copy + Sum<T> + PartialOrd + Div<T, Output = T>,
	Standard: Distribution<T>,
{
	let (n_rows, n_cols) = matrix.shape();
	
	let w = &mut DMatrix::new_random(n_rows, k);
	let h = &mut DMatrix::new_random(k, n_cols);

	let w_transpose = &mut DMatrix::zeros(k, n_rows);
	let h_transpose = &mut DMatrix::zeros(n_cols, k);

	let wh = &mut (w.clone() * h.clone());

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
		w.transpose_to(w_transpose);

		let wt_v = &*w_transpose * matrix;
		let wt_w_h = &*w_transpose * &*wh;

		// Component-wise update of h
		h.iter_mut().zip(wt_v.iter().zip(wt_w_h.iter())).map(|(h_old, (num, den))| *h_old = *h_old * (*num / *den)).last().unwrap();

		// Numerator & denominator for w update
		h.transpose_to(h_transpose);

		let v_ht = matrix * h_transpose.clone();
		let w_h_ht = w.clone() * h.clone() * h_transpose.clone();

		// Component-wise update of w
		w.iter_mut().zip(v_ht.iter().zip(w_h_ht.iter())).map(|(w_old, (num, den))| *w_old = *w_old * (*num / *den)).last().unwrap();

		*wh = w.clone() * h.clone();
    }

	(w.clone(), h.clone())
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
		
		let (w, h) = non_negative_matrix_factorization::<_, 4, 5, 3>(&matrix, 1000, 0.01);
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
		let (w_dyn, h_dyn) = non_negative_matrix_factorization_dyn::<f64>(&d_matrix, 3, 1000, 0.01);
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