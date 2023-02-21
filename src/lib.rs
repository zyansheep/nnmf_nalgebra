use nalgebra::{DMatrix, SMatrix};

pub fn non_negative_matrix_factorization<const R: usize, const C: usize, const K: usize> (
	matrix: &SMatrix<f64, R, C>,
	max_iter: usize,
	tolerance: f64,
) -> (SMatrix<f64, R, K>, SMatrix<f64, K, C>)
where
	// T: Scalar + ComplexField<RealField = T> + PartialOrd + Sum<T>,
	// Standard: Distribution<T>,
{
	// Two reduced-dimension vectors we are trying to calculate, each field is initialized to [0, 1)
	let mut w: SMatrix<f64, R, K> = SMatrix::new_random();
	let mut h: SMatrix<f64, K, C> = SMatrix::new_random();

	let mut w_transpose: SMatrix<f64, K, R> = SMatrix::zeros();
	let mut h_transpose: SMatrix<f64, C, K> = SMatrix::zeros();

	// Repeat until convergence
    for _ in 0..max_iter {
		let wh = w * h;

		// Return if cost is less than tolerance
		let cost = matrix
			.iter()
			.zip(wh.iter())
			.map(|(a, b)| (a - b).powi(2))
			.sum::<f64>();

		if cost < tolerance { break; }

		// Numerator & denominator for h update
		w.transpose_to(&mut w_transpose);

		let wt_v = w_transpose * matrix;
		let wt_w_h = w_transpose * wh;

		// Component-wise update of h
		h.iter_mut().zip(wt_v.iter().zip(wt_w_h.iter())).map(|(h_old, (num, den))| *h_old = *h_old * (num / den)).last().unwrap();

		// Numerator & denominator for w update
		h.transpose_to(&mut h_transpose);

		let v_ht = matrix * h_transpose;
		let w_h_ht = w * h * h_transpose;

		// Component-wise update of w
		w.iter_mut().zip(v_ht.iter().zip(w_h_ht.iter())).map(|(w_old, (num, den))| *w_old = *w_old * (num / den)).last().unwrap();
    }

	(w, h)
}

pub fn non_negative_matrix_factorization_dyn(
	matrix: &DMatrix<f64>,
	k: usize,
	max_iter: usize,
	tolerance: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
	let (n_rows, n_cols) = matrix.shape();
	
	let w = &mut DMatrix::from_fn(n_rows, k, |_i, _j| rand::random::<f64>());
	let h = &mut DMatrix::from_fn(k, n_cols, |_i, _j| rand::random::<f64>());

	let w_transpose = &mut w.clone();
	let h_transpose = &mut h.clone();

	let wh = &mut (w.clone() * h.clone());

	// Repeat until convergence
    for _ in 0..max_iter {

		// Return if cost is less than tolerance
		let cost = matrix
			.iter()
			.zip(wh.iter())
			.map(|(a, b)| (a - b).powi(2))
			.sum::<f64>();

		if cost < tolerance { break; }

		// Numerator & denominator for h update
		w.transpose_to(w_transpose);

		let wt_v = w_transpose.clone() * matrix;
		let wt_w_h = w_transpose.clone() * wh.clone();

		// Component-wise update of h
		h.iter_mut().zip(wt_v.iter().zip(wt_w_h.iter())).map(|(h_old, (num, den))| *h_old = *h_old * (num / den)).last();

		// Numerator & denominator for w update
		h.transpose_to(h_transpose);

		let v_ht = matrix * h_transpose.clone();
		let w_h_ht = w.clone() * h.clone() * h_transpose.clone();

		// Component-wise update of w
		w.iter_mut().zip(v_ht.iter().zip(w_h_ht.iter())).map(|(w_old, (num, den))| *w_old = *w_old * (num / den)).last();

		*wh = w.clone() * h.clone();
    }

	(w.clone(), h.clone())
}


#[cfg(test)]
mod tests {
    use nalgebra::{matrix, DMatrix, SMatrix};

    use crate::non_negative_matrix_factorization;

	#[test]
	fn put_it_to_the_test() {
		let matrix: SMatrix<f64, 4, 5> = matrix![
			1.0, 2.0, 0.0, 30.0, 1.5;
			0.0, 3.0, 1.0, 30.0, 1.6;
			5.0, 1.0, 10.0, 30.0, 1.7;
			3.0, 1.0, 10.0, 30.0, 1.7
		];

		/* let w: SMatrix<f32, 3, 1> = matrix![
			2.1;
			3.2;
			0.9
		];

		let h: SMatrix<f32, 1, 5> = matrix![3.2, 2.0, 1.0, 5.0, 3.0]; */


		/* assert_eq!(matrix, w * h); */
		let (w, h) = non_negative_matrix_factorization::<4, 5, 4>(&matrix, 1000, 0.01);
		println!("{w}\n{h}");
		let prediction = w * h;

		println!("Matrix: {}\n Predic: {}", matrix, prediction);
		assert!(matrix.relative_eq(&(w * h), 0.5, 0.5));



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