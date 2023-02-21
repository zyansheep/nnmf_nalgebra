# Non-negative Matrix Factorization (with nalgebra)

Takes a matrix V of size R*C, splits it into two smaller matricies W (R\*K) and H (K\*C) such that W * V approximates V.

 - Uses [multiplicative update rule](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms)
 - Generic over allocation strategy, matrix size, and scalar type.
 - Doesn't do allocation in the loop :)
 - 195 lines of code!
 - PRs welcome!