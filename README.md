# Non-negative Matrix Factorization (with nalgebra)
<p>
    <a href="https://docs.rs/nnmf_nalgebra">
        <img src="https://img.shields.io/docsrs/nnmf_nalgebra.svg" alt="docs.rs">
    </a>
    <a href="https://crates.io/crates/nnmf_nalgebra">
        <img src="https://img.shields.io/crates/v/nnmf_nalgebra.svg" alt="crates.io">
    </a>
</p>


Takes a matrix V of size R*C, splits it into two smaller matricies W (R\*K) and H (K\*C) such that W * V approximates V.

 - Uses [multiplicative update rule](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms)
 - Generic over allocation strategy, matrix size, and scalar type.
 - Doesn't do allocation in the loop :)
 - 195 lines of code!
 - PRs welcome!