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

 - Uses Lee and Seung's [multiplicative update rule](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms)
 - Generic over allocation strategy, matrix size, and scalar type.
 - Doesn't do allocation in the loop :)
 - 195 lines of code...
 - PRs welcome!

Potential future improvements:
 - Improved / alternative update functions - https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms
 - [Better initialization](https://www.semanticscholar.org/paper/SVD-based-initialization-%3A-A-head-start-for-Boutsidisa-Gallopoulosb/8c2fd0970b065ad14f704da4502684e2bcd89a3f) 
 - Support integers in addition to floats
 - Alternative [evaluation criteria](https://github.com/rhysnewell/nymph/blob/master/src/factorization/nmf.rs#L503) (Frobenius norm, Divergence, Connectivity, etc.)
