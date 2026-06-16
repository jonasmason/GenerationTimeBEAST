---
editor_options: 
  markdown: 
    wrap: 72
---

# CSparseJSolver.java explained

## What this file does

A wrapper around the CSparseJ library that solves a sparse symmetric
positive definite linear system `A x = rhs` using sparse Cholesky
factorisation.

------------------------------------------------------------------------

## Background

### Why a sparse solver?

The posterior precision matrix `C` (see `GenerationTimeGibbsOperator`)
is sparse. For a tree with n taxa, the matrix is roughly (2n-1) ×
(2n-1), but has only O(n) non-zero entries (two or three per edge).
Using a dense solver would cost O(n³) per step; a sparse Cholesky solver
exploits the sparsity to run in near-linear time for typical
phylogenetic trees.

### Compressed Sparse Column (CSC) format

A sparse matrix is stored in three arrays:

-   `colPtr[j]` — the index into `rowIdx` and `values` where column j
    starts. `colPtr` has length `n + 1`; `colPtr[n]` equals the total
    number of non-zeros.

-   `rowIdx[k]` — the row index of the k-th stored entry.

-   `values[k]` — the numeric value of the k-th stored entry.

So the non-zeros in column j are at positions `colPtr[j]` through
`colPtr[j+1] - 1` in `rowIdx` and `values`.

### Cholesky factorisation

For a symmetric positive definite matrix A, Cholesky writes A = L L\^T
where L is lower-triangular. To solve A x = rhs:

1\. Solve L z = rhs (forward substitution)

2\. Solve L\^T x = z (backward substitution)

CSparseJ works with a fill-reducing permutation P to minimise fill-in in
L:

```         
P A P^T = L L^T
```

Solving becomes:

1\. y = P rhs

2\. Solve L z = y

3\. Solve L\^T z = z (in-place)

4\. x = P\^T z

CSparseJ splits the factorisation into two phases:

-   **Symbolic** (`cs_schol`): analyses the sparsity pattern and
    computes the permutation P. Expensive; done once per tree topology.

-   **Numeric** (`cs_chol`): computes the actual L given numeric values.
    Done once per MCMC step.

------------------------------------------------------------------------

## Line-by-line walkthrough

### Imports

``` java
import edu.emory.mathcs.csparsej.tdouble.Dcs_chol;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcsn;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;
import edu.emory.mathcs.csparsej.tdouble.Dcs_ipvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_lsolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_ltsolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_pvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_schol;
```

These are the specific CSparseJ classes needed:

\- `Dcs` — the sparse matrix structure (double precision).

-   `Dcss` — the symbolic factorisation result (holds permutation `pinv`
    and symbolic L structure).

-   `Dcsn` — the numeric factorisation result (holds numeric L).

-   `Dcs_schol` — performs symbolic Cholesky analysis.

-   `Dcs_chol` — performs numeric Cholesky factorisation.

-   `Dcs_lsolve` — solves L x = b in-place.

-   `Dcs_ltsolve` — solves L\^T x = b in-place.

-   `Dcs_ipvec` — applies the inverse permutation: y[pinv[k]] = b[k].

-   `Dcs_pvec` — applies the permutation: x[k] = y[pinv[k]].

------------------------------------------------------------------------

### Fields

``` java
    private final int dimension;
    private Dcs matrix;
    private Dcss symbolic;
    private Dcsn numeric;
    private boolean patternAnalysed = false;
    private boolean numericFactorised = false;
```

-   `dimension` is the size of the square matrix (number of theta
    parameters). Declared final, it never changes for a given solver
    instance.
-   `matrix` holds the CSC structure plus numeric values.
-   `symbolic` holds the result of the expensive one-time analysis.
-   `numeric` holds the result of the per-step factorisation.
-   The two boolean flags enforce the correct call order:
    `analyzePattern` → `factorize` → `solve`.

------------------------------------------------------------------------

### `analyzePattern()`

``` java
    public void analyzePattern(int[] colPtr, int[] rowIdx) {
        validatePattern(colPtr, rowIdx);

        matrix = new Dcs();
        matrix.nzmax = rowIdx.length;
        matrix.m = dimension;
        matrix.n = dimension;
        matrix.p = colPtr.clone();
        matrix.i = rowIdx.clone();
        matrix.x = new double[rowIdx.length];
        matrix.nz = -1;
```

Populates CSparseJ's `Dcs` structure. The field names come from the
original C library (CSparse by Tim Davis):

-   `nzmax` — capacity (maximum number of non-zeros).

-   `m`, `n` — number of rows and columns.

-   `p` — `colPtr` array (column pointers).

-   `i` — `rowIdx` array (row indices).

-   `x` — numeric values (initialised to zeros; set properly in
    `factorize`).

-   `nz = -1` — signals CSC format (as opposed to triplet format where
    nz ≥ 0).

The arrays are cloned (`colPtr.clone()`) so that changes to the caller's
arrays do not affect the matrix object.

------------------------------------------------------------------------

``` java
        symbolic = Dcs_schol.cs_schol(1, matrix);

        if (symbolic == null) {
            throw new IllegalStateException(
                    "CSparseJ symbolic Cholesky analysis failed."
            );
        }
        patternAnalysed = true;
        numericFactorised = false;
    }
```

`cs_schol(1, matrix)`:

-   `1` selects the "AMD on A+A\^T" fill-reducing ordering. This
    reorders rows and columns to minimise fill-in in the Cholesky
    factor.

-   Returns `null` on failure (e.g., matrix is not positive definite in
    structure).

Setting `numericFactorised = false` resets the factorisation state so
that a fresh call to `factorize()` is required after every
`analyzePattern()` call.

------------------------------------------------------------------------

### `factorize()`

``` java
    public void factorize(double[] values) {
        if (!patternAnalysed) { throw ... }
        if (values.length != matrix.nzmax) { throw ... }

        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(values[i]) || Double.isInfinite(values[i])) {
                throw new IllegalArgumentException(
                        "Invalid numeric value at CSC position " + i + ": " + values[i]
                );
            }
        }
```

Validates that values are finite (NaN or ±∞ would produce a nonsensical
factorisation).

------------------------------------------------------------------------

``` java
        System.arraycopy(values, 0, matrix.x, 0, values.length);
```

`System.arraycopy(src, srcPos, dest, destPos, length)` is Java's fast
array copy. Here it copies all `values` into `matrix.x` (the numeric
values of the CSC matrix). This is faster than a manual loop.

------------------------------------------------------------------------

``` java
        numeric = Dcs_chol.cs_chol(matrix, symbolic);

        if (numeric == null || numeric.L == null) {
            throw new IllegalStateException(
                    "CSparseJ numeric Cholesky factorisation failed."
            );
        }
        numericFactorised = true;
    }
```

`cs_chol(matrix, symbolic)` performs the numeric Cholesky, using the
permutation and fill pattern computed in `cs_schol`. Returns `null` if
the matrix is not positive definite (e.g., due to a degenerate tree or
out-of-range variance parameter).

------------------------------------------------------------------------

### `solve()`

``` java
    public double[] solve(double[] rhs) {
        if (!patternAnalysed) { throw ... }
        if (!numericFactorised) { throw ... }
        if (rhs.length != dimension) { throw ... }

        double[] x = new double[dimension];
        double[] y = new double[dimension];
```

Allocates two temporary work arrays. `x` will hold the solution; `y` is
the intermediate permuted right-hand side.

------------------------------------------------------------------------

``` java
        /*
         * Solve A x = rhs using the CSparseJ Cholesky convention:
         *   P A P^T = L L^T
         *
         * 1. y = P rhs           (apply permutation to rhs)
         * 2. Solve L z = y       (forward substitution, in-place: y becomes z)
         * 3. Solve L^T w = z     (backward substitution, in-place: y becomes w)
         * 4. x = P^T w           (undo permutation)
         */
        Dcs_ipvec.cs_ipvec(symbolic.pinv, rhs, y, dimension);
```

`cs_ipvec(pinv, b, x, n)` computes `x[pinv[k]] = b[k]` for k = 0..n-1.
`symbolic.pinv` is the inverse-permutation vector computed during
`cs_schol`. This applies the row-permutation P to `rhs`, placing the
result in `y`.

------------------------------------------------------------------------

``` java
        Dcs_lsolve.cs_lsolve(numeric.L, y);
```

Solves L z = y in-place (overwrites `y` with z). `numeric.L` is the
lower Cholesky factor.

------------------------------------------------------------------------

``` java
        Dcs_ltsolve.cs_ltsolve(numeric.L, y);
```

Solves L\^T w = z in-place (overwrites y with w). After this line, `y`
holds the solution in the permuted space.

------------------------------------------------------------------------

``` java
        Dcs_pvec.cs_pvec(symbolic.pinv, y, x, dimension);
```

`cs_pvec(pinv, b, x, n)` computes `x[k] = b[pinv[k]]`, applying P\^T to
map from the permuted space back to the original space.

------------------------------------------------------------------------

``` java
        return x;
    }
```

Returns the solution vector.

------------------------------------------------------------------------

### `getCholeskyNonZeros()`

``` java
    public int getCholeskyNonZeros() {
        if (numeric == null || numeric.L == null || numeric.L.p == null) {
            return -1;
        }
        return numeric.L.p[numeric.L.n];
    }
```

Returns the number of non-zero entries in the Cholesky factor L.
`numeric.L.p` is the column pointer array of L in CSC format; its last
entry `p[n]` equals the total number of stored entries. This serves as a
diagnostic by showing how much fill-in the ordering produced.

------------------------------------------------------------------------

### `validatePattern()`

``` java
    private void validatePattern(int[] colPtr, int[] rowIdx) {
        if (colPtr.length != dimension + 1) { throw ... }
        if (colPtr[0] != 0) { throw ... }
        if (colPtr[dimension] != rowIdx.length) { throw ... }

        for (int j = 0; j < dimension; j++) {
            if (colPtr[j] > colPtr[j + 1]) { throw ... }
            for (int p = colPtr[j]; p < colPtr[j + 1]; p++) {
                int row = rowIdx[p];
                if (row < 0 || row >= dimension) { throw ... }
            }
        }
    }
```

Checks that the provided CSC arrays are internally consistent:

1\. `colPtr` must have exactly `dimension + 1` entries.

2\. `colPtr[0]` must be 0 (no non-zeros before column 0).

3\. `colPtr[dimension]` must equal the number of non-zeros.

4\. `colPtr` must be non-decreasing (each column has ≥ 0 entries).

5\. Every row index must be in the valid range [0, dimension).
