package gentime;

import edu.emory.mathcs.csparsej.tdouble.Dcs_chol;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcsn;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;
import edu.emory.mathcs.csparsej.tdouble.Dcs_ipvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_lsolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_ltsolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_pvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_schol;

public class CSparseJSolver {

    /*
     * Wrapper around CSparseJ sparse Cholesky routines.
     *
     * This is the production direct-solver path for the Gaussian Gibbs update.
     * The sparse matrix pattern is analysed once per topology, while numeric
     * factorisation is repeated whenever matrix values change.
     */
	
    private final int dimension;

    private Dcs matrix;
    private Dcss symbolic;
    private Dcsn numeric;

    private boolean patternAnalysed = false;
    private boolean numericFactorised = false;

    public CSparseJSolver(int dimension) {
        this.dimension = dimension;
    }
    
    // Store the CSC pattern and run CSparseJ symbolic Cholesky analysis.

    public void analyzePattern(
            int[] colPtr,
            int[] rowIdx
    ) {

        validatePattern(colPtr, rowIdx);

        matrix = new Dcs();
        matrix.nzmax = rowIdx.length;
        matrix.m = dimension;
        matrix.n = dimension;
        matrix.p = colPtr.clone();
        matrix.i = rowIdx.clone();
        matrix.x = new double[rowIdx.length];
        matrix.nz = -1;

        /*
         * order = 1 asks CSparseJ to use its Cholesky symbolic analysis
         * with fill-reducing ordering, following the CSparse cs_schol path.
         */
        symbolic = Dcs_schol.cs_schol(1, matrix);

        if (symbolic == null) {
            throw new IllegalStateException(
                    "CSparseJ symbolic Cholesky analysis failed."
            );
        }

        patternAnalysed = true;
        numericFactorised = false;
    }

    // Update CSC numeric values and run CSparseJ numeric Cholesky factorisation.
    
    public void factorize(double[] values) {

        if (!patternAnalysed) {
            throw new IllegalStateException(
                    "analyzePattern must be called before factorize."
            );
        }

        if (values.length != matrix.nzmax) {
            throw new IllegalArgumentException(
                    "values length must match CSC non-zero count."
            );
        }

        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(values[i]) || Double.isInfinite(values[i])) {
                throw new IllegalArgumentException(
                        "Invalid numeric value at CSC position " + i +
                                ": " + values[i]
                );
            }
        }

        System.arraycopy(values, 0, matrix.x, 0, values.length);

        numeric = Dcs_chol.cs_chol(matrix, symbolic);

        if (numeric == null || numeric.L == null) {
            throw new IllegalStateException(
                    "CSparseJ numeric Cholesky factorisation failed."
            );
        }

        numericFactorised = true;
    }
    
    // Solve A x = rhs using the current CSparseJ Cholesky factorisation.

    public double[] solve(double[] rhs) {

        if (!patternAnalysed) {
            throw new IllegalStateException(
                    "analyzePattern must be called before solve."
            );
        }

        if (!numericFactorised) {
            throw new IllegalStateException(
                    "factorize must be called before solve."
            );
        }

        if (rhs.length != dimension) {
            throw new IllegalArgumentException(
                    "rhs length must match matrix dimension."
            );
        }

        double[] x = new double[dimension];
        double[] y = new double[dimension];

        /*
         * Solve A x = rhs using the CSparseJ Cholesky convention:
         *
         *   P A P' = L L'
         *
         * Therefore:
         *   y = P rhs
         *   solve L z = y
         *   solve L' w = z
         *   x = P' w
         */
        Dcs_ipvec.cs_ipvec(symbolic.pinv, rhs, y, dimension);
        Dcs_lsolve.cs_lsolve(numeric.L, y);
        Dcs_ltsolve.cs_ltsolve(numeric.L, y);
        Dcs_pvec.cs_pvec(symbolic.pinv, y, x, dimension);

        return x;
    }

    public int getCholeskyNonZeros() {
        if (numeric == null || numeric.L == null || numeric.L.p == null) {
            return -1;
        }

        return numeric.L.p[numeric.L.n];
    }

    private void validatePattern(
            int[] colPtr,
            int[] rowIdx
    ) {

        if (colPtr.length != dimension + 1) {
            throw new IllegalArgumentException(
                    "colPtr length must be dimension + 1."
            );
        }

        if (colPtr[0] != 0) {
            throw new IllegalArgumentException(
                    "colPtr[0] must be zero."
            );
        }

        if (colPtr[dimension] != rowIdx.length) {
            throw new IllegalArgumentException(
                    "Final colPtr entry must equal rowIdx length."
            );
        }

        for (int j = 0; j < dimension; j++) {

            if (colPtr[j] > colPtr[j + 1]) {
                throw new IllegalArgumentException(
                        "colPtr must be non-decreasing."
                );
            }

            for (int p = colPtr[j]; p < colPtr[j + 1]; p++) {

                int row = rowIdx[p];

                if (row < 0 || row >= dimension) {
                    throw new IllegalArgumentException(
                            "rowIdx out of range at position " + p +
                                    ": " + row
                    );
                }
            }
        }
    }
}