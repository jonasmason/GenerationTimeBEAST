package gentime;

import java.util.HashMap;
import java.util.Map;

public class InternalSparseCholeskySolver {
	
    /*
     * Internal sparse Cholesky reference solver.
     *
     * This class is not CSparseJ and is not the main solver but rather a transparent Java implementation used by GenerationTimeGibbsOperatorTest to validate the CSparseJ solve path.
     */

    private final int dimension;

    private int[] colPtr;
    private int[] rowIdx;
    private int nonZeros;

    private boolean patternAnalysed = false;
    private boolean numericFactorised = false;
    
    private long lastClearNanos = 0L;
    private long lastPermuteNanos = 0L;
    private long lastNumericNanos = 0L;
    private long lastRowAccessNanos = 0L;

    /*
     * Store and validate the CSC sparsity pattern, then build the fixed beta-last permutation used by this internal reference factorisation.
     */
    
    private Map<Integer, Double>[] lowerColumns;
    private Map<Integer, Double>[] lowerRows;
    
    private boolean choleskyStorageAllocated = false;
    
    private int[][] previousColumnsByRow;
    
    private int[] oldToNew;
    private int[] newToOld;

    public InternalSparseCholeskySolver(int dimension) {
        this.dimension = dimension;
    }

    /**
     * Symbolic analysis step.
     *
     * Current implementation only stores the CSC pattern.
     * Later this will be extended to AMD ordering and symbolic Cholesky.
     */
    public void analyzePattern(
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

        this.colPtr = colPtr.clone();
        this.rowIdx = rowIdx.clone();
        this.nonZeros = rowIdx.length;

        buildBetaLastPermutation();

        this.patternAnalysed = true;
        this.numericFactorised = false;
    }

    /*
     * Numeric sparse Cholesky factorisation under the stored beta-last ordering.
     */
    @SuppressWarnings("unchecked")
    public void factorize(
            double[] values
    ) {

        if (!patternAnalysed) {
            throw new IllegalStateException(
                    "analyzePattern must be called before factorize."
            );
        }

        if (values.length != nonZeros) {
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

        long clearStart = System.nanoTime();

        allocateOrClearCholeskyStorage();

        lastClearNanos = System.nanoTime() - clearStart;

        long permuteStart = System.nanoTime();

        Map<Integer, Double>[] permutedColumns = new HashMap[dimension];

        for (int j = 0; j < dimension; j++) {
            permutedColumns[j] = new HashMap<>();
        }

        for (int oldColumn = 0; oldColumn < dimension; oldColumn++) {

            int newColumn = oldToNew[oldColumn];

            for (int p = colPtr[oldColumn]; p < colPtr[oldColumn + 1]; p++) {

                int oldRow = rowIdx[p];
                int newRow = oldToNew[oldRow];

                if (newRow >= newColumn) {
                    addToMap(
                            permutedColumns[newColumn],
                            newRow,
                            values[p]
                    );
                }
            }
        }

        lastPermuteNanos = System.nanoTime() - permuteStart;

        long numericStart = System.nanoTime();
        
        /*
         * Left-looking sparse Cholesky:
         *
         * For each column j:
         *   x = lower part of A[:, j]
         *   subtract previous Cholesky columns k where L[j, k] != 0
         *   normalise by sqrt(diagonal)
         */
        
        for (int j = 0; j < dimension; j++) {

        	Map<Integer, Double> workColumn = new HashMap<>(
        	        permutedColumns[j]
        	);

            /*
             * Apply previous columns.
             *
             * If L[j, k] is non-zero, then:
             *   x[j:n] -= L[j,k] * L[j:n,k]
             */
            
        	if (previousColumnsByRow == null) {

        	    for (int k = 0; k < j; k++) {

        	        Double ljkObject = lowerColumns[k].get(j);

        	        if (ljkObject == null) {
        	            continue;
        	        }

        	        double ljk = ljkObject;

        	        for (Map.Entry<Integer, Double> entry : lowerColumns[k].entrySet()) {

        	            int row = entry.getKey();

        	            if (row < j) {
        	                continue;
        	            }

        	            double lik = entry.getValue();

        	            addToMap(
        	                    workColumn,
        	                    row,
        	                    -lik * ljk
        	            );
        	        }
        	    }

        	} else {

        	    int[] previousColumns = previousColumnsByRow[j];

        	    for (int index = 0; index < previousColumns.length; index++) {

        	        int k = previousColumns[index];

        	        Double ljkObject = lowerColumns[k].get(j);

        	        if (ljkObject == null) {
        	            throw new IllegalStateException(
        	                    "Stored dependency missing L[" + j + "," + k + "]"
        	            );
        	        }

        	        double ljk = ljkObject;

        	        for (Map.Entry<Integer, Double> entry : lowerColumns[k].entrySet()) {

        	            int row = entry.getKey();

        	            if (row < j) {
        	                continue;
        	            }

        	            double lik = entry.getValue();

        	            addToMap(
        	                    workColumn,
        	                    row,
        	                    -lik * ljk
        	            );
        	        }
        	    }
        	}

            Double diagonalObject = workColumn.get(j);

            if (diagonalObject == null) {
                throw new IllegalArgumentException(
                        "Missing diagonal entry during sparse Cholesky at column " + j
                );
            }

            double diagonal = diagonalObject;

            if (diagonal <= 0.0 ||
                    Double.isNaN(diagonal) ||
                    Double.isInfinite(diagonal)) {
                throw new IllegalArgumentException(
                        "Sparse Cholesky failed at diagonal " + j +
                        " with value " + diagonal
                );
            }

            double ljj = Math.sqrt(diagonal);

            lowerColumns[j].put(j, ljj);

            for (Map.Entry<Integer, Double> entry : workColumn.entrySet()) {

                int row = entry.getKey();

                if (row <= j) {
                    continue;
                }

                double value = entry.getValue() / ljj;

                if (value != 0.0 &&
                        !Double.isNaN(value) &&
                        !Double.isInfinite(value)) {
                    lowerColumns[j].put(row, value);
                }
            }
        }

        lastNumericNanos = System.nanoTime() - numericStart;

        long rowAccessStart = System.nanoTime();

        buildRowAccess();

        if (previousColumnsByRow == null) {
            buildPreviousColumnsByRow();
        }

        lastRowAccessNanos = System.nanoTime() - rowAccessStart;

        this.numericFactorised = true;
    }

    /*
     * Solve A x = rhs using the current sparse Cholesky factor.
     */
    
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

    double[] permutedRhs = new double[dimension];

    for (int newIndex = 0; newIndex < dimension; newIndex++) {
        int oldIndex = newToOld[newIndex];
        permutedRhs[newIndex] = rhs[oldIndex];
    }

    double[] y = new double[dimension];
    double[] permutedX = new double[dimension];

    for (int i = 0; i < dimension; i++) {

        double sum = permutedRhs[i];

        for (Map.Entry<Integer, Double> entry : lowerRows[i].entrySet()) {

            int k = entry.getKey();

            if (k >= i) {
                continue;
            }

            sum -= entry.getValue() * y[k];
        }

        Double diagonal = lowerRows[i].get(i);

        if (diagonal == null || diagonal == 0.0) {
            throw new IllegalStateException(
                    "Missing diagonal in forward solve at row " + i
            );
        }

        y[i] = sum / diagonal;
    }

    for (int i = dimension - 1; i >= 0; i--) {

        double sum = y[i];

        for (Map.Entry<Integer, Double> entry : lowerColumns[i].entrySet()) {

            int k = entry.getKey();

            if (k <= i) {
                continue;
            }

            sum -= entry.getValue() * permutedX[k];
        }

        Double diagonal = lowerColumns[i].get(i);

        if (diagonal == null || diagonal == 0.0) {
            throw new IllegalStateException(
                    "Missing diagonal in backward solve at column " + i
            );
        }

        permutedX[i] = sum / diagonal;
    }

    double[] x = new double[dimension];

    for (int newIndex = 0; newIndex < dimension; newIndex++) {
        int oldIndex = newToOld[newIndex];
        x[oldIndex] = permutedX[newIndex];
    }

    return x;
}

    @SuppressWarnings("unchecked")
	private void allocateOrClearCholeskyStorage() {
	
	    if (!choleskyStorageAllocated) {
	
	        lowerColumns = new HashMap[dimension];
	        lowerRows = new HashMap[dimension];
	
	        for (int i = 0; i < dimension; i++) {
	            lowerColumns[i] = new HashMap<>();
	            lowerRows[i] = new HashMap<>();
	        }
	
	        choleskyStorageAllocated = true;
	        return;
	    }
	
	    for (int i = 0; i < dimension; i++) {
	        lowerColumns[i].clear();
	        lowerRows[i].clear();
	    }
	}

	private void buildRowAccess() {
	
	    for (int column = 0; column < dimension; column++) {
	
	        for (Map.Entry<Integer, Double> entry : lowerColumns[column].entrySet()) {
	
	            int row = entry.getKey();
	            double value = entry.getValue();
	
	            lowerRows[row].put(column, value);
	        }
	    }
	}

	private void buildPreviousColumnsByRow() {
	
	    previousColumnsByRow = new int[dimension][];
	
	    for (int row = 0; row < dimension; row++) {
	
	        int count = 0;
	
	        for (int column = 0; column < row; column++) {
	            if (lowerColumns[column].containsKey(row)) {
	                count++;
	            }
	        }
	
	        int[] columns = new int[count];
	        int index = 0;
	
	        for (int column = 0; column < row; column++) {
	            if (lowerColumns[column].containsKey(row)) {
	                columns[index] = column;
	                index++;
	            }
	        }
	
	        previousColumnsByRow[row] = columns;
	    }
	}
	
    private void addToMap(
            Map<Integer, Double> map,
            int index,
            double value
    ) {

        Double oldValue = map.get(index);

        if (oldValue == null) {
            map.put(index, value);
        } else {
            map.put(index, oldValue + value);
        }
    }

    /*
     * Move beta to the final position before factorisation.
     *
     * The posterior precision has an arrowhead-like beta/tip structure. Placing beta last reduces unnecessary fill-in.
     */
    
    private void buildBetaLastPermutation() {

        oldToNew = new int[dimension];
        newToOld = new int[dimension];

        /*
         * Original ordering:
         *   0 = beta
         *   1..dimension-1 = latent effects
         *
         * New ordering:
         *   0..dimension-2 = latent effects
         *   dimension-1 = beta
         */
        for (int oldIndex = 1; oldIndex < dimension; oldIndex++) {
            int newIndex = oldIndex - 1;
            oldToNew[oldIndex] = newIndex;
            newToOld[newIndex] = oldIndex;
        }

        oldToNew[0] = dimension - 1;
        newToOld[dimension - 1] = 0;
    }
    
    public int getDimension() {
        return dimension;
    }

    public int getNonZeros() {
        return nonZeros;
    }

    public double getLastClearMillis() {
        return lastClearNanos / 1.0e6;
    }

    public double getLastPermuteMillis() {
        return lastPermuteNanos / 1.0e6;
    }

    public double getLastNumericMillis() {
        return lastNumericNanos / 1.0e6;
    }

    public double getLastRowAccessMillis() {
        return lastRowAccessNanos / 1.0e6;
    }
    
    public int getCholeskyNonZeros() {

        if (lowerColumns == null) {
            return 0;
        }

        int total = 0;

        for (int j = 0; j < dimension; j++) {
            total += lowerColumns[j].size();
        }

        return total;
    }
}