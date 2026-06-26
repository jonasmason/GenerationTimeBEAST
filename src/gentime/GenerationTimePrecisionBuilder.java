package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.CalculationNode;

public class GenerationTimePrecisionBuilder extends CalculationNode {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree from which sparse Brownian precision structures are built.",
            Input.Validate.REQUIRED
    );

    public Input<Double> minBranchLengthInput = new Input<>(
            "minBranchLength",
            "Lower floor applied to each branch length before forming the Brownian " +
            "precision. A zero or degenerate branch (tied node heights in a starting " +
            "tree, identical sequences, or an unresolved polytomy) then yields a large " +
            "but finite precision instead of an error. Keep it tiny so it only catches " +
            "true degeneracies and does not shrink genuine short-branch signal.",
            1.0e-8
    );
    
    protected int nodeCount;
    protected int edgeCount;

    protected int[] parent;
    protected int[] child;
    protected double[] branchLength;
    protected double minBranchLength;

    protected int nTriplets;
    protected int[] qRow;
    protected int[] qCol;
    protected double[] qValue;

    protected int nUniqueTriplets;
    protected int[] qUniqueRow;
    protected int[] qUniqueCol;
    protected double[] qUniqueValue;
    
    private long topologySignature;
    private long branchLengthSignature;

    private long structureVersion = 0L;
    private long numericVersion = 0L;
    
    private long lastStructureRefreshVersion = 0L;
    private long lastNumericRefreshVersion = 0L;

    /*
     * Type of the most recent actual global refresh performed by this precision builder.
     *
     * This is global, because other BEAST components may refresh the precision builder before the Gibbs operator sees it.
     */
    private String lastGlobalRefreshType = "initial";
    
    private long lastRefreshNanos = 0L;
    private long totalTopologyRefreshNanos = 0L;
    private long totalNumericRefreshNanos = 0L;
    private long topologyRefreshCount = 0L;
    private long numericRefreshCount = 0L;
    
    private int storedNodeCount;
    private int storedEdgeCount;

    private int[] storedParent;
    private int[] storedChild;
    private double[] storedBranchLength;

    private int storedNTriplets;
    private int[] storedQRow;
    private int[] storedQCol;
    private double[] storedQValue;

    private int storedNUniqueTriplets;
    private int[] storedQUniqueRow;
    private int[] storedQUniqueCol;
    private double[] storedQUniqueValue;

    private long storedTopologySignature;
    private long storedBranchLengthSignature;

    private long storedStructureVersion;
    private long storedNumericVersion;

    private long storedLastStructureRefreshVersion;
    private long storedLastNumericRefreshVersion;
    private String storedLastGlobalRefreshType;
    
    @Override
    public void initAndValidate() {
        minBranchLength = minBranchLengthInput.get();
        if (!(minBranchLength > 0.0) || Double.isInfinite(minBranchLength)) {
            throw new IllegalArgumentException(
                    "minBranchLength must be a positive, finite number."
            );
        }
        buildFromTree();
        topologySignature = computeTopologySignature(treeInput.get());
        branchLengthSignature = computeBranchLengthSignature(treeInput.get());
    }

    /** Lower floor applied to branch lengths before forming the Brownian precision. */
    public double getMinBranchLength() {
        return minBranchLength;
    }
    
    private void addTriplet(int index, int row, int col, double value) {
        qRow[index] = row;
        qCol[index] = col;
        qValue[index] = value;
    }

    private void buildFromTree() {

        Tree tree = treeInput.get();

        nodeCount = tree.getNodeCount();
        edgeCount = nodeCount - 1;

        parent = new int[edgeCount];
        child = new int[edgeCount];
        branchLength = new double[edgeCount];

        int edgeIndex = 0;

        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            Node parentNode = node.getParent();

            parent[edgeIndex] = parentNode.getNr();
            child[edgeIndex] = node.getNr();
            branchLength[edgeIndex] = Math.max(node.getLength(), minBranchLength);

            edgeIndex++;
        }

        if (edgeIndex != edgeCount) {
            throw new IllegalStateException(
                    "Expected " + edgeCount + " edges but found " + edgeIndex
            );
        }
        
        buildEdgePrecisionTriplets();
        combineDuplicateTriplets();
    }

    public int getNodeCount() {
        return nodeCount;
    }

    public int getEdgeCount() {
        return edgeCount;
    }

    public int getParent(int edgeIndex) {
        return parent[edgeIndex];
    }

    public int getChild(int edgeIndex) {
        return child[edgeIndex];
    }

    public double getBranchLength(int edgeIndex) {
        return branchLength[edgeIndex];
    }
        
    public long getStructureVersion() {
        return structureVersion;
    }

    public long getNumericVersion() {
        return numericVersion;
    }

    public long getLastStructureRefreshVersion() {
        return lastStructureRefreshVersion;
    }

    public long getLastNumericRefreshVersion() {
        return lastNumericRefreshVersion;
    }

    public String getLastGlobalRefreshType() {
        return lastGlobalRefreshType;
    }
    
    public long getLastRefreshNanos() {
        return lastRefreshNanos;
    }

    public long getTotalTopologyRefreshNanos() {
        return totalTopologyRefreshNanos;
    }

    public long getTotalNumericRefreshNanos() {
        return totalNumericRefreshNanos;
    }

    public long getTopologyRefreshCount() {
        return topologyRefreshCount;
    }

    public long getNumericRefreshCount() {
        return numericRefreshCount;
    }

    public String refreshFromTreeIfNeeded() {

        long refreshStart = System.nanoTime();

        Tree tree = treeInput.get();

        long newTopologySignature = computeTopologySignature(tree);
        long newBranchLengthSignature = computeBranchLengthSignature(tree);

        if (tree.getNodeCount() != nodeCount ||
                newTopologySignature != topologySignature) {

            buildFromTree();

            topologySignature = computeTopologySignature(tree);
            branchLengthSignature = computeBranchLengthSignature(tree);

            structureVersion++;
            numericVersion++;

            lastStructureRefreshVersion = structureVersion;
            lastNumericRefreshVersion = numericVersion;
            lastGlobalRefreshType = "topology";

            lastRefreshNanos = System.nanoTime() - refreshStart;
            totalTopologyRefreshNanos += lastRefreshNanos;
            topologyRefreshCount++;

            return lastGlobalRefreshType;
        }

        if (newBranchLengthSignature != branchLengthSignature) {

            updateBranchLengthsFromTree();
            buildEdgePrecisionTriplets();
            combineDuplicateTriplets();

            branchLengthSignature = newBranchLengthSignature;

            numericVersion++;

            lastNumericRefreshVersion = numericVersion;
            lastGlobalRefreshType = "numeric";

            lastRefreshNanos = System.nanoTime() - refreshStart;
            totalNumericRefreshNanos += lastRefreshNanos;
            numericRefreshCount++;

            return lastGlobalRefreshType;
        }

        return "none";
    }

    private void updateBranchLengthsFromTree() {

        Tree tree = treeInput.get();

        int edgeIndex = 0;

        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            Node parentNode = node.getParent();

            if (parent[edgeIndex] != parentNode.getNr() ||
                    child[edgeIndex] != node.getNr()) {
                throw new IllegalStateException(
                        "Topology changed during numeric-only precision update."
                );
            }

            branchLength[edgeIndex] = Math.max(node.getLength(), minBranchLength);
            edgeIndex++;
        }

        if (edgeIndex != edgeCount) {
            throw new IllegalStateException(
                    "Expected " + edgeCount + " edges but found " + edgeIndex
            );
        }
    }

    private long computeTopologySignature(Tree tree) {

        long hash = 1125899906842597L;

        hash = 31L * hash + tree.getNodeCount();
        hash = 31L * hash + tree.getRoot().getNr();

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            hash = 31L * hash + node.getNr();
            hash = 31L * hash + node.getParent().getNr();
        }

        return hash;
    }

    private long computeBranchLengthSignature(Tree tree) {

        long hash = 1469598103934665603L;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            long bits = Double.doubleToLongBits(node.getLength());
            hash ^= bits;
            hash *= 1099511628211L;
        }

        return hash;
    }
    
    private void buildEdgePrecisionTriplets() {

        nTriplets = edgeCount * 4;

        qRow = new int[nTriplets];
        qCol = new int[nTriplets];
        qValue = new double[nTriplets];

        int k = 0;

        for (int e = 0; e < edgeCount; e++) {

            int p = parent[e];
            int c = child[e];
            double length = branchLength[e];

            if (!(length > 0.0)) {
                // Branch lengths are already floored at assignment; this is a
                // defensive backstop that also catches non-finite values, so a
                // degenerate branch never aborts the run.
                length = minBranchLength;
            }

            double w = 1.0 / length;

            addTriplet(k++, c, c,  w);
            addTriplet(k++, p, p,  w);
            addTriplet(k++, c, p, -w);
            addTriplet(k++, p, c, -w);
        }

        if (k != nTriplets) {
            throw new IllegalStateException(
                    "Expected " + nTriplets + " precision triplets but wrote " + k
            );
        }
    }

    protected void printPrecisionTriplets() {

        System.out.println("Edge-based sparse Brownian precision triplets:");
        System.out.println("nTriplets = " + nTriplets);

        for (int k = 0; k < nTriplets; k++) {
            System.out.println(
                    "Q[" + qRow[k] + "," + qCol[k] + "] += " + qValue[k]
            );
        }
    }
    
    private void combineDuplicateTriplets() {

        long[] keys = new long[nTriplets];
        double[] values = new double[nTriplets];

        for (int k = 0; k < nTriplets; k++) {
            keys[k] = tripletKey(
                    qRow[k],
                    qCol[k]
            );
            values[k] = qValue[k];
        }

        sortKeyValuePairs(
                keys,
                values,
                0,
                nTriplets - 1
        );

        int uniqueCount = 0;

        int k = 0;

        while (k < nTriplets) {

            long currentKey = keys[k];
            double sum = values[k];

            k++;

            while (k < nTriplets && keys[k] == currentKey) {
                sum += values[k];
                k++;
            }

            keys[uniqueCount] = currentKey;
            values[uniqueCount] = sum;
            uniqueCount++;
        }

        nUniqueTriplets = uniqueCount;

        qUniqueRow = new int[nUniqueTriplets];
        qUniqueCol = new int[nUniqueTriplets];
        qUniqueValue = new double[nUniqueTriplets];

        for (int i = 0; i < nUniqueTriplets; i++) {
            qUniqueRow[i] = tripletRowFromKey(keys[i]);
            qUniqueCol[i] = tripletColFromKey(keys[i]);
            qUniqueValue[i] = values[i];
        }
    }

    private long tripletKey(int row, int col) {
        return (((long) row) << 32) | (col & 0xffffffffL);
    }

    private int tripletRowFromKey(long key) {
        return (int) (key >>> 32);
    }

    private int tripletColFromKey(long key) {
        return (int) key;
    }

    private void sortKeyValuePairs(
            long[] keys,
            double[] values,
            int left,
            int right
    ) {

        if (left >= right) {
            return;
        }

        int i = left;
        int j = right;
        long pivot = keys[left + (right - left) / 2];

        while (i <= j) {

            while (keys[i] < pivot) {
                i++;
            }

            while (keys[j] > pivot) {
                j--;
            }

            if (i <= j) {
                long tmpKey = keys[i];
                keys[i] = keys[j];
                keys[j] = tmpKey;

                double tmpValue = values[i];
                values[i] = values[j];
                values[j] = tmpValue;

                i++;
                j--;
            }
        }

        if (left < j) {
            sortKeyValuePairs(keys, values, left, j);
        }

        if (i < right) {
            sortKeyValuePairs(keys, values, i, right);
        }
    }

    protected void printUniquePrecisionTriplets() {

        System.out.println("Accumulated sparse Brownian precision triplets:");
        System.out.println("nUniqueTriplets = " + nUniqueTriplets);

        for (int k = 0; k < nUniqueTriplets; k++) {
            System.out.println(
                    "Q[" + qUniqueRow[k] + "," + qUniqueCol[k] + "] = " + qUniqueValue[k]
            );
        }
    }
    
    public int getTripletCount() {
        return nTriplets;
    }

    public int getQRow(int index) {
        return qRow[index];
    }

    public int getQCol(int index) {
        return qCol[index];
    }

    public double getQValue(int index) {
        return qValue[index];
    }
    
    public int getUniqueTripletCount() {
        return nUniqueTriplets;
    }

    public int getQUniqueRow(int index) {
        return qUniqueRow[index];
    }

    public int getQUniqueCol(int index) {
        return qUniqueCol[index];
    }

    public double getQUniqueValue(int index) {
        return qUniqueValue[index];
    }

    @Override
    protected void store() {

        storedNodeCount = nodeCount;
        storedEdgeCount = edgeCount;

        storedParent = parent == null ? null : parent.clone();
        storedChild = child == null ? null : child.clone();
        storedBranchLength = branchLength == null ? null : branchLength.clone();

        storedNTriplets = nTriplets;
        storedQRow = qRow == null ? null : qRow.clone();
        storedQCol = qCol == null ? null : qCol.clone();
        storedQValue = qValue == null ? null : qValue.clone();

        storedNUniqueTriplets = nUniqueTriplets;
        storedQUniqueRow = qUniqueRow == null ? null : qUniqueRow.clone();
        storedQUniqueCol = qUniqueCol == null ? null : qUniqueCol.clone();
        storedQUniqueValue = qUniqueValue == null ? null : qUniqueValue.clone();

        storedTopologySignature = topologySignature;
        storedBranchLengthSignature = branchLengthSignature;

        storedStructureVersion = structureVersion;
        storedNumericVersion = numericVersion;

        storedLastStructureRefreshVersion = lastStructureRefreshVersion;
        storedLastNumericRefreshVersion = lastNumericRefreshVersion;
        storedLastGlobalRefreshType = lastGlobalRefreshType;

        super.store();
    }

    @Override
    protected void restore() {

        nodeCount = storedNodeCount;
        edgeCount = storedEdgeCount;

        parent = storedParent;
        child = storedChild;
        branchLength = storedBranchLength;

        nTriplets = storedNTriplets;
        qRow = storedQRow;
        qCol = storedQCol;
        qValue = storedQValue;

        nUniqueTriplets = storedNUniqueTriplets;
        qUniqueRow = storedQUniqueRow;
        qUniqueCol = storedQUniqueCol;
        qUniqueValue = storedQUniqueValue;

        topologySignature = storedTopologySignature;
        branchLengthSignature = storedBranchLengthSignature;

        structureVersion = storedStructureVersion;
        numericVersion = storedNumericVersion;

        lastStructureRefreshVersion = storedLastStructureRefreshVersion;
        lastNumericRefreshVersion = storedLastNumericRefreshVersion;
        lastGlobalRefreshType = storedLastGlobalRefreshType;

        super.restore();
    }

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }
}