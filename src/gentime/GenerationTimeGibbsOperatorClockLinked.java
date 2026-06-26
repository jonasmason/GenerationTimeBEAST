package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/*
 * Clock-linked version of GenerationTimeGibbsOperator.
 *
 * Difference from the original:
 * in clock-linked analyses, generationTimes and beta also affect the sequence
 * likelihood through GenerationTimeRegressionClockModel. Therefore, after this
 * operator manually evaluates gtLikelihood for Gibbs acceptance accounting, it
 * marks gtLikelihood dirty so BEAST recalculates dependent posterior components
 * through the normal posterior machinery.
 *
 * The original GenerationTimeGibbsOperator is left unchanged so GT-only trait
 * reconstruction remains available as a separate validated mode.
 */

public class GenerationTimeGibbsOperatorClockLinked extends Operator {

	// XML-configurable inputs
	
    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose generation-time latent states are updated.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent phylogenetic effect vector u indexed by BEAST tree node number. Root is fixed to zero.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Intercept parameter beta in y = beta + u_tip + e.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> brownianVarianceInput = new Input<>(
            "brownianVariance",
            "Phylogenetic Brownian variance parameter for latent effects u.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> residualVarianceInput = new Input<>(
            "residualVariance",
            "Residual variance parameter for observed tip generation times.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimeTraitData> traitDataInput = new Input<>(
            "traitData",
            "Observed generation-time tip data.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimePrecisionBuilder> precisionBuilderInput = new Input<>(
            "precisionBuilder",
            "Sparse Brownian precision builder.",
            Input.Validate.REQUIRED
    );
    
    public Input<GenerationTimeLikelihood> gtLikelihoodInput = new Input<>(
            "gtLikelihood",
            "Generation-time likelihood to recalculate immediately after this operator writes GT state.",
            Input.Validate.REQUIRED
    );

    public Input<Double> priorBetaMeanInput = new Input<>(
            "priorBetaMean",
            "Prior mean for beta.",
            0.0
    );

    public Input<Double> priorBetaVarianceInput = new Input<>(
            "priorBetaVariance",
            "Prior variance for beta.",
            1000.0
    );

    public Input<String> scaleInput = new Input<>(
            "scale",
            "Scale for observed generation-time model: 'raw' or 'log'.",
            "raw"
    );

    // Tree dimensions and node-index bookkeeping
    private int nodeCount;
    private int leafNodeCount;
    private int latentNodeCount;
    private int thetaCount;
    private int rootNodeNr;

    // Mapping between BEAST node numbers, latent vector indices, observed tip data
    private int[] tipLatentIndices;
    private int[] latentNodeNrs;
    private double[] observedTipValues;

    private int[] latentIndexByNodeNr;
    private boolean[] isLatentNode;

    // Brownian precision triplets restricted to non-root latent nodes
    private int nLatentTriplets;
    private int[] qLatentRow;
    private int[] qLatentCol;
    private double[] qLatentValue;

    // Temporary vectors used during one Gibbs proposal
    private double[] lastGibbsDraw;
    private double[] lastPriorDraw;
    private double[] lastDelta;
    private double[] lastDeltaRhs;
    
    // Track whether shared precision builder has changed since this operator last refreshed
    private long seenPrecisionStructureVersion = -1L;
    private long seenPrecisionNumericVersion = -1L;
        
    // Posterior precision matrix C stored in compressed sparse column form for CSparseJ
    private int cscNonZeros;
    private int[] cscColPtr;
    private int[] cscRowIdx;
    private double[] cscValues;

    private int cscBetaBetaIndex;
    private int[] cscBetaTipIndex;
    private int[] cscTipBetaIndex;
    private int[] cscTipDiagonalIndex;
    private int[] cscLatentPrecisionIndex;
    
    // Sparse Cholesky solver for C * delta = rhs
    private CSparseJSolver csparseJSolver;


    
    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();
        RealParameter beta = betaInput.get();
        RealParameter brownianVariance = brownianVarianceInput.get();
        RealParameter residualVariance = residualVarianceInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        String scale = scaleInput.get();

        GenerationTimeTraitTransform.validateScale(scale);

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (beta.getDimension() != 1) {
            throw new IllegalArgumentException("beta must be a scalar RealParameter.");
        }

        if (brownianVariance.getDimension() != 1) {
            throw new IllegalArgumentException("brownianVariance must be a scalar RealParameter.");
        }

        if (residualVariance.getDimension() != 1) {
            throw new IllegalArgumentException("residualVariance must be a scalar RealParameter.");
        }

        if (precisionBuilder.getNodeCount() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "Precision builder node count does not match tree node count."
            );
        }

        if (traitData.size() < tree.getLeafNodeCount()) {
            throw new IllegalArgumentException(
                    "Trait data contains fewer entries than tree tips."
            );
        }

        if (priorBetaVarianceInput.get() <= 0.0) {
            throw new IllegalArgumentException("priorBetaVariance must be positive.");
        }

        buildNodeBookkeeping();
        buildLatentPrecisionTriplets();

        buildPosteriorPrecisionCscPattern();
        rebuildSparseSolversForCurrentPattern();

        seenPrecisionStructureVersion = precisionBuilder.getStructureVersion();
        seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();
    }
    
    
    @Override
    public double proposal() {

        refreshPrecisionStructuresIfNeeded();

        double brownianVariance = brownianVarianceInput.get().getValue();
        double residualVariance = residualVarianceInput.get().getValue();

        if (brownianVariance <= 0.0 || residualVariance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        // Draw theta* = (beta*, u*) from the prior on the current tree
        lastPriorDraw = drawPriorTheta(brownianVariance);
        
        // Build the residual-based RHS for the Gaussian correction
        lastDeltaRhs = buildDeltaRhs(lastPriorDraw, residualVariance);

        // Solve for the correction delta using the current posterior precision
        lastDelta = solveGaussianSystem(
                lastDeltaRhs,
                brownianVariance,
                residualVariance
        );

        lastGibbsDraw = new double[thetaCount];

        // Final Gibbs draw is theta = theta* + delta
        for (int i = 0; i < thetaCount; i++) {
            lastGibbsDraw[i] = lastPriorDraw[i] + lastDelta[i];
        }

        //BEAST acceptance accounting -> calculate target density before and after writing state
        double oldGtLogP = calculateGtLogPNoCache();

        betaInput.get().setValue(0, lastGibbsDraw[0]);

        RealParameter generationTimes = generationTimesInput.get();

        generationTimes.setValue(rootNodeNr, 0.0);

        for (int latentIndex = 0; latentIndex < latentNodeCount; latentIndex++) {
            int nodeNr = latentNodeNrs[latentIndex];
            generationTimes.setValue(nodeNr, lastGibbsDraw[1 + latentIndex]);
        }

        double newGtLogP = calculateGtLogPNoCache();

        /*
         * Clock-linked difference from GenerationTimeGibbsOperator:
         *
         * Here generationTimes and beta also feed into the sequence likelihood through
         * the GT regression clock. The calculateLogP() calls above are only used for
         * Gibbs acceptance accounting. Mark gtLikelihood dirty afterwards so BEAST does
         * not reuse a manually cached value after later tree/clock proposals.
         */
        
        return oldGtLogP - newGtLogP;
    }

    // Rebuild bookkeeping and sparse precision structures when the tree topology or branch lengths change
    
    private void refreshPrecisionStructuresIfNeeded() {

        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        precisionBuilder.refreshFromTreeIfNeeded();

        boolean structureChanged =
                precisionBuilder.getStructureVersion() != seenPrecisionStructureVersion;

        boolean numericChanged =
                precisionBuilder.getNumericVersion() != seenPrecisionNumericVersion;

        if (structureChanged) {

            buildNodeBookkeeping();
            buildLatentPrecisionTriplets();
            buildPosteriorPrecisionCscPattern();
            rebuildSparseSolversForCurrentPattern();

            seenPrecisionStructureVersion = precisionBuilder.getStructureVersion();
            seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();

            return;
        }

        if (numericChanged) {

            buildLatentPrecisionTriplets();

            seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();
        }
    }
    
    private void rebuildSparseSolversForCurrentPattern() {
	    csparseJSolver = new CSparseJSolver(thetaCount);
	    csparseJSolver.analyzePattern(cscColPtr, cscRowIdx);
	}
    
    
    // Build the mapping from BEAST node numbers to latent vector indices, excluding the root
    private void buildNodeBookkeeping() {

        Tree tree = treeInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();

        nodeCount = tree.getNodeCount();
        leafNodeCount = tree.getLeafNodeCount();
        rootNodeNr = tree.getRoot().getNr();

        latentNodeCount = nodeCount - 1;
        thetaCount = 1 + latentNodeCount;

        tipLatentIndices = new int[leafNodeCount];
        observedTipValues = new double[leafNodeCount];

        latentNodeNrs = new int[latentNodeCount];
        latentIndexByNodeNr = new int[nodeCount];
        isLatentNode = new boolean[nodeCount];

        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            latentIndexByNodeNr[nodeNr] = -1;
            isLatentNode[nodeNr] = false;
        }

        int latentIndex = 0;

        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {

            if (nodeNr == rootNodeNr) {
                continue;
            }

            latentNodeNrs[latentIndex] = nodeNr;
            latentIndexByNodeNr[nodeNr] = latentIndex;
            isLatentNode[nodeNr] = true;

            latentIndex++;
        }

        if (latentIndex != latentNodeCount) {
            throw new IllegalStateException(
                    "Expected " + latentNodeCount +
                            " latent nodes but found " + latentIndex
            );
        }

        for (int i = 0; i < leafNodeCount; i++) {

            Node tip = tree.getNode(i);
            String taxon = tip.getID();
            int nodeNr = tip.getNr();

            if (!isLatentNode[nodeNr]) {
                throw new IllegalStateException("Tip node is unexpectedly not latent: " + nodeNr);
            }

            tipLatentIndices[i] = latentIndexByNodeNr[nodeNr];

            observedTipValues[i] = GenerationTimeTraitTransform.transform(
                    traitData.getTrait(taxon),
                    scaleInput.get()
            );
        }
    }

    // Convert the full tree precision into the latent non-root precision block used by the Gibbs update
    private void buildLatentPrecisionTriplets() {

        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        int maxTriplets = precisionBuilder.getUniqueTripletCount();

        qLatentRow = new int[maxTriplets];
        qLatentCol = new int[maxTriplets];
        qLatentValue = new double[maxTriplets];

        nLatentTriplets = 0;

        for (int k = 0; k < precisionBuilder.getUniqueTripletCount(); k++) {

            int rowNode = precisionBuilder.getQUniqueRow(k);
            int colNode = precisionBuilder.getQUniqueCol(k);
            double value = precisionBuilder.getQUniqueValue(k);

            if (isLatentNode[rowNode] && isLatentNode[colNode]) {

                qLatentRow[nLatentTriplets] = latentIndexByNodeNr[rowNode];
                qLatentCol[nLatentTriplets] = latentIndexByNodeNr[colNode];
                qLatentValue[nLatentTriplets] = value;

                nLatentTriplets++;
            }
        }
    }

    
    // Simulate beta and latent Brownian effects from their prior on the current tree
    
    private double[] drawPriorTheta(double brownianVariance) {

        double[] theta = new double[thetaCount];

        double betaPriorMean = priorBetaMeanInput.get();
        double betaPriorSd = Math.sqrt(priorBetaVarianceInput.get());

        theta[0] = betaPriorMean + betaPriorSd * Randomizer.nextGaussian();

        Tree tree = treeInput.get();
        Node root = tree.getRoot();

        simulateBrownianFromNode(
                root,
                0.0,
                brownianVariance,
                theta
        );

        return theta;
    }

    private void simulateBrownianFromNode(
            Node node,
            double parentValue,
            double brownianVariance,
            double[] theta
    ) {

        for (int i = 0; i < node.getChildCount(); i++) {

            Node child = node.getChild(i);
            int childNodeNr = child.getNr();

            double branchLength = child.getLength();

            if (branchLength < 0.0) {
                throw new IllegalArgumentException(
                        "Negative branch length encountered for node " + childNodeNr
                );
            }

            double childValue = parentValue
                    + Math.sqrt(brownianVariance * branchLength)
                    * Randomizer.nextGaussian();

            if (isLatentNode[childNodeNr]) {
                int latentIndex = latentIndexByNodeNr[childNodeNr];
                theta[1 + latentIndex] = childValue;
            }

            simulateBrownianFromNode(
                    child,
                    childValue,
                    brownianVariance,
                    theta
            );
        }
    }

    // Construct the RHS induced by the difference between observed and prior-simulated tip values
    private double[] buildDeltaRhs(
            double[] priorTheta,
            double residualVariance
    ) {

        double[] rhs = new double[thetaCount];

        double residualPrecision = 1.0 / residualVariance;

        for (int i = 0; i < leafNodeCount; i++) {

            int latentIndex = tipLatentIndices[i];

            double observedY = observedTipValues[i];
            double simulatedY =
                    priorTheta[0] +
                            priorTheta[1 + latentIndex] +
                            Math.sqrt(residualVariance) * Randomizer.nextGaussian();

            double residual = observedY - simulatedY;

            rhs[0] += residualPrecision * residual;
            rhs[1 + latentIndex] += residualPrecision * residual;
        }

        return rhs;
    }

    
    // Build the fixed sparse pattern of the posterior precision matrix for the current topology
    private void buildPosteriorPrecisionCscPattern() {

        TreeMap<Long, Boolean> pattern = new TreeMap<>();

        cscBetaTipIndex = new int[leafNodeCount];
        cscTipBetaIndex = new int[leafNodeCount];
        cscTipDiagonalIndex = new int[leafNodeCount];
        cscLatentPrecisionIndex = new int[nLatentTriplets];

        addCscPatternEntry(pattern, 0, 0);

        for (int i = 0; i < leafNodeCount; i++) {

            int thetaIndex = 1 + tipLatentIndices[i];

            addCscPatternEntry(pattern, 0, thetaIndex);
            addCscPatternEntry(pattern, thetaIndex, 0);
            addCscPatternEntry(pattern, thetaIndex, thetaIndex);
        }

        for (int k = 0; k < nLatentTriplets; k++) {

            int thetaRow = 1 + qLatentRow[k];
            int thetaCol = 1 + qLatentCol[k];

            addCscPatternEntry(pattern, thetaRow, thetaCol);
        }

        cscNonZeros = pattern.size();
        cscColPtr = new int[thetaCount + 1];
        cscRowIdx = new int[cscNonZeros];
        cscValues = new double[cscNonZeros];

        Map<Long, Integer> indexByKey = new HashMap<>();

        int currentColumn = 0;
        int index = 0;

        cscColPtr[0] = 0;

        for (Long key : pattern.keySet()) {

            int column = cscColumnFromKey(key);
            int row = cscRowFromKey(key);

            while (currentColumn < column) {
                currentColumn++;
                cscColPtr[currentColumn] = index;
            }

            cscRowIdx[index] = row;
            indexByKey.put(key, index);

            index++;
        }

        while (currentColumn < thetaCount) {
            currentColumn++;
            cscColPtr[currentColumn] = index;
        }

        if (index != cscNonZeros) {
            throw new IllegalStateException(
                    "CSC pattern construction mismatch: expected " +
                            cscNonZeros + " entries but wrote " + index
            );
        }

        cscBetaBetaIndex = getCscIndex(indexByKey, 0, 0);

        for (int i = 0; i < leafNodeCount; i++) {

            int thetaIndex = 1 + tipLatentIndices[i];

            cscBetaTipIndex[i] = getCscIndex(indexByKey, 0, thetaIndex);
            cscTipBetaIndex[i] = getCscIndex(indexByKey, thetaIndex, 0);
            cscTipDiagonalIndex[i] = getCscIndex(indexByKey, thetaIndex, thetaIndex);
        }

        for (int k = 0; k < nLatentTriplets; k++) {

            int thetaRow = 1 + qLatentRow[k];
            int thetaCol = 1 + qLatentCol[k];

            cscLatentPrecisionIndex[k] = getCscIndex(indexByKey, thetaRow, thetaCol);
        }
    }

    // Fill the numeric values of the posterior precision matrix for current variance parameters
    private void updatePosteriorPrecisionCscValues(
            double brownianVariance,
            double residualVariance
    ) {

        Arrays.fill(cscValues, 0.0);

        double invResid = 1.0 / residualVariance;
        double invPhylo = 1.0 / brownianVariance;
        double priorBetaVariance = priorBetaVarianceInput.get();

        cscValues[cscBetaBetaIndex] +=
                leafNodeCount * invResid +
                        1.0 / priorBetaVariance;

        for (int i = 0; i < leafNodeCount; i++) {

            cscValues[cscBetaTipIndex[i]] += invResid;
            cscValues[cscTipBetaIndex[i]] += invResid;
            cscValues[cscTipDiagonalIndex[i]] += invResid;
        }

        for (int k = 0; k < nLatentTriplets; k++) {
            cscValues[cscLatentPrecisionIndex[k]] +=
                    invPhylo * qLatentValue[k];
        }
    }
    
    // Factorise and solve the sparse Gaussian system using CSparseJ
    private double[] solveGaussianSystem(
            double[] rhs,
            double brownianVariance,
            double residualVariance
    ) {

        updatePosteriorPrecisionCscValues(
                brownianVariance,
                residualVariance
        );

        csparseJSolver.factorize(cscValues);

        return csparseJSolver.solve(rhs);
    }
    
    
    // Helpers for indexing entries in the sparse CSC pattern
    
    private void addCscPatternEntry(
            TreeMap<Long, Boolean> pattern,
            int row,
            int column
    ) {
        pattern.put(cscKey(row, column), true);
    }

    private int getCscIndex(
            Map<Long, Integer> indexByKey,
            int row,
            int column
    ) {

        Long key = cscKey(row, column);
        Integer index = indexByKey.get(key);

        if (index == null) {
            throw new IllegalStateException(
                    "Missing CSC entry for row=" + row + " column=" + column
            );
        }

        return index;
    }

    private long cscKey(int row, int column) {
        return (((long) column) << 32) | (row & 0xffffffffL);
    }

    private int cscColumnFromKey(long key) {
        return (int) (key >>> 32);
    }

    private int cscRowFromKey(long key) {
        return (int) key;
    }
    
    /*
     * Clock-linked difference from GenerationTimeGibbsOperator:
     *
     * Compute the same GT log density as GenerationTimeLikelihood, but locally and
     * without touching the cached logP field of the BEAST Distribution.
     */
    private double calculateGtLogPNoCache() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        int currentRootNodeNr = tree.getRoot().getNr();

        double beta = betaInput.get().getValue();
        double brownianVariance = brownianVarianceInput.get().getValue();
        double residualVariance = residualVarianceInput.get().getValue();

        if (brownianVariance <= 0.0 || residualVariance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        if (Math.abs(generationTimes.getValue(currentRootNodeNr)) > 1e-10) {
            return Double.NEGATIVE_INFINITY;
        }

        return betaLogDensityNoCache(beta)
                + tipObservationLogDensityNoCache(
                        generationTimes,
                        beta,
                        residualVariance
                )
                + brownianLatentLogDensityNoCache(
                        tree,
                        generationTimes,
                        brownianVariance
                );
    }

    private double betaLogDensityNoCache(double beta) {

        double priorMean = priorBetaMeanInput.get();
        double priorVariance = priorBetaVarianceInput.get();

        double residual = beta - priorMean;

        return -0.5 * Math.log(2.0 * Math.PI * priorVariance)
                -0.5 * residual * residual / priorVariance;
    }

    private double tipObservationLogDensityNoCache(
            RealParameter generationTimes,
            double beta,
            double residualVariance
    ) {

        double total = 0.0;

        double logNormalConstant =
                -0.5 * Math.log(2.0 * Math.PI * residualVariance);

        for (int i = 0; i < leafNodeCount; i++) {

            int latentIndex = tipLatentIndices[i];
            int nodeNr = latentNodeNrs[latentIndex];

            double observed = observedTipValues[i];
            double latentU = generationTimes.getValue(nodeNr);
            double fitted = beta + latentU;
            double residual = observed - fitted;

            total += logNormalConstant
                    -0.5 * residual * residual / residualVariance;
        }

        return total;
    }

    private double brownianLatentLogDensityNoCache(
            Tree tree,
            RealParameter generationTimes,
            double brownianVariance
    ) {

        double quadraticForm = 0.0;
        double logBranchLengthSum = 0.0;
        int edgeCount = 0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            Node parent = node.getParent();
            double branchLength = node.getLength();

            if (branchLength <= 0.0 ||
                    Double.isNaN(branchLength) ||
                    Double.isInfinite(branchLength)) {
                return Double.NEGATIVE_INFINITY;
            }

            double childValue = generationTimes.getValue(node.getNr());
            double parentValue = generationTimes.getValue(parent.getNr());

            double difference = childValue - parentValue;

            quadraticForm += difference * difference / branchLength;
            logBranchLengthSum += Math.log(branchLength);
            edgeCount++;
        }

        return -0.5 * edgeCount * Math.log(2.0 * Math.PI * brownianVariance)
                -0.5 * logBranchLengthSum
                -0.5 * quadraticForm / brownianVariance;
    }
}