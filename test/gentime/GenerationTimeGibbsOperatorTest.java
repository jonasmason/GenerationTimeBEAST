package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class GenerationTimeGibbsOperatorTest extends Operator {

    // XML-configurable runtime inputs

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

    // Inputs for validation/testing/debugging options

    public Input<Boolean> debugInput = new Input<>(
            "debug",
            "Print Gibbs operator diagnostics when true.",
            false
    );

    public Input<Boolean> writeBetaInput = new Input<>(
            "writeBeta",
            "Whether this diagnostic Gibbs operator writes beta to the BEAST state.",
            true
    );

    public Input<Boolean> writeGenerationTimesInput = new Input<>(
            "writeGenerationTimes",
            "Whether this diagnostic Gibbs operator writes generationTimes to the BEAST state.",
            true
    );

    public Input<String> outputDirectoryInput = new Input<>(
            "outputDirectory",
            "Directory where Gibbs diagnostic CSV files are written.",
            "."
    );

    public Input<Integer> maxCgIterationsInput = new Input<>(
            "maxCgIterations",
            "Maximum number of conjugate-gradient iterations.",
            10000
    );

    public Input<Double> cgToleranceInput = new Input<>(
            "cgTolerance",
            "Relative conjugate-gradient tolerance.",
            1e-5
    );

    public Input<Boolean> writePosteriorMeanInput = new Input<>(
            "writePosteriorMean",
            "Whether to compute posterior means for beast_node_order.csv diagnostics.",
            true
    );

    public Input<Boolean> writeLatentPrecisionDiagnosticsInput = new Input<>(
            "writeLatentPrecisionDiagnostics",
            "Whether to write beast_latent_precision_triplets.csv diagnostics.",
            false
    );

    public Input<String> solverInput = new Input<>(
            "solver",
            "Linear solver for the Gaussian Gibbs update: 'pcg', 'internal', or 'csparsej'. Default is 'csparsej'.",
            "csparsej"
    );

    public Input<Boolean> compareSolversInput = new Input<>(
            "compareSolvers",
            "When true, compare CSparseJ linear-system solutions against the internal sparse Cholesky reference solver.",
            false
    );

    public Input<Boolean> profileInput = new Input<>(
            "profile",
            "Print timing diagnostics for the Gibbs operator.",
            false
    );

    public Input<Integer> profileFrequencyInput = new Input<>(
            "profileFrequency",
            "Print Gibbs timing diagnostics every this many Gibbs proposals.",
            100
    );

    public Input<Boolean> printPrecisionRefreshInput = new Input<>(
            "printPrecisionRefresh",
            "Print sparse precision refresh diagnostics for dynamic-tree debugging.",
            false
    );

    public Input<Integer> snapshotFrequencyInput = new Input<>(
            "snapshotFrequency",
            "Write dynamic-tree validation snapshots every this many Gibbs proposals. Use 0 to disable.",
            0
    );

    public Input<String> snapshotPrefixInput = new Input<>(
            "snapshotPrefix",
            "Prefix for dynamic-tree validation snapshot CSV files.",
            "gt_snapshot"
    );

    // Tree dimensions and node-index bookkeeping

    private int nodeCount;
    private int leafNodeCount;
    private int latentNodeCount;
    private int thetaCount;
    private int rootNodeNr;

    // Mapping between BEAST node numbers, latent vector indices, and observed tip data

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

    private double[] lastPosteriorMean;
    private double[] lastGibbsDraw;
    private double[] lastPriorDraw;
    private double[] lastDelta;
    private double[] lastDeltaRhs;

    // Tracks whether the shared precision builder has changed since this operator last refreshed

    private long seenPrecisionStructureVersion = -1L;
    private long seenPrecisionNumericVersion = -1L;

    // Posterior precision matrix C stored in compressed sparse column form

    private int cscNonZeros;
    private int[] cscColPtr;
    private int[] cscRowIdx;
    private double[] cscValues;

    private int cscBetaBetaIndex;
    private int[] cscBetaTipIndex;
    private int[] cscTipBetaIndex;
    private int[] cscTipDiagonalIndex;
    private int[] cscLatentPrecisionIndex;

    // Sparse Cholesky solvers

    private InternalSparseCholeskySolver internalSparseCholeskySolver;
    private CSparseJSolver csparseJSolver;

    // Validation-only bookkeeping, profiling, and output state

    private static boolean wroteNodeOrderFile = false;

    private int pcgIterations;
    private double pcgFinalResidual;

    private long refreshNoneCount = 0L;
    private long refreshNumericCount = 0L;
    private long refreshTopologyCount = 0L;

    private boolean printedCsparseProfile = false;

    private long profileProposalCount = 0L;
    private long snapshotProposalCount = 0L;

    private long profileTotalNanos = 0L;
    private long profilePosteriorMeanNanos = 0L;
    private long profilePriorDrawNanos = 0L;
    private long profileRhsNanos = 0L;
    private long profileSolverNanos = 0L;
    private long profileCsparseUpdateNanos = 0L;
    private long profileCsparseFactorNanos = 0L;
    private long profileCsparseSolveNanos = 0L;
    private long profileCombineNanos = 0L;
    private long profileStateWriteNanos = 0L;
    private long profileDebugNanos = 0L;

    private long profileTotalPcgIterations = 0L;
    private int profileMaxPcgIterations = 0;

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
        String solver = solverInput.get();

        if (!solver.equals("pcg") &&
                !solver.equals("internal") &&
                !solver.equals("csparsej")) {
            throw new IllegalArgumentException(
                    "solver must be 'pcg', 'internal', or 'csparsej', but found: " + solver
            );
        }

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

        if (writeLatentPrecisionDiagnosticsInput.get()) {
            writeLatentPrecisionDiagnostics();
        }

        buildPosteriorPrecisionCscPattern();
        rebuildSparseSolversForCurrentPattern();

        seenPrecisionStructureVersion = precisionBuilder.getStructureVersion();
        seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();

        if (debugInput.get()) {
            printInitialDebugDiagnostics(tree, precisionBuilder);
        }
    }

    @Override
    public double proposal() {

        long proposalStart = System.nanoTime();

        long posteriorMeanNanos = 0L;
        long priorDrawNanos = 0L;
        long rhsNanos = 0L;
        long solverNanos = 0L;
        long csparseUpdateNanos = 0L;
        long csparseFactorNanos = 0L;
        long csparseSolveNanos = 0L;
        long combineNanos = 0L;
        long stateWriteNanos = 0L;
        long debugNanos = 0L;

        Tree tree = treeInput.get();

        refreshPrecisionStructuresIfNeeded();

        double brownianVariance = brownianVarianceInput.get().getValue();
        double residualVariance = residualVarianceInput.get().getValue();

        if (brownianVariance <= 0.0 || residualVariance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        if (!wroteNodeOrderFile && writePosteriorMeanInput.get()) {

            long posteriorMeanStart = System.nanoTime();

            lastPosteriorMean = computePosteriorMean(
                    brownianVariance,
                    residualVariance
            );

            posteriorMeanNanos += System.nanoTime() - posteriorMeanStart;

            writeDetailedNodeOrderFile(tree);
            wroteNodeOrderFile = true;
        }

        long priorDrawStart = System.nanoTime();

        // Draw theta* = (beta*, u*) from the prior on the current tree
        lastPriorDraw = drawPriorTheta(brownianVariance);

        priorDrawNanos += System.nanoTime() - priorDrawStart;

        long rhsStart = System.nanoTime();

        // Build the residual-based RHS for the Gaussian correction
        lastDeltaRhs = buildDeltaRhs(lastPriorDraw, residualVariance);

        rhsNanos += System.nanoTime() - rhsStart;

        SolverTiming solverTiming = new SolverTiming();

        long solverStart = System.nanoTime();

        // Solve for the correction delta using the current posterior precision
        lastDelta = solveGaussianSystem(
                lastDeltaRhs,
                brownianVariance,
                residualVariance,
                maxCgIterationsInput.get(),
                cgToleranceInput.get(),
                solverTiming
        );

        solverNanos += System.nanoTime() - solverStart;
        csparseUpdateNanos += solverTiming.csparseUpdateNanos;
        csparseFactorNanos += solverTiming.csparseFactorNanos;
        csparseSolveNanos += solverTiming.csparseSolveNanos;

        long combineStart = System.nanoTime();

        lastGibbsDraw = new double[thetaCount];

        // Final Gibbs draw is theta = theta* + delta
        for (int i = 0; i < thetaCount; i++) {
            lastGibbsDraw[i] = lastPriorDraw[i] + lastDelta[i];
        }

        combineNanos += System.nanoTime() - combineStart;

        // BEAST acceptance accounting -> calculate target density before and after writing state
        double oldGtLogP = gtLikelihoodInput.get().calculateLogP();

        long stateWriteStart = System.nanoTime();

        if (writeBetaInput.get()) {
            betaInput.get().setValue(0, lastGibbsDraw[0]);
        }

        if (writeGenerationTimesInput.get()) {

            RealParameter generationTimes = generationTimesInput.get();

            generationTimes.setValue(rootNodeNr, 0.0);

            for (int latentIndex = 0; latentIndex < latentNodeCount; latentIndex++) {

                int nodeNr = latentNodeNrs[latentIndex];
                double value = lastGibbsDraw[1 + latentIndex];

                generationTimes.setValue(nodeNr, value);
            }
        }

        stateWriteNanos += System.nanoTime() - stateWriteStart;

        if (debugInput.get()) {

            long debugStart = System.nanoTime();

            printProposalDebugDiagnostics();

            debugNanos += System.nanoTime() - debugStart;
        }

        writeSnapshotIfNeeded(tree, brownianVariance, residualVariance);

        long totalNanos = System.nanoTime() - proposalStart;

        double newGtLogP = gtLikelihoodInput.get().calculateLogP();

        updateProfileTotals(
                totalNanos,
                posteriorMeanNanos,
                priorDrawNanos,
                rhsNanos,
                solverNanos,
                csparseUpdateNanos,
                csparseFactorNanos,
                csparseSolveNanos,
                combineNanos,
                stateWriteNanos,
                debugNanos
        );

        return oldGtLogP - newGtLogP;
    }

    // Rebuild bookkeeping and sparse precision structures when the tree topology or branch lengths change

    private void refreshPrecisionStructuresIfNeeded() {

        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        String localRefreshType = precisionBuilder.refreshFromTreeIfNeeded();
        String globalRefreshType = precisionBuilder.getLastGlobalRefreshType();

        boolean structureChanged =
                precisionBuilder.getStructureVersion() != seenPrecisionStructureVersion;

        boolean numericChanged =
                precisionBuilder.getNumericVersion() != seenPrecisionNumericVersion;

        if (structureChanged) {

            refreshTopologyCount++;

            buildNodeBookkeeping();
            buildLatentPrecisionTriplets();
            buildPosteriorPrecisionCscPattern();
            rebuildSparseSolversForCurrentPattern();

            seenPrecisionStructureVersion = precisionBuilder.getStructureVersion();
            seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();

            if (printPrecisionRefreshInput.get()) {
                System.out.println(
                        "GenerationTimeGibbsOperatorTest precision refresh: topology path; " +
                                "localRefreshType=" + localRefreshType +
                                " globalRefreshType=" + globalRefreshType +
                                " structureVersion=" + seenPrecisionStructureVersion +
                                " numericVersion=" + seenPrecisionNumericVersion +
                                " lastStructureRefreshVersion=" + precisionBuilder.getLastStructureRefreshVersion() +
                                " lastNumericRefreshVersion=" + precisionBuilder.getLastNumericRefreshVersion() +
                                " cscNonZeros=" + cscNonZeros +
                                " topologyCount=" + refreshTopologyCount +
                                " numericCount=" + refreshNumericCount +
                                " noneCount=" + refreshNoneCount
                );
            }

            return;
        }

        if (numericChanged) {

            refreshNumericCount++;

            buildLatentPrecisionTriplets();

            seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();

            if (printPrecisionRefreshInput.get() && (refreshNumericCount % 100) == 0) {
                System.out.println(
                        "GenerationTimeGibbsOperatorTest precision refresh summary: " +
                                "localRefreshType=" + localRefreshType +
                                " globalRefreshType=" + globalRefreshType +
                                " topologyCount=" + refreshTopologyCount +
                                " numericCount=" + refreshNumericCount +
                                " noneCount=" + refreshNoneCount +
                                " structureVersion=" + seenPrecisionStructureVersion +
                                " numericVersion=" + seenPrecisionNumericVersion +
                                " lastStructureRefreshVersion=" + precisionBuilder.getLastStructureRefreshVersion() +
                                " lastNumericRefreshVersion=" + precisionBuilder.getLastNumericRefreshVersion() +
                                " cscNonZeros=" + cscNonZeros
                );
            }

            return;
        }

        refreshNoneCount++;
    }

    private void rebuildSparseSolversForCurrentPattern() {

        String solver = solverInput.get();

        if (solver.equals("internal") || compareSolversInput.get()) {
            internalSparseCholeskySolver = new InternalSparseCholeskySolver(thetaCount);
            internalSparseCholeskySolver.analyzePattern(cscColPtr, cscRowIdx);
        }

        if (solver.equals("csparsej") || compareSolversInput.get()) {
            csparseJSolver = new CSparseJSolver(thetaCount);
            csparseJSolver.analyzePattern(cscColPtr, cscRowIdx);
        }
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

    private void writeLatentPrecisionDiagnostics() {

        try {

            String outputDirectory = outputDirectoryInput.get();

            java.io.File outputFile = new java.io.File(
                    outputDirectory,
                    "beast_latent_precision_triplets.csv"
            );

            PrintWriter out = new PrintWriter(
                    new FileWriter(outputFile)
            );

            out.println(
                    "latent_row,latent_col,row_node_nr,col_node_nr,value"
            );

            for (int k = 0; k < nLatentTriplets; k++) {

                int latentRow = qLatentRow[k];
                int latentCol = qLatentCol[k];

                out.println(
                        latentRow + "," +
                                latentCol + "," +
                                latentNodeNrs[latentRow] + "," +
                                latentNodeNrs[latentCol] + "," +
                                qLatentValue[k]
                );
            }

            out.close();

            System.out.println(
                    "Wrote beast_latent_precision_triplets.csv to " +
                            outputFile.getAbsolutePath()
            );

        } catch (IOException e) {
            throw new RuntimeException(
                    "Could not write beast_latent_precision_triplets.csv",
                    e
            );
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

    // Factorise and solve the sparse Gaussian system, optionally comparing solver paths

    private double[] solveGaussianSystem(
            double[] rhs,
            double brownianVariance,
            double residualVariance,
            int maxIterations,
            double tolerance,
            SolverTiming solverTiming
    ) {

        String solver = solverInput.get();

        if (compareSolversInput.get()) {
            return solveGaussianSystemWithComparison(
                    rhs,
                    brownianVariance,
                    residualVariance,
                    maxIterations,
                    tolerance,
                    solverTiming
            );
        }

        if (solver.equals("pcg")) {
            return solveByConjugateGradient(
                    rhs,
                    brownianVariance,
                    residualVariance,
                    maxIterations,
                    tolerance
            );
        }

        if (solver.equals("internal")) {
            return solveByInternalSparseCholesky(
                    rhs,
                    brownianVariance,
                    residualVariance,
                    solverTiming
            );
        }

        if (solver.equals("csparsej")) {
            return solveByCSparseJ(
                    rhs,
                    brownianVariance,
                    residualVariance,
                    solverTiming
            );
        }

        throw new IllegalArgumentException("Unknown solver: " + solver);
    }

    private double[] solveGaussianSystemWithComparison(
            double[] rhs,
            double brownianVariance,
            double residualVariance,
            int maxIterations,
            double tolerance,
            SolverTiming solverTiming
    ) {

        String solver = solverInput.get();

        double[] pcgSolution = solveByConjugateGradient(
                rhs,
                brownianVariance,
                residualVariance,
                maxIterations,
                tolerance
        );

        double[] internalSolution = solveByInternalSparseCholesky(
                rhs,
                brownianVariance,
                residualVariance,
                solverTiming
        );

        double[] csparseJSolution = solveByCSparseJ(
                rhs,
                brownianVariance,
                residualVariance,
                solverTiming
        );

        double pcgResidual = gaussianSystemResidualNorm(
                pcgSolution,
                rhs,
                brownianVariance,
                residualVariance
        );

        double internalResidual = gaussianSystemResidualNorm(
                internalSolution,
                rhs,
                brownianVariance,
                residualVariance
        );
        
        double csparseJResidual = gaussianSystemResidualNorm(
                csparseJSolution,
                rhs,
                brownianVariance,
                residualVariance
        );

        System.out.println(
                "CSparseJ validation: " +
                        "max_abs_difference_vs_internal_sparse_cholesky = " +
                        maxAbsDifference(internalSolution, csparseJSolution) +
                        " max_abs_difference_vs_pcg = " +
                        maxAbsDifference(pcgSolution, csparseJSolution) +
                        " csparsej_residual_norm = " +
                        csparseJResidual +
                        " internal_sparse_cholesky_residual_norm = " +
                        internalResidual +
                        " pcg_residual_norm = " +
                        pcgResidual +
                        " pcg_iterations = " +
                        pcgIterations +
                        " pcg_final_residual = " +
                        pcgFinalResidual +
                        " csparsej_cholesky_nonzeros = " +
                        csparseJSolver.getCholeskyNonZeros()
        );

        if (solver.equals("csparsej")) {
            return csparseJSolution;
        }

        if (solver.equals("pcg")) {
            return pcgSolution;
        }

        if (solver.equals("internal")) {
            return internalSolution;
        }

        throw new IllegalArgumentException("Unknown solver: " + solver);
    }

    private double[] solveByInternalSparseCholesky(
            double[] rhs,
            double brownianVariance,
            double residualVariance,
            SolverTiming solverTiming
    ) {

        long updateStart = System.nanoTime();

        updatePosteriorPrecisionCscValues(
                brownianVariance,
                residualVariance
        );

        solverTiming.csparseUpdateNanos += System.nanoTime() - updateStart;

        long factorStart = System.nanoTime();

        internalSparseCholeskySolver.factorize(cscValues);

        solverTiming.csparseFactorNanos += System.nanoTime() - factorStart;

        printInternalSparseCholeskyProfileOnce();

        long solveStart = System.nanoTime();

        double[] solution = internalSparseCholeskySolver.solve(rhs);

        solverTiming.csparseSolveNanos += System.nanoTime() - solveStart;

        return solution;
    }

    private double[] solveByCSparseJ(
            double[] rhs,
            double brownianVariance,
            double residualVariance,
            SolverTiming solverTiming
    ) {

        long updateStart = System.nanoTime();

        updatePosteriorPrecisionCscValues(
                brownianVariance,
                residualVariance
        );

        solverTiming.csparseUpdateNanos += System.nanoTime() - updateStart;

        long factorStart = System.nanoTime();

        csparseJSolver.factorize(cscValues);

        solverTiming.csparseFactorNanos += System.nanoTime() - factorStart;

        long solveStart = System.nanoTime();

        double[] solution = csparseJSolver.solve(rhs);

        solverTiming.csparseSolveNanos += System.nanoTime() - solveStart;

        return solution;
    }

    // Validation/reference solver path: preconditioned conjugate gradient

    private double[] solveByConjugateGradient(
            double[] rhs,
            double brownianVariance,
            double residualVariance,
            int maxIterations,
            double tolerance
    ) {

        double invResid = 1.0 / residualVariance;
        double invPhylo = 1.0 / brownianVariance;
        double priorBetaVariance = priorBetaVarianceInput.get();

        double[] x = new double[thetaCount];
        double[] r = rhs.clone();

        BlockPreconditioner preconditioner = buildBlockPreconditioner(
                invResid,
                invPhylo,
                priorBetaVariance
        );

        double[] z = new double[thetaCount];
        applyBlockPreconditionerInto(r, preconditioner, z);

        double[] p = z.clone();
        double[] Ap = new double[thetaCount];

        double rzOld = dot(r, z);
        double initialResidualNorm = Math.sqrt(dot(r, r));

        if (initialResidualNorm == 0.0) {
            pcgIterations = 0;
            pcgFinalResidual = 0.0;
            return x;
        }

        int iter;
        double finalResidualNorm = initialResidualNorm;

        for (iter = 1; iter <= maxIterations; iter++) {

            multiplyPosteriorPrecisionInto(
                    p,
                    Ap,
                    invResid,
                    invPhylo,
                    priorBetaVariance
            );

            double denom = dot(p, Ap);

            if (denom == 0.0) {
                throw new IllegalArgumentException(
                        "Preconditioned conjugate gradient failed: zero denominator."
                );
            }

            double alpha = rzOld / denom;

            for (int i = 0; i < thetaCount; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            finalResidualNorm = Math.sqrt(dot(r, r));

            if (finalResidualNorm / initialResidualNorm < tolerance) {
                break;
            }

            applyBlockPreconditionerInto(r, preconditioner, z);

            double rzNew = dot(r, z);
            double beta = rzNew / rzOld;

            for (int i = 0; i < thetaCount; i++) {
                p[i] = z[i] + beta * p[i];
            }

            rzOld = rzNew;
        }

        if (iter > maxIterations) {
            iter = maxIterations;
        }

        pcgIterations = iter;
        pcgFinalResidual = finalResidualNorm;

        return x;
    }

    private static class BlockPreconditioner {

        double[] diagonal;
        int[] tipThetaIndex;
        double[] invResidOverTipDiagonal;
        double invResid;
        double betaSchur;

        BlockPreconditioner(
                double[] diagonal,
                int[] tipThetaIndex,
                double[] invResidOverTipDiagonal,
                double invResid,
                double betaSchur
        ) {
            this.diagonal = diagonal;
            this.tipThetaIndex = tipThetaIndex;
            this.invResidOverTipDiagonal = invResidOverTipDiagonal;
            this.invResid = invResid;
            this.betaSchur = betaSchur;
        }
    }

    private BlockPreconditioner buildBlockPreconditioner(
            double invResid,
            double invPhylo,
            double priorBetaVariance
    ) {

        double[] diagonal = buildDiagonalPreconditioner(
                invResid,
                invPhylo,
                priorBetaVariance
        );

        int[] tipThetaIndex = new int[leafNodeCount];
        double[] invResidOverTipDiagonal = new double[leafNodeCount];

        double betaSchur = diagonal[0];

        for (int i = 0; i < leafNodeCount; i++) {

            int thetaIndex = 1 + tipLatentIndices[i];

            tipThetaIndex[i] = thetaIndex;
            invResidOverTipDiagonal[i] = invResid / diagonal[thetaIndex];

            betaSchur -= invResid * invResidOverTipDiagonal[i];
        }

        if (betaSchur <= 0.0) {
            throw new IllegalArgumentException(
                    "Non-positive Schur complement in block preconditioner: " + betaSchur
            );
        }

        return new BlockPreconditioner(
                diagonal,
                tipThetaIndex,
                invResidOverTipDiagonal,
                invResid,
                betaSchur
        );
    }

    private double[] buildDiagonalPreconditioner(
            double invResid,
            double invPhylo,
            double priorBetaVariance
    ) {

        double[] diagonal = new double[thetaCount];

        diagonal[0] = leafNodeCount * invResid + 1.0 / priorBetaVariance;

        for (int i = 0; i < leafNodeCount; i++) {
            int thetaIndex = 1 + tipLatentIndices[i];
            diagonal[thetaIndex] += invResid;
        }

        for (int k = 0; k < nLatentTriplets; k++) {

            if (qLatentRow[k] == qLatentCol[k]) {
                int thetaIndex = 1 + qLatentRow[k];
                diagonal[thetaIndex] += invPhylo * qLatentValue[k];
            }
        }

        for (int i = 0; i < thetaCount; i++) {
            if (diagonal[i] <= 0.0) {
                throw new IllegalArgumentException(
                        "Non-positive preconditioner diagonal entry at theta index " + i
                );
            }
        }

        return diagonal;
    }

    private void applyBlockPreconditionerInto(
            double[] r,
            BlockPreconditioner preconditioner,
            double[] z
    ) {

        double[] diagonal = preconditioner.diagonal;

        for (int i = 0; i < thetaCount; i++) {
            z[i] = r[i] / diagonal[i];
        }

        double betaRightHandSide = r[0];

        for (int i = 0; i < leafNodeCount; i++) {

            int thetaIndex = preconditioner.tipThetaIndex[i];

            betaRightHandSide -=
                    preconditioner.invResidOverTipDiagonal[i] *
                            r[thetaIndex];
        }

        double zBeta = betaRightHandSide / preconditioner.betaSchur;

        z[0] = zBeta;

        for (int i = 0; i < leafNodeCount; i++) {

            int thetaIndex = preconditioner.tipThetaIndex[i];

            z[thetaIndex] =
                    (r[thetaIndex] - preconditioner.invResid * zBeta) /
                            diagonal[thetaIndex];
        }
    }

    private double gaussianSystemResidualNorm(
            double[] solution,
            double[] rhs,
            double brownianVariance,
            double residualVariance
    ) {

        double invResid = 1.0 / residualVariance;
        double invPhylo = 1.0 / brownianVariance;
        double priorBetaVariance = priorBetaVarianceInput.get();

        double[] product = new double[thetaCount];

        multiplyPosteriorPrecisionInto(
                solution,
                product,
                invResid,
                invPhylo,
                priorBetaVariance
        );

        double sumSquares = 0.0;

        for (int i = 0; i < thetaCount; i++) {
            double residual = product[i] - rhs[i];
            sumSquares += residual * residual;
        }

        return Math.sqrt(sumSquares);
    }

    private void multiplyPosteriorPrecisionInto(
            double[] x,
            double[] out,
            double invResid,
            double invPhylo,
            double priorBetaVariance
    ) {

        Arrays.fill(out, 0.0);

        out[0] +=
                (leafNodeCount * invResid + 1.0 / priorBetaVariance) *
                        x[0];

        for (int i = 0; i < leafNodeCount; i++) {

            int thetaIndex = 1 + tipLatentIndices[i];

            out[0] += invResid * x[thetaIndex];
            out[thetaIndex] += invResid * x[0];
            out[thetaIndex] += invResid * x[thetaIndex];
        }

        for (int k = 0; k < nLatentTriplets; k++) {

            int thetaRow = 1 + qLatentRow[k];
            int thetaCol = 1 + qLatentCol[k];

            out[thetaRow] +=
                    invPhylo *
                            qLatentValue[k] *
                            x[thetaCol];
        }
    }

    private double maxAbsDifference(double[] a, double[] b) {

        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Cannot compare vectors with different lengths."
            );
        }

        double maxDifference = 0.0;

        for (int i = 0; i < a.length; i++) {

            double difference = Math.abs(a[i] - b[i]);

            if (difference > maxDifference) {
                maxDifference = difference;
            }
        }

        return maxDifference;
    }

    private double dot(double[] a, double[] b) {

        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Dot product vectors must have same length."
            );
        }

        double total = 0.0;

        for (int i = 0; i < a.length; i++) {
            total += a[i] * b[i];
        }

        return total;
    }

    // Runtime profiling

    private static class SolverTiming {

        long csparseUpdateNanos = 0L;
        long csparseFactorNanos = 0L;
        long csparseSolveNanos = 0L;
    }

    private void updateProfileTotals(
            long totalNanos,
            long posteriorMeanNanos,
            long priorDrawNanos,
            long rhsNanos,
            long solverNanos,
            long csparseUpdateNanos,
            long csparseFactorNanos,
            long csparseSolveNanos,
            long combineNanos,
            long stateWriteNanos,
            long debugNanos
    ) {

        if (!profileInput.get()) {
            return;
        }

        profileProposalCount++;

        profileTotalNanos += totalNanos;
        profilePosteriorMeanNanos += posteriorMeanNanos;
        profilePriorDrawNanos += priorDrawNanos;
        profileRhsNanos += rhsNanos;
        profileSolverNanos += solverNanos;
        profileCsparseUpdateNanos += csparseUpdateNanos;
        profileCsparseFactorNanos += csparseFactorNanos;
        profileCsparseSolveNanos += csparseSolveNanos;
        profileCombineNanos += combineNanos;
        profileStateWriteNanos += stateWriteNanos;
        profileDebugNanos += debugNanos;

        profileTotalPcgIterations += pcgIterations;

        if (pcgIterations > profileMaxPcgIterations) {
            profileMaxPcgIterations = pcgIterations;
        }

        int frequency = profileFrequencyInput.get();

        if (frequency <= 0) {
            frequency = 100;
        }

        if (profileProposalCount % frequency == 0L) {
            printProfileSummary();
        }
    }

    private void printProfileSummary() {

        double n = (double) profileProposalCount;

        System.out.println("GenerationTimeGibbsOperatorTest profile:");
        System.out.println("  proposals = " + profileProposalCount);
        System.out.println("  mean_total_ms = " + nanosToMillis(profileTotalNanos / n));
        System.out.println("  mean_posterior_mean_ms = " + nanosToMillis(profilePosteriorMeanNanos / n));
        System.out.println("  mean_prior_draw_ms = " + nanosToMillis(profilePriorDrawNanos / n));
        System.out.println("  mean_rhs_ms = " + nanosToMillis(profileRhsNanos / n));
        System.out.println("  mean_solver_total_ms = " + nanosToMillis(profileSolverNanos / n));
        System.out.println("  mean_csparse_update_ms = " + nanosToMillis(profileCsparseUpdateNanos / n));
        System.out.println("  mean_csparse_factor_ms = " + nanosToMillis(profileCsparseFactorNanos / n));
        System.out.println("  mean_csparse_solve_ms = " + nanosToMillis(profileCsparseSolveNanos / n));
        System.out.println("  mean_combine_ms = " + nanosToMillis(profileCombineNanos / n));
        System.out.println("  mean_state_write_ms = " + nanosToMillis(profileStateWriteNanos / n));
        System.out.println("  mean_debug_ms = " + nanosToMillis(profileDebugNanos / n));
        System.out.println("  mean_pcg_iterations = " + (profileTotalPcgIterations / n));
        System.out.println("  max_pcg_iterations = " + profileMaxPcgIterations);
        System.out.println("  last_pcg_iterations = " + pcgIterations);
        System.out.println("  last_pcg_final_residual = " + pcgFinalResidual);

        GenerationTimePrecisionBuilder precisionBuilder =
                precisionBuilderInput.get();

        long topologyCount = precisionBuilder.getTopologyRefreshCount();
        long numericCount = precisionBuilder.getNumericRefreshCount();

        double meanTopologyRefreshMs =
                topologyCount == 0L
                        ? 0.0
                        : nanosToMillis(
                        precisionBuilder.getTotalTopologyRefreshNanos()
                ) / topologyCount;

        double meanNumericRefreshMs =
                numericCount == 0L
                        ? 0.0
                        : nanosToMillis(
                        precisionBuilder.getTotalNumericRefreshNanos()
                ) / numericCount;

        System.out.println("  precision_topology_refresh_count = " + topologyCount);
        System.out.println("  precision_numeric_refresh_count = " + numericCount);
        System.out.println("  mean_precision_topology_refresh_ms = " + meanTopologyRefreshMs);
        System.out.println("  mean_precision_numeric_refresh_ms = " + meanNumericRefreshMs);
        System.out.println(
                "  last_precision_refresh_ms = " +
                        nanosToMillis(precisionBuilder.getLastRefreshNanos())
        );
    }

    private double nanosToMillis(double nanos) {
        return nanos / 1.0e6;
    }

    private double nanosToMillis(long nanos) {
        return nanos / 1.0e6;
    }

    private void printInternalSparseCholeskyProfileOnce() {

        if (!profileInput.get()) {
            return;
        }

        if (printedCsparseProfile) {
            return;
        }

        if (internalSparseCholeskySolver == null) {
            return;
        }

        System.out.println(
                "Internal sparse Cholesky profile: csc_nonzeros = " +
                        cscNonZeros +
                        " cholesky_nonzeros = " +
                        internalSparseCholeskySolver.getCholeskyNonZeros() +
                        " clear_ms = " +
                        internalSparseCholeskySolver.getLastClearMillis() +
                        " permute_ms = " +
                        internalSparseCholeskySolver.getLastPermuteMillis() +
                        " numeric_ms = " +
                        internalSparseCholeskySolver.getLastNumericMillis() +
                        " row_access_ms = " +
                        internalSparseCholeskySolver.getLastRowAccessMillis()
        );

        printedCsparseProfile = true;
    }

    // Export 'snapshots'

    private void writeSnapshotIfNeeded(
            Tree tree,
            double brownianVariance,
            double residualVariance
    ) {

        int snapshotFrequency = snapshotFrequencyInput.get();

        if (snapshotFrequency <= 0) {
            return;
        }

        snapshotProposalCount++;

        if ((snapshotProposalCount % snapshotFrequency) != 0L) {
            return;
        }

        String outputDirectory = outputDirectoryInput.get();
        String prefix = snapshotPrefixInput.get();

        String snapshotId = prefix + "_" + snapshotProposalCount;

        writeSnapshotTreeEdges(tree, outputDirectory, snapshotId);
        writeSnapshotPrecisionTriplets(outputDirectory, snapshotId);
        writeSnapshotState(tree, outputDirectory, snapshotId, brownianVariance, residualVariance);
    }

    private void writeSnapshotTreeEdges(
            Tree tree,
            String outputDirectory,
            String snapshotId
    ) {

        try {

            java.io.File outputFile = new java.io.File(
                    outputDirectory,
                    snapshotId + "_tree_edges.csv"
            );

            PrintWriter out = new PrintWriter(new FileWriter(outputFile));

            out.println(
                    "snapshot,node_nr,node_id,parent_nr,is_leaf,is_root,height,length,clade_key"
            );

            for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

                Node node = tree.getNode(nodeNr);

                int parentNr = node.isRoot() ? -1 : node.getParent().getNr();

                out.println(
                        snapshotId + "," +
                                nodeNr + "," +
                                safeCsv(node.getID()) + "," +
                                parentNr + "," +
                                node.isLeaf() + "," +
                                node.isRoot() + "," +
                                node.getHeight() + "," +
                                node.getLength() + "," +
                                safeCsv(cladeKey(node))
                );
            }

            out.close();

        } catch (IOException e) {
            throw new RuntimeException(
                    "Could not write snapshot tree edges for " + snapshotId,
                    e
            );
        }
    }

    private void writeSnapshotPrecisionTriplets(
            String outputDirectory,
            String snapshotId
    ) {

        try {

            GenerationTimePrecisionBuilder precisionBuilder =
                    precisionBuilderInput.get();

            java.io.File outputFile = new java.io.File(
                    outputDirectory,
                    snapshotId + "_precision_triplets.csv"
            );

            PrintWriter out = new PrintWriter(new FileWriter(outputFile));

            out.println(
                    "snapshot,row_node_nr,col_node_nr,value"
            );

            for (int k = 0; k < precisionBuilder.getUniqueTripletCount(); k++) {

                out.println(
                        snapshotId + "," +
                                precisionBuilder.getQUniqueRow(k) + "," +
                                precisionBuilder.getQUniqueCol(k) + "," +
                                precisionBuilder.getQUniqueValue(k)
                );
            }

            out.close();

        } catch (IOException e) {
            throw new RuntimeException(
                    "Could not write snapshot precision triplets for " + snapshotId,
                    e
            );
        }
    }

    private void writeSnapshotState(
            Tree tree,
            String outputDirectory,
            String snapshotId,
            double brownianVariance,
            double residualVariance
    ) {

        try {

            RealParameter generationTimes = generationTimesInput.get();

            java.io.File outputFile = new java.io.File(
                    outputDirectory,
                    snapshotId + "_state.csv"
            );

            PrintWriter out = new PrintWriter(new FileWriter(outputFile));

            out.println(
                    "snapshot,node_nr,node_id,is_leaf,is_root,generation_time,beta,brownian_variance,residual_variance,root_node_nr,structure_version,numeric_version,last_structure_refresh_version,last_numeric_refresh_version"
            );

            double beta = betaInput.get().getValue();

            GenerationTimePrecisionBuilder precisionBuilder =
                    precisionBuilderInput.get();

            for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

                Node node = tree.getNode(nodeNr);

                out.println(
                        snapshotId + "," +
                                nodeNr + "," +
                                safeCsv(node.getID()) + "," +
                                node.isLeaf() + "," +
                                node.isRoot() + "," +
                                generationTimes.getValue(nodeNr) + "," +
                                beta + "," +
                                brownianVariance + "," +
                                residualVariance + "," +
                                rootNodeNr + "," +
                                precisionBuilder.getStructureVersion() + "," +
                                precisionBuilder.getNumericVersion() + "," +
                                precisionBuilder.getLastStructureRefreshVersion() + "," +
                                precisionBuilder.getLastNumericRefreshVersion()
                );
            }

            out.close();

        } catch (IOException e) {
            throw new RuntimeException(
                    "Could not write snapshot state for " + snapshotId,
                    e
            );
        }
    }

    // Export posterior mean and node-order

    private double[] computePosteriorMean(
            double brownianVariance,
            double residualVariance
    ) {

        double[] rhs = new double[thetaCount];

        double betaPriorMean = priorBetaMeanInput.get();
        double betaPriorVariance = priorBetaVarianceInput.get();

        rhs[0] = betaPriorMean / betaPriorVariance;

        double residualPrecision = 1.0 / residualVariance;

        for (int i = 0; i < leafNodeCount; i++) {

            double y = observedTipValues[i];
            int latentIndex = tipLatentIndices[i];

            rhs[0] += residualPrecision * y;
            rhs[1 + latentIndex] += residualPrecision * y;
        }

        return solveGaussianSystem(
                rhs,
                brownianVariance,
                residualVariance,
                maxCgIterationsInput.get(),
                cgToleranceInput.get(),
                new SolverTiming()
        );
    }

    private void writeDetailedNodeOrderFile(Tree tree) {

        try {

            String outputDirectory = outputDirectoryInput.get();

            java.io.File outputFile = new java.io.File(
                    outputDirectory,
                    "beast_node_order.csv"
            );

            PrintWriter out = new PrintWriter(
                    new FileWriter(outputFile)
            );

            out.println(
                    "beast_node_nr,node_id,is_leaf,is_root,parent_nr,left_child_nr,right_child_nr," +
                            "height,length,clade_key,latent_index,posterior_mean"
            );

            for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

                Node node = tree.getNode(nodeNr);

                int parentNr = node.isRoot() ? -1 : node.getParent().getNr();

                int leftChildNr = -1;
                int rightChildNr = -1;

                if (!node.isLeaf()) {
                    leftChildNr = node.getChild(0).getNr();
                    rightChildNr = node.getChild(1).getNr();
                }

                int latentIndex = -1;
                double mean = Double.NaN;

                if (isLatentNode[nodeNr]) {
                    latentIndex = latentIndexByNodeNr[nodeNr];

                    if (lastPosteriorMean == null) {
                        mean = Double.NaN;
                    } else {
                        mean = lastPosteriorMean[1 + latentIndex];
                    }
                }

                out.println(
                        nodeNr + "," +
                                safeCsv(node.getID()) + "," +
                                node.isLeaf() + "," +
                                node.isRoot() + "," +
                                parentNr + "," +
                                leftChildNr + "," +
                                rightChildNr + "," +
                                node.getHeight() + "," +
                                node.getLength() + "," +
                                safeCsv(cladeKey(node)) + "," +
                                latentIndex + "," +
                                mean
                );
            }

            out.close();

            System.out.println(
                    "Wrote detailed beast_node_order.csv to " +
                            outputFile.getAbsolutePath()
            );

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    // Debug print helpers

    private void printInitialDebugDiagnostics(
            Tree tree,
            GenerationTimePrecisionBuilder precisionBuilder
    ) {

        System.out.println("GenerationTimeGibbsOperatorTest initialised with mixed-model Gibbs updates.");
        System.out.println("Model: y = beta + u_tip + e");
        System.out.println("nodeCount = " + tree.getNodeCount());
        System.out.println("leafNodeCount = " + tree.getLeafNodeCount());
        System.out.println("rootNodeNr = " + rootNodeNr);
        System.out.println("latentNodeCount = " + latentNodeCount);
        System.out.println("thetaCount = " + thetaCount);
        System.out.println("uniquePrecisionTriplets = " + precisionBuilder.getUniqueTripletCount());
        System.out.println("nLatentTriplets = " + nLatentTriplets);
        System.out.println("cscNonZeros = " + cscNonZeros);
        System.out.println("solver = " + solverInput.get());
    }

    private void printProposalDebugDiagnostics() {

        System.out.println("Mixed-model Gibbs draw:");
        System.out.println("betaDraw = " + lastGibbsDraw[0]);
        System.out.println("pcgIterations = " + pcgIterations);
        System.out.println("pcgFinalResidual = " + pcgFinalResidual);

        for (int i = 0; i < latentNodeCount; i++) {
            System.out.println(
                    "latentIndex=" + i +
                            " nodeNr=" + latentNodeNrs[i] +
                            " posteriorMean=" + (
                            lastPosteriorMean == null ? Double.NaN : lastPosteriorMean[1 + i]
                    ) +
                            " draw=" + lastGibbsDraw[1 + i]
            );
        }
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

    // CSV/tree helpers

    private String cladeKey(Node node) {

        List<String> tips = new ArrayList<>();
        collectDescendantTips(node, tips);

        Collections.sort(tips);

        return String.join("|", tips);
    }

    private void collectDescendantTips(Node node, List<String> tips) {

        if (node.isLeaf()) {
            tips.add(node.getID());
            return;
        }

        for (int i = 0; i < node.getChildCount(); i++) {
            collectDescendantTips(node.getChild(i), tips);
        }
    }

    private String safeCsv(String value) {

        if (value == null) {
            return "NA";
        }

        return value.replace(",", "_");
    }
}