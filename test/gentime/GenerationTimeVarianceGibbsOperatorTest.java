package gentime;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeVarianceGibbsOperatorTest
        extends GenerationTimeVarianceGibbsOperator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree used only for Brownian variance diagnostics.",
            Input.Validate.OPTIONAL
    );

    public Input<String> outputDirectoryInput = new Input<>(
            "outputDirectory",
            "Directory where Brownian variance diagnostic CSV is written.",
            "."
    );

    public Input<Boolean> writeDiagnosticsInput = new Input<>(
            "writeDiagnostics",
            "Write Brownian variance Gibbs update diagnostics to CSV.",
            false
    );

    public Input<Boolean> debugInput = new Input<>(
            "debug",
            "Print variance Gibbs diagnostics when true.",
            false
    );

    private long diagnosticProposalCount = 0L;
    private boolean wroteDiagnosticHeader = false;

    @Override
    public void initAndValidate() {

        super.initAndValidate();

        if (debugInput.get()) {
            GenerationTimePrecisionBuilder precisionBuilder =
                    precisionBuilderInput.get();

            System.out.println("GenerationTimeVarianceGibbsOperatorTest initialised.");
            System.out.println("nodeCount = " + precisionBuilder.getNodeCount());
            System.out.println(
                    "uniquePrecisionTriplets = " +
                            precisionBuilder.getUniqueTripletCount()
            );
        }
    }

    @Override
    public double proposal() {

        RealParameter generationTimes = generationTimesInput.get();
        RealParameter brownianVariance = brownianVarianceInput.get();
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        precisionBuilder.refreshFromTreeIfNeeded();

        double oldVariance = brownianVariance.getValue();

        /*
         * Compute the Brownian quadratic form in two independent ways:
         * 1. using accumulated sparse precision triplets;
         * 2. using explicit edge increments.
         */
        double quadraticFormTriplet = sparseQuadraticForm(
                generationTimes,
                precisionBuilder
        );

        double quadraticFormEdges = edgeIncrementQuadraticForm(
                generationTimes,
                precisionBuilder
        );

        int brownianRank = precisionBuilder.getEdgeCount();

        double priorAlpha = priorAlphaInput.get();
        double priorBeta = priorBetaInput.get();

        double posteriorAlpha = priorAlpha + 0.5 * brownianRank;
        double posteriorBeta = priorBeta + 0.5 * quadraticFormTriplet;

        double oldConditionalLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        oldVariance,
                        posteriorAlpha,
                        posteriorBeta
                );

        double gammaDraw =
                GenerationTimeMathUtils.sampleGammaShapeScaleOne(posteriorAlpha);

        double varianceDraw = posteriorBeta / gammaDraw;

        if (varianceDraw <= 0.0 ||
                Double.isNaN(varianceDraw) ||
                Double.isInfinite(varianceDraw)) {
            return Double.NEGATIVE_INFINITY;
        }

        double newConditionalLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        varianceDraw,
                        posteriorAlpha,
                        posteriorBeta
                );

        double logHastingsRatio =
                oldConditionalLogDensity - newConditionalLogDensity;

        brownianVariance.setValue(0, varianceDraw);

        diagnosticProposalCount++;

        if (writeDiagnosticsInput.get()) {
            writeBrownianVarianceDiagnostics(
                    oldVariance,
                    quadraticFormTriplet,
                    quadraticFormEdges,
                    brownianRank,
                    priorAlpha,
                    priorBeta,
                    posteriorAlpha,
                    posteriorBeta,
                    gammaDraw,
                    varianceDraw
            );
        }

        if (debugInput.get()) {
            System.out.println("Variance Gibbs update:");
            System.out.println("oldVariance = " + oldVariance);
            System.out.println("quadraticFormTriplet = " + quadraticFormTriplet);
            System.out.println("quadraticFormEdges = " + quadraticFormEdges);
            System.out.println(
                    "quadraticFormAbsDiff = " +
                            Math.abs(quadraticFormTriplet - quadraticFormEdges)
            );
            System.out.println("brownianRank = " + brownianRank);
            System.out.println("posteriorAlpha = " + posteriorAlpha);
            System.out.println("posteriorBeta = " + posteriorBeta);
            System.out.println("gammaDraw = " + gammaDraw);
            System.out.println("varianceDraw = " + varianceDraw);
        }

        return logHastingsRatio;
    }

    // Append Brownian variance Gibbs diagnostics to CSV
   
    private void writeBrownianVarianceDiagnostics(
            double oldVariance,
            double quadraticFormTriplet,
            double quadraticFormEdges,
            int brownianRank,
            double priorAlpha,
            double priorBeta,
            double posteriorAlpha,
            double posteriorBeta,
            double gammaDraw,
            double varianceDraw
    ) {

        Tree tree = treeInput.get();

        int rootNodeNr = -1;
        double rootValue = Double.NaN;

        if (tree != null) {
            rootNodeNr = tree.getRoot().getNr();
            rootValue = generationTimesInput.get().getValue(rootNodeNr);
        }

        String outputDirectory = outputDirectoryInput.get();
        String path = outputDirectory + "/brownian_variance_gibbs_diagnostics.csv";

        try (PrintWriter writer = new PrintWriter(new FileWriter(path, true))) {

            if (!wroteDiagnosticHeader) {
                writer.println(
                        "proposal,oldVariance,quadraticFormTriplet,quadraticFormEdges," +
                                "quadraticFormAbsDiff,brownianRank,generationTimesDimension," +
                                "precisionNodeCount,precisionEdgeCount,rootNodeNr,rootValue," +
                                "priorAlpha,priorBeta,posteriorAlpha,posteriorBeta,gammaDraw,varianceDraw"
                );
                wroteDiagnosticHeader = true;
            }

            writer.println(
                    diagnosticProposalCount + "," +
                            oldVariance + "," +
                            quadraticFormTriplet + "," +
                            quadraticFormEdges + "," +
                            Math.abs(quadraticFormTriplet - quadraticFormEdges) + "," +
                            brownianRank + "," +
                            generationTimesInput.get().getDimension() + "," +
                            precisionBuilderInput.get().getNodeCount() + "," +
                            precisionBuilderInput.get().getEdgeCount() + "," +
                            rootNodeNr + "," +
                            rootValue + "," +
                            priorAlpha + "," +
                            priorBeta + "," +
                            posteriorAlpha + "," +
                            posteriorBeta + "," +
                            gammaDraw + "," +
                            varianceDraw
            );

        } catch (IOException e) {
            throw new RuntimeException(
                    "Could not write Brownian variance diagnostics to " + path,
                    e
            );
        }
    }
}