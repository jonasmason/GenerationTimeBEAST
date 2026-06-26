package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeClockResidualVarianceGibbsOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root branches have clock residuals.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockResidualsInput = new Input<>(
            "clockResiduals",
            "Branch-specific log-rate residuals indexed by child BEAST node number. Root residual is ignored.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockResidualVarianceInput = new Input<>(
            "clockResidualVariance",
            "Clock residual variance parameter to update.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimeClockResidualPrior> clockResidualPriorInput = new Input<>(
            "clockResidualPrior",
            "Clock residual prior to recalculate after this operator writes the variance.",
            Input.Validate.REQUIRED
    );

    public Input<Distribution> clockResidualVariancePriorInput = new Input<>(
            "clockResidualVariancePrior",
            "Clock residual variance prior to recalculate after this operator writes the variance.",
            Input.Validate.REQUIRED
    );

    public Input<Double> priorAlphaInput = new Input<>(
            "priorAlpha",
            "Inverse-gamma prior shape parameter.",
            2.0
    );

    public Input<Double> priorBetaInput = new Input<>(
            "priorBeta",
            "Inverse-gamma prior scale parameter.",
            1.0
    );

    private int branchCount;

    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        RealParameter clockResiduals = clockResidualsInput.get();
        RealParameter clockResidualVariance = clockResidualVarianceInput.get();

        if (clockResiduals.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "clockResiduals dimension must match tree node count. " +
                    "Residuals are indexed by child node number; root residual is ignored."
            );
        }

        if (clockResidualVariance.getDimension() != 1) {
            throw new IllegalArgumentException(
                    "clockResidualVariance must be a scalar RealParameter."
            );
        }

        if (priorAlphaInput.get() <= 0.0 || priorBetaInput.get() <= 0.0) {
            throw new IllegalArgumentException(
                    "priorAlpha and priorBeta must both be positive."
            );
        }

        branchCount = tree.getNodeCount() - 1;
    }

    @Override
    public double proposal() {

        Tree tree = treeInput.get();
        RealParameter clockResiduals = clockResidualsInput.get();
        RealParameter clockResidualVariance = clockResidualVarianceInput.get();

        clockResidualPriorInput.get().calculateLogP();
        clockResidualVariancePriorInput.get().calculateLogP();

        double oldVariance = clockResidualVariance.getValue();

        double residualSumOfSquares = residualSumOfSquares(
                tree,
                clockResiduals
        );

        double priorAlpha = priorAlphaInput.get();
        double priorBeta = priorBetaInput.get();

        double posteriorAlpha = priorAlpha + 0.5 * branchCount;
        double posteriorBeta = priorBeta + 0.5 * residualSumOfSquares;

        double oldConditionalLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        oldVariance,
                        posteriorAlpha,
                        posteriorBeta
                );

        double varianceDraw =
                GenerationTimeMathUtils.sampleInverseGamma(
                        posteriorAlpha,
                        posteriorBeta
                );

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

        clockResidualVariance.setValue(0, varianceDraw);

        clockResidualPriorInput.get().calculateLogP();
        clockResidualVariancePriorInput.get().calculateLogP();

        return logHastingsRatio;
    }

    private double residualSumOfSquares(
            Tree tree,
            RealParameter clockResiduals
    ) {

        double total = 0.0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            double residual = clockResiduals.getValue(nodeNr);

            if (Double.isNaN(residual) || Double.isInfinite(residual)) {
                return Double.POSITIVE_INFINITY;
            }

            total += residual * residual;
        }

        return total;
    }
}