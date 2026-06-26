package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GenerationTimeClockResidualPrior extends Distribution {

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
            "Variance of branch-specific log-rate residuals.",
            Input.Validate.REQUIRED
    );

    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);

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
    }

    @Override
    public double calculateLogP() {

        Tree tree = treeInput.get();
        RealParameter clockResiduals = clockResidualsInput.get();

        double variance = clockResidualVarianceInput.get().getValue();

        if (variance <= 0.0 || Double.isNaN(variance) || Double.isInfinite(variance)) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }

        double logDensity = 0.0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            double residual = clockResiduals.getValue(nodeNr);

            if (Double.isNaN(residual) || Double.isInfinite(residual)) {
                logP = Double.NEGATIVE_INFINITY;
                return logP;
            }

            logDensity += -0.5 * (
                    LOG_TWO_PI
                            + Math.log(variance)
                            + residual * residual / variance
            );
        }

        logP = logDensity;
        return logP;
    }

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }

    @Override
    public List<String> getArguments() {
        ArrayList<String> arguments = new ArrayList<>();
        arguments.add(clockResidualsInput.get().getID());
        arguments.add(clockResidualVarianceInput.get().getID());
        return arguments;
    }

    @Override
    public List<String> getConditions() {
        return new ArrayList<>();
    }

    @Override
    public void sample(State state, Random random) {
        // Prior is evaluated only; sampling is handled by operators.
    }
}