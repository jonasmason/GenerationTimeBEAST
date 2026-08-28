package gentime;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Conjugate Gibbs update for the clock LEVEL: the clock intercept a and the common level of the
 * branch residuals r_b, drawn jointly.
 *
 * <p>This direction does not exist under the alpha=0 (fix_base_rate) timescale, where a is pinned at 0.
 * It appears the moment a is estimated. Write the per-branch level contribution
 *
 * <pre>
 *     c_b = a + r_b        so that     log mu_b = c_b + s * p_b
 * </pre>
 *
 * The sequence data constrain {c_b}. They say nothing whatsoever about how c_b is split between a and
 * r_b: shifting a by +k and every r_b by -k leaves every branch rate EXACTLY unchanged, on every branch,
 * root-adjacent ones included. The split is determined only by the two priors,
 * a ~ Normal(priorMean, priorSigma^2) and r_b ~ Normal(0, clockResidualVariance).
 *
 * <p>Left alone this is a slow direction, because the only operator that moves {r_b} is a BEAST
 * RealRandomWalkOperator, which updates ONE of the ~2n-2 residual dimensions per proposal; the common
 * level of r therefore diffuses at rate 1/B and drags the clockIntercept trace with it.
 *
 * <p>Conditional on {c_b} (which this move holds fixed) the full conditional of a is Gaussian:
 *
 * <pre>
 *     precision P = 1/priorSigma^2 + B/clockResidualVariance
 *     mean      M = ( priorMean/priorSigma^2 + sum_b c_b / clockResidualVariance ) / P
 *     a' ~ Normal(M, 1/P) ;   r_b' = c_b - a'
 * </pre>
 *
 * <p>The map (a, r) -&gt; (a', r') is a translation of r by (a - a'), so its Jacobian is 1, and the
 * sequence likelihood is invariant. Returning the usual Gibbs-as-Metropolis correction (old minus new
 * conditional log-density, as in {@link GenerationTimeVarianceGibbsOperator} and
 * {@link GenerationTimeHalfCauchyHyperGibbsOperator}) therefore gives acceptance 1 up to rounding.
 *
 * <p>priorMean and priorSigma MUST match the Normal prior placed on clockIntercept in the XML; they are
 * supplied here as plain numbers for the same reason the variance Gibbs operators take priorAlpha /
 * priorBeta rather than reading the prior object.
 *
 * <p>Do NOT attach this operator when clockIntercept is fixed (estimate="false"): with a pinned there is
 * no level to split, and the move would write to a constant.
 */
@Description("Joint conjugate Gibbs draw of the clock intercept and the common level of the branch residuals.")
public class GenerationTimeClockLevelGibbsOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root branches carry clock residuals.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockInterceptInput = new Input<>(
            "clockIntercept",
            "Clock intercept a; must be estimated.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockResidualsInput = new Input<>(
            "clockResiduals",
            "Branch-specific log-rate residuals indexed by child BEAST node number. Root entry is ignored.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockResidualVarianceInput = new Input<>(
            "clockResidualVariance",
            "Variance of the branch residuals; supplies the residual precision for the split.",
            Input.Validate.REQUIRED
    );

    public Input<Double> priorMeanInput = new Input<>(
            "priorMean",
            "Mean of the Normal prior on clockIntercept. Must match the XML prior.",
            0.0
    );

    public Input<Double> priorSigmaInput = new Input<>(
            "priorSigma",
            "Standard deviation of the Normal prior on clockIntercept. Must match the XML prior.",
            Math.sqrt(1000.0)
    );

    @Override
    public void initAndValidate() {

        final Tree tree = treeInput.get();

        if (clockInterceptInput.get().getDimension() != 1) {
            throw new IllegalArgumentException("clockIntercept must be a scalar RealParameter.");
        }
        if (clockResidualVarianceInput.get().getDimension() != 1) {
            throw new IllegalArgumentException("clockResidualVariance must be a scalar RealParameter.");
        }
        if (clockResidualsInput.get().getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "clockResiduals dimension must match tree node count. "
                            + "Residuals are indexed by child node number; the root entry is ignored."
            );
        }
        if (!(priorSigmaInput.get() > 0.0) || Double.isInfinite(priorSigmaInput.get())) {
            throw new IllegalArgumentException("priorSigma must be positive and finite.");
        }
    }

    @Override
    public double proposal() {

        final Tree tree = treeInput.get();
        final RealParameter clockIntercept = clockInterceptInput.get();
        final RealParameter clockResiduals = clockResidualsInput.get();

        final double residualVariance = clockResidualVarianceInput.get().getValue();
        if (!(residualVariance > 0.0) || Double.isInfinite(residualVariance)) {
            return Double.NEGATIVE_INFINITY;
        }

        final double priorMean = priorMeanInput.get();
        final double priorSigma = priorSigmaInput.get();
        final double priorVariance = priorSigma * priorSigma;

        final double oldIntercept = clockIntercept.getValue();

        // c_b = a + r_b, the per-branch level the sequences actually see. Held fixed by this move.
        final int nodeCount = tree.getNodeCount();
        double sumC = 0.0;
        int branchCount = 0;
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            final Node node = tree.getNode(nodeNr);
            if (node.isRoot()) {
                continue;
            }
            final double residual = clockResiduals.getValue(nodeNr);
            if (Double.isNaN(residual) || Double.isInfinite(residual)) {
                return Double.NEGATIVE_INFINITY;
            }
            sumC += oldIntercept + residual;
            branchCount++;
        }
        if (branchCount == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        final double posteriorPrecision =
                1.0 / priorVariance + branchCount / residualVariance;
        final double posteriorMean =
                (priorMean / priorVariance + sumC / residualVariance) / posteriorPrecision;
        final double posteriorSd = Math.sqrt(1.0 / posteriorPrecision);

        final double newIntercept = posteriorMean + posteriorSd * Randomizer.nextGaussian();
        if (Double.isNaN(newIntercept) || Double.isInfinite(newIntercept)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (newIntercept < clockIntercept.getLower() || newIntercept > clockIntercept.getUpper()) {
            return Double.NEGATIVE_INFINITY;
        }

        final double shift = newIntercept - oldIntercept;

        // r_b' = c_b - a' = r_b - shift, so every log mu_b = a + s*p_b + r_b is exactly unchanged.
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            final Node node = tree.getNode(nodeNr);
            if (node.isRoot()) {
                continue;
            }
            clockResiduals.setValue(nodeNr, clockResiduals.getValue(nodeNr) - shift);
        }

        clockIntercept.setValue(0, newIntercept);

        // Gibbs-as-Metropolis correction: old minus new conditional log-density. The map is a
        // translation (Jacobian 1) and the sequence likelihood is invariant, so acceptance is 1.
        return normalLogDensity(oldIntercept, posteriorMean, posteriorSd)
                - normalLogDensity(newIntercept, posteriorMean, posteriorSd);
    }

    private static double normalLogDensity(final double x, final double mean, final double sd) {
        final double z = (x - mean) / sd;
        return -0.5 * Math.log(2.0 * Math.PI) - Math.log(sd) - 0.5 * z * z;
    }
}
