package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Coordinated "level ridge" operator for the beta-decoupled, clock-linked GT model.
 *
 * The model has a soft (sloppy) direction along which the data are nearly
 * uninformative. This operator proposes a single coordinated step along it:
 *
 *     beta              -> beta + delta
 *     u_node (non-root) -> u_node - delta            (every non-root node)
 *
 * Under this move:
 *   - the trait likelihood y_tip = beta + u_tip is invariant, because every tip is
 *     a non-root node and beta + u_tip is unchanged;
 *   - the Brownian increments (u_child - u_parent) are invariant for every branch
 *     that does not touch the root (only the root-adjacent increments change,
 *     because the root is deliberately held at its anchor and not shifted);
 *   - the clock genetic distances are invariant when the clock predictor is
 *     mean-centred, because a uniform field shift moves the predictor and its
 *     across-branch mean together, leaving (u_mid - mean) unchanged; no base-rate
 *     compensation is therefore needed.
 *
 * So the proposal is almost free: only the (typically two) root-adjacent branches
 * and their Brownian increments provide a weak restoring force. Independent
 * operators on beta and on individual u_node cannot traverse this direction,
 * because each single-coordinate step is fought by the trait and sequence data;
 * this operator steps along it in one coordinated proposal, which is what is
 * needed to mix the weakly-identified level / base-rate direction.
 *
 * The root latent state is intentionally NOT shifted. Holding the root at its
 * anchor (u_root = 0) is the identifying convention that separates beta (the
 * level) from the latent field (the deviations). Freeing the root would make beta
 * and the field level fully non-identified, so the operator preserves it.
 *
 * The proposal in delta is symmetric and involves no transformed (multiplicative)
 * variable, so there is no Jacobian; proposal() returns log Hastings ratio = 0.
 *
 * Intended for use with includeBetaInRate="false" (beta decoupled from the clock)
 * and meanCenterPredictor="true" (so the clock is invariant to the field shift).
 */
public class GenerationTimeLevelRidgeOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root latent generation-time states are shifted.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent generation-time state vector indexed by BEAST node number.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Generation-time intercept beta (the level parameter).",
            Input.Validate.REQUIRED
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width for the level shift delta.",
            0.1
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter beta;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        beta = betaInput.get();
        windowSize = windowSizeInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (beta.getDimension() != 1) {
            throw new IllegalArgumentException("beta must be a scalar RealParameter.");
        }

        if (windowSize <= 0.0 || Double.isNaN(windowSize) || Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException("windowSize must be positive and finite.");
        }
    }

    @Override
    public double proposal() {

        final double delta = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);

        // beta += delta
        beta.setValue(0, beta.getValue() + delta);

        // every non-root latent state -= delta; the root is held at its anchor
        final int nodeCount = tree.getNodeCount();
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            if (tree.getNode(nodeNr).isRoot()) {
                continue;
            }
            generationTimes.setValue(
                    nodeNr,
                    generationTimes.getValue(nodeNr) - delta
            );
        }

        // symmetric proposal, no transformed variable, so no Jacobian.
        return 0.0;
    }

    // ---- adaptive tuning of the window (targets ~0.234 acceptance) --------------

    @Override
    public double getCoercableParameterValue() {
        return windowSize;
    }

    @Override
    public void setCoercableParameterValue(final double value) {
        windowSize = value;
    }

    @Override
    public void optimize(final double logAlpha) {
        double newWindow = Math.log(windowSize) + calcDelta(logAlpha);
        windowSize = Math.exp(newWindow);
    }

    @Override
    public String getPerformanceSuggestion() {
        // Window is auto-tuned via optimize(); no manual suggestion needed.
        return "";
    }
}
