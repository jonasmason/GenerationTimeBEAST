package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Coordinated "level-and-scale ridge" operator for the beta-decoupled GT clock, for use WITHOUT clock
 * mean-centring (meanCenterPredictor="false").
 *
 * The soft (sloppy) direction of the model couples the field level, beta, the clock slope and the tree
 * scale. This operator steps along it in one coordinated proposal:
 *
 *     beta              -> beta + delta
 *     u_node (non-root) -> u_node - delta                  (every non-root latent state)
 *     internal heights  -> height * exp(clockSlope*delta)  (uniform tree rescale)
 *
 * Invariances (why the move is nearly free):
 *   - Trait likelihood y_tip = beta + u_tip: invariant, since (beta+delta) + (u_tip-delta) = beta+u_tip.
 *   - Brownian increments u_child - u_parent: invariant off the root (only root-adjacent increments
 *     change, because the root is held at its anchor u_root = 0).
 *   - Clock genetic distances d_b = rate_b * time_b: for a branch not touching the root the predictor
 *     0.5(u_pa+u_ch) shifts by -delta, so rate_b -> rate_b*exp(-s*delta); rescaling every branch time by
 *     exp(s*delta) (via the tree rescale) restores d_b. The (~2) root-adjacent branches shift by
 *     -delta/2 and are only half-compensated, giving the same weak restoring force the level-ridge
 *     operator already leaves.
 *   - Field prior (unitHeight): scale-invariant, so the tree rescale does not change it.
 *
 * Unlike {@link GenerationTimeLevelRidgeOperator} (which needs meanCenterPredictor="true" for the clock
 * to be invariant to a field shift), this operator adds the tree rescale so the clock is invariant
 * WITHOUT centring. It restores the level/base-rate mixing that dynamic centring used to provide, with
 * no state-dependent term in the likelihood - so it stays valid when the clock intercept is identified.
 *
 * Hastings ratio: beta and u updates are translations (Jacobian 1); the rescale multiplies the
 * nInternal internal node heights by f = exp(clockSlope*delta), contributing log-Jacobian
 * nInternal*clockSlope*delta. The delta proposal is symmetric (no proposal-ratio term). The Yule/tree
 * prior change is NOT returned here: it is part of the posterior and is evaluated by BEAST when the tree
 * changes (and vanishes under a scale-free tree prior).
 *
 * Assumes contemporaneous tips (tip heights = 0); internal-only rescaling then preserves node ordering.
 */
public class GenerationTimeLevelScaleRidgeOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root field states are shifted and whose internal heights are rescaled.",
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

    public Input<RealParameter> clockSlopeInput = new Input<>(
            "clockSlope",
            "Clock slope s; the tree is rescaled by exp(s*delta) so clock genetic distances stay invariant.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockInterceptInput = new Input<>(
            "clockIntercept",
            "Clock intercept a. REQUIRED when compensateVia is 'clockIntercept'; ignored otherwise.",
            (RealParameter) null
    );

    public Input<String> compensateViaInput = new Input<>(
            "compensateVia",
            "How the level move holds the clock genetic distances fixed:\n"
                    + "  'treeHeight'    (default) - rescale internal node heights by exp(s*delta). Valid only "
                    + "when the tree scale is free to absorb the level, i.e. under the alpha=0 "
                    + "(fix_base_rate) timescale with the clock intercept fixed. Historical behaviour.\n"
                    + "  'clockIntercept' - set a += s*delta and leave the tree alone. Use whenever node ages "
                    + "are pinned and the clock intercept is estimated. Hastings ratio is 0.\n"
                    + "  'none'          - no compensation. Correct ONLY when beta is carried into the clock "
                    + "predictor (includeBetaInRate='true'), because the predictor 0.5(u_p+u_c)+beta is then "
                    + "already invariant under this move. Reproduces GenerationTimeLevelRidgeOperator exactly "
                    + "and is regime-independent.",
            "treeHeight"
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width for the level shift delta.",
            0.1
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter beta;
    private RealParameter clockSlope;
    private RealParameter clockIntercept;
    private String compensateVia;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        beta = betaInput.get();
        clockSlope = clockSlopeInput.get();
        clockIntercept = clockInterceptInput.get();
        compensateVia = compensateViaInput.get();
        windowSize = windowSizeInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (beta.getDimension() != 1) {
            throw new IllegalArgumentException("beta must be a scalar RealParameter.");
        }

        if (clockSlope.getDimension() != 1) {
            throw new IllegalArgumentException("clockSlope must be a scalar RealParameter.");
        }

        if (!compensateVia.equals("treeHeight")
                && !compensateVia.equals("clockIntercept")
                && !compensateVia.equals("none")) {
            throw new IllegalArgumentException(
                    "compensateVia must be 'treeHeight', 'clockIntercept' or 'none'."
            );
        }

        if (compensateVia.equals("clockIntercept")) {
            if (clockIntercept == null) {
                throw new IllegalArgumentException(
                        "compensateVia='clockIntercept' requires the clockIntercept input."
                );
            }
            if (clockIntercept.getDimension() != 1) {
                throw new IllegalArgumentException("clockIntercept must be a scalar RealParameter.");
            }
        }

        if (!(windowSize > 0.0) || Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException("windowSize must be positive and finite.");
        }
    }

    @Override
    public double proposal() {

        final double delta = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);
        final double s = clockSlope.getValue();

        // beta += delta
        beta.setValue(0, beta.getValue() + delta);

        // every non-root latent state -= delta; the root is held at its anchor u_root = 0
        final int nodeCount = tree.getNodeCount();
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            if (tree.getNode(nodeNr).isRoot()) {
                continue;
            }
            generationTimes.setValue(nodeNr, generationTimes.getValue(nodeNr) - delta);
        }

        // Off-root the predictor shifted by -delta, so log mu_b moved by -s*delta. Restore the level
        // through the clock intercept instead of the tree: a += s*delta. Genetic distances are then
        // invariant on every branch not touching the root, exactly as in the tree-rescaling variant, and
        // the (typically 2) root-adjacent branches keep the same +s*delta/2 residue. Pure translation,
        // so there is no Jacobian.
        if (compensateVia.equals("clockIntercept")) {
            clockIntercept.setValue(0, clockIntercept.getValue() + s * delta);
            return 0.0;
        }

        // No compensation: correct only when beta is carried into the predictor, where the move is
        // already rate-invariant off-root. Equivalent to GenerationTimeLevelRidgeOperator.
        if (compensateVia.equals("none")) {
            return 0.0;
        }

        // rescale the tree so clock distances rate*time stay invariant: every branch time *= exp(s*delta)
        final double f = Math.exp(s * delta);
        if (f <= 0.0 || Double.isNaN(f) || Double.isInfinite(f)) {
            return Double.NEGATIVE_INFINITY;
        }
        int nInternal = 0;
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            final Node node = tree.getNode(nodeNr);
            if (node.isLeaf()) {
                continue;   // contemporaneous tips stay at height 0
            }
            node.setHeight(node.getHeight() * f);
            nInternal++;
        }

        // Only internal heights were scaled: safe when every tip sits at height 0, not when tips are dated.
        // Reject rather than leave a negative branch length in the state.
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            final Node guard = tree.getNode(nodeNr);
            if (guard.isRoot()) {
                continue;
            }
            if (guard.getParent().getHeight() < guard.getHeight()) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        // log Hastings ratio = log-Jacobian of scaling nInternal heights by f; translations contribute 0.
        return nInternal * s * delta;
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
        windowSize = Math.exp(Math.log(windowSize) + calcDelta(logAlpha));
    }

    @Override
    public String getPerformanceSuggestion() {
        return "";
    }
}
