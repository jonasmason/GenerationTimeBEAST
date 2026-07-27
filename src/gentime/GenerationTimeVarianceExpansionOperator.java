package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Parameter-expansion ("redundant multiplicative") operator for the Brownian field variance.
 *
 * <p>When {@code brownianVariance} approaches zero the latent generation-time field is squeezed
 * toward the root anchor and the (field, variance) pair forms a funnel. A single-coordinate
 * variance Gibbs step (see {@link GenerationTimeVarianceGibbsOperator}) then sticks near zero
 * and independent chains disagree, which is the poor mixing observed on low-signal replicates.
 *
 * <p>Following parameter expansion (C. H. Liu, Rubin &amp; Wu 1998; J. S. Liu &amp; Wu 1999;
 * Gelman, van Dyk et al. 2008) and matching the redundant-working-parameter scheme used in
 * MCMCglmm, this operator proposes ONE coordinated scale move along the funnel axis:
 *
 * <pre>
 *     alpha              ~ exp(Uniform(-w, w))          // a redundant working scale
 *     u_node (non-root)  -> alpha   * u_node            // every non-root latent state
 *     brownianVariance   -> alpha^2 * brownianVariance
 * </pre>
 *
 * <p>The Brownian field-prior EXPONENT is invariant under the move: the sufficient statistic
 * SS = u^T Q u scales as alpha^2 and so does the variance, so SS/V is unchanged. The move
 * therefore slides freely along exactly the direction the centred parameterisation cannot
 * traverse. The root latent state is held at its anchor (u_root = 0) as in
 * {@link GenerationTimeLevelRidgeOperator}, preserving the identifying convention and mapping
 * 0 -> 0 under scaling.
 *
 * <p>Acceptance uses the full posterior, which BEAST evaluates automatically: the trait
 * likelihood and Brownian field prior (via {@link GenerationTimeLikelihood} /
 * {@link GenerationTimeUnitHeightLikelihood}, including the field prior's alpha^-rank
 * normaliser), the sequence/clock likelihood (branch rates depend on the field), and the
 * inverse-gamma prior on {@code brownianVariance} (via {@link GenerationTimeInverseGammaPrior}).
 *
 * <p>The proposal in delta = log(alpha) is symmetric, so — exactly as in
 * {@code GenerationTimeLevelRidgeOperator} — the only Hastings contribution is the Jacobian of
 * the deterministic scaling: {@code rank} field values by alpha and the variance by alpha^2,
 * giving
 *
 * <pre>
 *     log Hastings ratio = (rank + 2) * delta
 * </pre>
 *
 * where {@code rank} is the number of scaled (non-root) latent states, i.e. the Brownian
 * field-prior rank (edge count = nodeCount - 1).
 *
 * <p>Use this ALONGSIDE the ordinary variance Gibbs (not instead of it): the Gibbs updates the
 * variance conditional on the field within the funnel, while this operator provides the
 * along-funnel "breathing" step that lets the pair escape zero.
 */
public class GenerationTimeVarianceExpansionOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root latent generation-time states are scaled.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent generation-time field indexed by BEAST node number.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> brownianVarianceInput = new Input<>(
            "brownianVariance",
            "Brownian field variance to expand.",
            Input.Validate.REQUIRED
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width for delta = log(alpha).",
            0.5
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter brownianVariance;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        brownianVariance = brownianVarianceInput.get();
        windowSize = windowSizeInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }
        if (brownianVariance.getDimension() != 1) {
            throw new IllegalArgumentException(
                    "brownianVariance must be a scalar RealParameter."
            );
        }
        if (windowSize <= 0.0 || Double.isNaN(windowSize) || Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException("windowSize must be positive and finite.");
        }
    }

    @Override
    public double proposal() {

        final double delta = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);
        final double alpha = Math.exp(delta);

        // brownianVariance -> alpha^2 * brownianVariance
        final double newVariance = brownianVariance.getValue() * Math.exp(2.0 * delta);
        if (newVariance <= 0.0 || Double.isNaN(newVariance) || Double.isInfinite(newVariance)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (newVariance < brownianVariance.getLower() || newVariance > brownianVariance.getUpper()) {
            return Double.NEGATIVE_INFINITY;
        }

        // every non-root latent state -> alpha * state; the root is held at its anchor (0 -> 0)
        int scaledCount = 0;
        final int nodeCount = tree.getNodeCount();
        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            if (tree.getNode(nodeNr).isRoot()) {
                continue;
            }
            generationTimes.setValue(nodeNr, generationTimes.getValue(nodeNr) * alpha);
            scaledCount++;
        }

        brownianVariance.setValue(0, newVariance);

        // symmetric proposal in delta; Jacobian = (rank field values * alpha) + (variance * alpha^2)
        return (scaledCount + 2.0) * delta;
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
        // Window is auto-tuned via optimize(); no manual suggestion needed.
        return "";
    }
}
