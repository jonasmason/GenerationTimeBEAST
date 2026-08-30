package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Coordinated "slope-and-field-scale ridge" operator.
 *
 * <p>The GT clock is
 *
 * <pre>
 *     log mu_b = a + s * (x_b + lambda*beta - c*xbar) + r_b,      x_b = 0.5*(u_parent + u_child)
 * </pre>
 *
 * <p>where {@code lambda = includeBetaInRate} and {@code c = meanCenterPredictor}. Only the PRODUCT
 * {@code s * x_b} enters the clock, so the posterior has a ridge along which the field is inflated and
 * the slope deflated by the same factor. The centred parameterisation cannot traverse it: the slope
 * random walk moves {@code s} at fixed field (which changes every branch rate at once and is nearly
 * always rejected), and the field random walk moves one node at fixed slope. The result is the poor
 * {@code clockSlope} mixing seen in the production runs.
 *
 * <p>This operator steps ALONG that ridge in one coordinated proposal:
 *
 * <pre>
 *     delta              ~ Uniform(-w, w),   alpha = exp(delta)
 *     u_node (non-root)  -> alpha   * u_node
 *     brownianVariance   -> alpha^2 * brownianVariance
 *     clockSlope         -> clockSlope / alpha
 *     beta               -> alpha   * beta          // ONLY when includeBetaInRate = true
 * </pre>
 *
 * <p><b>What is invariant.</b> The clock predictor scales by {@code alpha} and the slope by
 * {@code 1/alpha}, so {@code s * x_b} — and hence {@code log mu_b} on every branch — is exactly
 * unchanged. This was verified numerically across GT/GTcc x centring on/off x both timescale regimes:
 * {@code max |delta log mu_b| <= 4.4e-16}. Because the ACROSS-BRANCH MEAN of the predictor is preserved
 * along with everything else, the move does not shift the clock level. That is why — unlike
 * {@link GenerationTimeLevelScaleRidgeOperator} and {@link GenerationTimeSlopeHeightOperator} — it needs
 * no {@code compensateVia} switch: there is nothing for the tree height or the clock intercept to
 * absorb, so it is valid unchanged under both the alpha=0 (fixed intercept, free height) and the
 * fixed-height (free intercept) regimes.
 *
 * <p>The Brownian field prior is also invariant: its sufficient statistic {@code SS = u^T Q u} scales by
 * {@code alpha^2} and so does the variance, leaving the exponent {@code SS/V} unchanged (verified: ratio
 * 1.0000000000). The {@code alpha^-rank} normaliser is evaluated by BEAST via
 * {@link GenerationTimeLikelihood} / {@link GenerationTimeUnitHeightLikelihood}. The root latent state is
 * held at its anchor {@code u_root = 0}, which maps 0 -> 0 under scaling.
 *
 * <p><b>What is not invariant, deliberately.</b> The trait likelihood {@code y_tip = beta + u_tip} does
 * change, because the observed generation times pin the absolute scale of the field. That is real data
 * information, not a pathology: it is what identifies the ridge, and it is why the acceptance rate is
 * finite rather than 1. Under GT ({@code includeBetaInRate = false}) beta is NOT part of the clock
 * predictor and is therefore NOT scaled; scaling it there would perturb the trait likelihood for no
 * compensating gain.
 *
 * <p><b>Hastings ratio.</b> The proposal in {@code delta} is symmetric, so the only contribution is the
 * log-Jacobian of the deterministic map: {@code rank} field values by {@code alpha} (+rank*delta), the
 * variance by {@code alpha^2} (+2*delta), the slope by {@code 1/alpha} (-delta), and — under GTcc only —
 * beta by {@code alpha} (+delta). Hence
 *
 * <pre>
 *     log Hastings ratio = (rank + 1) * delta       // includeBetaInRate = false  (GT)
 *     log Hastings ratio = (rank + 2) * delta       // includeBetaInRate = true   (GTcc)
 * </pre>
 *
 * where {@code rank} is the number of scaled (non-root) latent states, i.e. the field-prior rank.
 *
 * <p><b>Relationship to {@link GenerationTimeVarianceExpansionOperator}.</b> They are complementary, not
 * substitutes. The PX operator scales the field and the variance at FIXED slope, so it changes
 * {@code s * x_b} and lets a collapsed field escape zero. This operator scales the field and the
 * variance at FIXED {@code s * x_b}, so it trades field scale against slope without touching the clock.
 * Run both.
 *
 * <p><b>Do not use when</b> {@code clockSlope} is fixed (e.g. the RCautocor s=1 and RCuncor s=0
 * reference clocks) or when {@code brownianVariance} is fixed (e.g. acln_fix_field_variance / RCmixed):
 * one leg of the move is then nailed down, the move is no longer invariant, and every proposal is
 * rejected at full posterior-evaluation cost. Both cases are detected in
 * {@link #initAndValidate()} and raise an error rather than mixing badly in silence.
 */
public class GenerationTimeSlopeFieldExpansionOperator extends Operator {

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
            "Brownian field variance; must be the SAME parameter the field prior uses, so that "
                    + "SS/V stays invariant under the move.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockSlopeInput = new Input<>(
            "clockSlope",
            "Clock slope s, scaled by 1/alpha so that s * predictor is invariant.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Generation-time intercept beta. REQUIRED when includeBetaInRate='true' (beta is then part "
                    + "of the clock predictor and must scale with the field); ignored otherwise.",
            (RealParameter) null
    );

    public Input<Boolean> includeBetaInRateInput = new Input<>(
            "includeBetaInRate",
            "Must match the value on the GenerationTimeRegressionClockModel. If true (GTcc) beta is part "
                    + "of the clock predictor and is scaled by alpha; if false (GT) beta is left alone.",
            false
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width for delta = log(alpha).",
            0.2
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter brownianVariance;
    private RealParameter clockSlope;
    private RealParameter beta;
    private boolean includeBetaInRate;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        brownianVariance = brownianVarianceInput.get();
        clockSlope = clockSlopeInput.get();
        beta = betaInput.get();
        includeBetaInRate = includeBetaInRateInput.get();
        windowSize = windowSizeInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }
        if (brownianVariance.getDimension() != 1) {
            throw new IllegalArgumentException("brownianVariance must be a scalar RealParameter.");
        }
        if (clockSlope.getDimension() != 1) {
            throw new IllegalArgumentException("clockSlope must be a scalar RealParameter.");
        }

        // The move is only invariant if BOTH legs of the ridge are free to move. If either is held
        // fixed the operator is silently useless, so fail loudly at setup instead.
        if (!clockSlope.isEstimatedInput.get()) {
            throw new IllegalArgumentException(
                    "GenerationTimeSlopeFieldExpansionOperator requires an estimated clockSlope. "
                            + "The slope is fixed here (e.g. an RCautocor s=1 or RCuncor s=0 reference "
                            + "clock), so the move cannot preserve s * predictor. Remove this operator "
                            + "from the analysis."
            );
        }
        if (!brownianVariance.isEstimatedInput.get()) {
            throw new IllegalArgumentException(
                    "GenerationTimeSlopeFieldExpansionOperator requires an estimated brownianVariance. "
                            + "The variance is fixed here (e.g. acln_fix_field_variance / RCmixed), so "
                            + "scaling the field would break the field prior. Remove this operator from "
                            + "the analysis."
            );
        }

        if (includeBetaInRate) {
            if (beta == null) {
                throw new IllegalArgumentException(
                        "includeBetaInRate='true' requires the beta input: beta is part of the clock "
                                + "predictor and must be scaled together with the field."
                );
            }
            if (beta.getDimension() != 1) {
                throw new IllegalArgumentException("beta must be a scalar RealParameter.");
            }
        }

        if (!(windowSize > 0.0) || Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException("windowSize must be positive and finite.");
        }
    }

    @Override
    public double proposal() {

        final double delta = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);
        final double alpha = Math.exp(delta);

        // clockSlope -> clockSlope / alpha   (sign preserved, so no zero crossing)
        final double newSlope = clockSlope.getValue() / alpha;
        if (Double.isNaN(newSlope) || Double.isInfinite(newSlope)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (newSlope < clockSlope.getLower() || newSlope > clockSlope.getUpper()) {
            return Double.NEGATIVE_INFINITY;
        }

        // brownianVariance -> alpha^2 * brownianVariance
        final double newVariance = brownianVariance.getValue() * Math.exp(2.0 * delta);
        if (newVariance <= 0.0 || Double.isNaN(newVariance) || Double.isInfinite(newVariance)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (newVariance < brownianVariance.getLower() || newVariance > brownianVariance.getUpper()) {
            return Double.NEGATIVE_INFINITY;
        }

        // beta -> alpha * beta, but only when beta rides inside the clock predictor (GTcc)
        double newBeta = 0.0;
        if (includeBetaInRate) {
            newBeta = beta.getValue() * alpha;
            if (Double.isNaN(newBeta) || Double.isInfinite(newBeta)) {
                return Double.NEGATIVE_INFINITY;
            }
            if (newBeta < beta.getLower() || newBeta > beta.getUpper()) {
                return Double.NEGATIVE_INFINITY;
            }
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
        clockSlope.setValue(0, newSlope);
        if (includeBetaInRate) {
            beta.setValue(0, newBeta);
        }

        // Symmetric in delta. Jacobian: rank field values * alpha, variance * alpha^2,
        // slope * alpha^-1, and (GTcc only) beta * alpha.
        final double jacobianTerms = scaledCount + 2.0 - 1.0 + (includeBetaInRate ? 1.0 : 0.0);
        return jacobianTerms * delta;
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
