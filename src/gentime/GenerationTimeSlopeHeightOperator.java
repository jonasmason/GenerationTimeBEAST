package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Slope-height decorrelating operator for the GT clock.
 *
 * The slope s is collinear with the base rate level s*pbar, where pbar is the across-branch mean of the
 * clock predictor. Under the alpha=0 timescale the tree height absorbs that level. A plain random walk on
 * clockSlope shifts the level, the sequence data fight it, and the slope mixes poorly - which leaves the
 * slope weakly identified (spurious GT signal in null scenarios, inflated GT fraction, worse node ages).
 *
 * This operator proposes the slope together with a compensating tree rescale so the base level is held
 * fixed and only the identified contrast part of the slope changes:
 *
 *     s                -> s + eps
 *     internal heights -> height * exp(-eps * pbar)
 *
 * Then log mu_b -> log mu_b + eps*p_b and time_b -> time_b*exp(-eps*pbar), so the genetic distance
 * d_b = rate_b*time_b changes only by exp(eps*(p_b - pbar)) - the contrasts (the slope's signal), with the
 * base level cancelled. On an ultrametric tree with tips at height 0, EVERY branch time scales by the same
 * factor, so this cancellation is exact on every branch, root-adjacent ones included. This reproduces, as
 * an MCMC move, the slope decorrelation that mean-centring gave in the likelihood, with no state-dependent
 * term in the model (so it stays valid under an identified intercept). Pair it with
 * {@link GenerationTimeLevelScaleRidgeOperator} (beta decoupled) or {@link GenerationTimeLevelRidgeOperator}
 * (beta carried), which handle the beta / field-level direction.
 *
 * THE PREDICTOR MEAN MUST MATCH THE CLOCK MODEL. {@link GenerationTimeRegressionClockModel} forms the
 * branch predictor as
 *
 *     p_b = 0.5*(u_parent + u_child) + (includeBetaInRate ? beta : 0)   [ - pbar, if meanCenterPredictor ]
 *
 * so this operator takes the same two switches and must be configured with the same values:
 *
 *   - includeBetaInRate="true"  (the carry-beta / GTcc parameterisation): pbar = xbar + beta. Earlier
 *     versions of this class computed pbar from the latent field alone and omitted beta, which left a
 *     level leakage of exactly -eps*beta per proposal. That was harmless only because the carry-beta
 *     setup mean-centres the trait data so beta ~ 0; it is not harmless in general, and it is not
 *     harmless once the clock intercept is estimated.
 *   - meanCenterPredictor="true": the clock model already subtracts the predictor mean each evaluation,
 *     so the slope no longer tilts the rate level and NO compensation is wanted. The effective predictor
 *     mean is then zero, the tree is left alone, and this operator degenerates to a plain symmetric
 *     random walk on the slope. That is the correct behaviour, not a special case.
 *
 * Defaults (includeBetaInRate=false, meanCenterPredictor=false, beta absent) reproduce the previous
 * behaviour exactly, so XMLs written before these inputs existed are unaffected.
 *
 * pbar is computed from the current state only to design the move; it is not part of the target density,
 * so the honest no-centring posterior is preserved. Because the move changes neither the latent field nor
 * beta, the reverse proposal is designed with the same pbar, so the map is properly reversible.
 *
 * Hastings ratio: the slope update is a translation (Jacobian 1); scaling the nInternal internal heights
 * by f = exp(-eps*pbar) contributes log-Jacobian -nInternal*eps*pbar. The tree-prior change is evaluated
 * by BEAST.
 *
 * Assumes contemporaneous tips: only internal heights are scaled, which preserves node ordering when all
 * tips sit at height 0. With dated tips (tip dating / fossilised birth-death) that is NOT true, so the
 * proposal is checked for ordering violations and rejected rather than writing a negative branch length
 * into the state.
 */
public class GenerationTimeSlopeHeightOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose internal node heights are rescaled to compensate the slope move.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent field vector; used to form the across-branch predictor mean.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockSlopeInput = new Input<>(
            "clockSlope",
            "Clock slope s, proposed jointly with the compensating tree rescale.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Generation-time intercept beta. REQUIRED when includeBetaInRate is true, because the "
                    + "predictor mean is then xbar + beta. Ignored otherwise.",
            (RealParameter) null
    );

    public Input<Boolean> includeBetaInRateInput = new Input<>(
            "includeBetaInRate",
            "Must match the same-named input on GenerationTimeRegressionClockModel. If true, beta is part "
                    + "of the branch predictor and therefore part of the predictor mean this operator "
                    + "compensates. Default false, which reproduces the pre-existing behaviour.",
            false
    );

    public Input<Boolean> meanCenterPredictorInput = new Input<>(
            "meanCenterPredictor",
            "Must match the same-named input on GenerationTimeRegressionClockModel. If true, the clock "
                    + "already centres the predictor, the slope does not move the rate level, and no tree "
                    + "compensation is applied (the move becomes a plain random walk on the slope). "
                    + "Default false.",
            false
    );

    public Input<RealParameter> clockInterceptInput = new Input<>(
            "clockIntercept",
            "Clock intercept a. REQUIRED when compensateVia is 'clockIntercept'; ignored otherwise.",
            (RealParameter) null
    );

    public Input<String> compensateViaInput = new Input<>(
            "compensateVia",
            "How the slope move holds the overall rate level fixed:\n"
                    + "  'treeHeight'    (default) - rescale internal node heights by exp(-eps*pbar). Valid "
                    + "only when the tree scale is free to absorb the level, i.e. under the alpha=0 "
                    + "(fix_base_rate) timescale with the clock intercept fixed.\n"
                    + "  'clockIntercept' - set a -= eps*pbar and leave the tree alone. This is the move to "
                    + "use whenever the node ages are pinned (fixed root height, or fossil calibrations) and "
                    + "the clock intercept is estimated. Hastings ratio is 0.\n"
                    + "  'none'          - no compensation; degenerates to a plain random walk on the slope.",
            "treeHeight"
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width for the slope step eps.",
            0.1
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter clockSlope;
    private RealParameter beta;
    private RealParameter clockIntercept;
    private String compensateVia;
    private boolean includeBetaInRate;
    private boolean meanCenterPredictor;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        clockSlope = clockSlopeInput.get();
        beta = betaInput.get();
        clockIntercept = clockInterceptInput.get();
        compensateVia = compensateViaInput.get();
        includeBetaInRate = includeBetaInRateInput.get();
        meanCenterPredictor = meanCenterPredictorInput.get();
        windowSize = windowSizeInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (clockSlope.getDimension() != 1) {
            throw new IllegalArgumentException("clockSlope must be a scalar RealParameter.");
        }

        if (includeBetaInRate && beta == null) {
            throw new IllegalArgumentException(
                    "includeBetaInRate is true, so beta must be supplied: the clock predictor mean is "
                            + "xbar + beta, and compensating with xbar alone leaves a level leakage of "
                            + "-eps*beta per proposal."
            );
        }

        if (beta != null && beta.getDimension() != 1) {
            throw new IllegalArgumentException("beta must be a scalar RealParameter.");
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

        final double eps = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);
        final int nodeCount = tree.getNodeCount();

        // s += eps
        clockSlope.setValue(0, clockSlope.getValue() + eps);

        // When the clock mean-centres the predictor, the effective predictor mean is zero: the slope does
        // not move the rate level, so there is nothing to compensate and nothing else is touched.
        if (meanCenterPredictor || compensateVia.equals("none")) {
            return 0.0;
        }

        // Across-branch predictor mean, formed exactly as GenerationTimeRegressionClockModel forms the
        // predictor: pbar = mean over non-root branches of 0.5*(u_parent + u_child), plus beta when beta
        // is carried into the rate (beta is constant across branches, so it adds directly to the mean).
        double sum = 0.0;
        int count = 0;
        for (int nr = 0; nr < nodeCount; nr++) {
            final Node ch = tree.getNode(nr);
            if (ch.isRoot()) {
                continue;
            }
            final int paNr = ch.getParent().getNr();
            sum += 0.5 * (generationTimes.getValue(paNr) + generationTimes.getValue(nr));
            count++;
        }
        double pbar = (count > 0) ? sum / count : 0.0;
        if (includeBetaInRate) {
            pbar += beta.getValue();
        }

        // Compensate the base level through the clock intercept: log mu_b gains eps*p_b, so subtracting
        // eps*pbar from a leaves the level fixed and moves only the identified contrasts eps*(p_b - pbar).
        // The tree is not touched, so there is no Jacobian and no tree-prior or calibration cost.
        if (compensateVia.equals("clockIntercept")) {
            clockIntercept.setValue(0, clockIntercept.getValue() - eps * pbar);
            return 0.0;
        }

        // compensate the base level: every branch time *= exp(-eps*pbar)
        final double f = Math.exp(-eps * pbar);
        if (!(f > 0.0) || Double.isInfinite(f)) {
            return Double.NEGATIVE_INFINITY;
        }
        int nInternal = 0;
        for (int nr = 0; nr < nodeCount; nr++) {
            final Node node = tree.getNode(nr);
            if (node.isLeaf()) {
                continue;
            }
            node.setHeight(node.getHeight() * f);
            nInternal++;
        }

        // Only internal heights were scaled. That preserves node ordering when every tip sits at height 0,
        // but not when tips are dated. Reject rather than leave a negative branch length in the state
        // (BEAST's own Node.scale throws in this situation; rejecting is the safe equivalent here).
        for (int nr = 0; nr < nodeCount; nr++) {
            final Node node = tree.getNode(nr);
            if (node.isRoot()) {
                continue;
            }
            if (node.getParent().getHeight() < node.getHeight()) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        // log Hastings ratio = log-Jacobian of scaling nInternal heights by f
        return -nInternal * eps * pbar;
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
