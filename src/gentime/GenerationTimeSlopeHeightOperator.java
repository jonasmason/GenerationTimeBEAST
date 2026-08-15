package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Slope-height decorrelating operator for the GT clock WITHOUT clock mean-centring
 * (meanCenterPredictor="false").
 *
 * Without centring the slope s is collinear with the base rate level s*xbar (xbar = across-branch mean of
 * the predictor 0.5(u_parent+u_child)), which the tree height absorbs. A plain random walk on clockSlope
 * shifts that level, the sequence data fight it, and the slope mixes poorly - which leaves the slope
 * weakly identified (spurious GT signal in null scenarios, inflated GT fraction, worse node ages).
 *
 * This operator proposes the slope together with a compensating tree rescale so the base level is held
 * fixed and only the identified contrast part of the slope changes:
 *
 *     s                -> s + eps
 *     internal heights -> height * exp(-eps * xbar)
 *
 * Then log mu_b -> log mu_b + eps*x_b and time_b -> time_b*exp(-eps*xbar), so the genetic distance
 * d_b = rate_b*time_b changes only by exp(eps*(x_b - xbar)) - the contrasts (the slope's signal), with the
 * base level cancelled. This reproduces, as an MCMC move, the slope decorrelation that mean-centring gave
 * in the likelihood, with no state-dependent term in the model (so it stays valid under an identified
 * intercept). Pair it with {@link GenerationTimeLevelScaleRidgeOperator}, which does the same for the
 * beta / field-level direction.
 *
 * xbar is computed from the current field only to design the move; it is not part of the target density,
 * so the honest no-centring posterior is preserved. Hastings ratio: the slope update is a translation
 * (Jacobian 1); scaling the nInternal internal heights by f = exp(-eps*xbar) contributes log-Jacobian
 * -nInternal*eps*xbar. The tree-prior change is evaluated by BEAST. Assumes contemporaneous tips.
 */
public class GenerationTimeSlopeHeightOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose internal node heights are rescaled to compensate the slope move.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent field vector; used to form the across-branch predictor mean xbar.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockSlopeInput = new Input<>(
            "clockSlope",
            "Clock slope s, proposed jointly with the compensating tree rescale.",
            Input.Validate.REQUIRED
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width for the slope step eps.",
            0.1
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter clockSlope;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        clockSlope = clockSlopeInput.get();
        windowSize = windowSizeInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (clockSlope.getDimension() != 1) {
            throw new IllegalArgumentException("clockSlope must be a scalar RealParameter.");
        }

        if (!(windowSize > 0.0) || Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException("windowSize must be positive and finite.");
        }
    }

    @Override
    public double proposal() {

        final double eps = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);
        final int nodeCount = tree.getNodeCount();

        // across-branch predictor mean xbar = mean over non-root branches of 0.5(u_parent + u_child)
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
        final double xbar = (count > 0) ? sum / count : 0.0;

        // s += eps
        clockSlope.setValue(0, clockSlope.getValue() + eps);

        // compensate the base level: every branch time *= exp(-eps*xbar)
        final double f = Math.exp(-eps * xbar);
        if (f <= 0.0 || Double.isNaN(f) || Double.isInfinite(f)) {
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

        // log Hastings ratio = log-Jacobian of scaling nInternal heights by f
        return -nInternal * eps * xbar;
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
