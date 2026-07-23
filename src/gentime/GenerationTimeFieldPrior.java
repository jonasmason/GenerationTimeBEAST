package gentime;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;

/**
 * Field-only Brownian (GMRF-over-tree) prior on the latent field {@code generationTimes},
 * used to turn that field into an AUTOCORRELATED log-rate field for an autocorrelated
 * relaxed clock.
 *
 * <p>This is the Brownian-latent term of {@link GenerationTimeLikelihood} lifted out on its
 * own: it drops the tip-observation (trait-data) term and the beta prior, so the field is no
 * longer anchored to observed generation times. It only imposes the tree-structured Brownian
 * increment density
 *
 * <pre>
 *   log p(u | v, tree) = -0.5 (n-1) log(2*pi v)
 *                        - 0.5 * sum_e log(length_e)
 *                        - 0.5 * ( sum_e (u_child - u_parent)^2 / length_e ) / v
 * </pre>
 *
 * which is exactly {@code brownianLatentLogLikelihood} / {@code edgeIncrementQuadraticForm}
 * in {@link GenerationTimeLikelihood}. Neighbouring branches therefore have correlated field
 * values: this is the autocorrelation that distinguishes the model from the i.i.d.
 * {@link GenerationTimeClockResidualPrior}.
 *
 * <p>When {@link #generationTimesInput} is used as the clock predictor with
 * {@code clockSlope = 1}, {@code beta} dropped and {@code clockResiduals = 0}, the per-branch
 * log-rate becomes {@code clockIntercept + u_branch}, i.e. a Thorne&ndash;Kishino-style
 * autocorrelated lognormal relaxed clock. {@code brownianVariance} (v) is the amplitude of the
 * rate autocorrelation.
 *
 * <p>The Brownian process constrains only increments, so the overall level of the field is a
 * flat direction. It is pinned exactly as in {@link GenerationTimeLikelihood}: the root value
 * is required to be zero. Pair this prior with a field operator that never moves the root
 * (see {@code GenerationTimeFieldRandomWalkOperator}).
 *
 * <p>{@link #unitHeightInput}: when true the density is evaluated with effective variance
 * {@code v = brownianVariance / H} (H = root height), mirroring
 * {@link GenerationTimeUnitHeightLikelihood}. Use it under the alpha=0 (fix_base_rate)
 * timescale, where the tree height is estimated in substitution units, so the amplitude
 * parameter stays scale-invariant and does not couple to the tree height.
 */
@Description("Field-only Brownian GMRF prior on the latent field, giving an autocorrelated log-rate clock field.")
public class GenerationTimeFieldPrior extends Distribution {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree over whose branches the latent field follows Brownian motion.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent field vector indexed by BEAST tree node number. Root is fixed to zero.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> brownianVarianceInput = new Input<>(
            "brownianVariance",
            "Brownian variance parameter (amplitude of the field / rate autocorrelation).",
            Input.Validate.REQUIRED
    );

    public Input<Boolean> unitHeightInput = new Input<>(
            "unitHeight",
            "If true, evaluate the density on a unit-height tree (effective variance = "
                    + "brownianVariance / rootHeight), so the amplitude is scale-invariant under "
                    + "an estimated tree height (fix_base_rate / alpha=0). Default false.",
            false
    );

    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);

    protected boolean unitHeight;

    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();
        RealParameter brownianVariance = brownianVarianceInput.get();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must equal tree node count. " +
                    "Expected " + tree.getNodeCount() +
                    " but found " + generationTimes.getDimension()
            );
        }

        if (brownianVariance.getDimension() != 1) {
            throw new IllegalArgumentException("brownianVariance must be a scalar RealParameter.");
        }

        unitHeight = unitHeightInput.get();
    }

    @Override
    public double calculateLogP() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        double brownianVariance = brownianVarianceInput.get().getValue();

        if (brownianVariance <= 0.0 ||
                Double.isNaN(brownianVariance) ||
                Double.isInfinite(brownianVariance)) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }

        // Pin the flat level direction: the root value is the reference and must be zero.
        int currentRootNodeNr = tree.getRoot().getNr();
        if (Math.abs(generationTimes.getValue(currentRootNodeNr)) > 1e-10) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }

        double effectiveVariance = brownianVariance;

        if (unitHeight) {
            double h = tree.getRoot().getHeight();
            if (h <= 0.0 || Double.isNaN(h) || Double.isInfinite(h)) {
                logP = Double.NEGATIVE_INFINITY;
                return logP;
            }
            // v = s2 / H  ->  raw density with v == unit-height density with s2
            effectiveVariance = brownianVariance / h;
        }

        logP = brownianLatentLogLikelihood(tree, generationTimes, effectiveVariance);

        return logP;
    }

    protected double brownianLatentLogLikelihood(
            Tree tree,
            RealParameter generationTimes,
            double brownianVariance
    ) {

        double quadraticForm = edgeIncrementQuadraticForm(tree, generationTimes);

        if (Double.isInfinite(quadraticForm) || Double.isNaN(quadraticForm)) {
            return Double.NEGATIVE_INFINITY;
        }

        double logBranchLengthSum = 0.0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            double branchLength = node.getLength();

            if (branchLength <= 0.0 ||
                    Double.isNaN(branchLength) ||
                    Double.isInfinite(branchLength)) {
                return Double.NEGATIVE_INFINITY;
            }

            logBranchLengthSum += Math.log(branchLength);
        }

        int brownianRank = tree.getNodeCount() - 1;

        // full log-density of Brownian process:
        //   log p(u | v, tree) = -0.5(n-1)log(2pi v) - 0.5 * sum log(length_e) - 0.5 * q / v
        return -0.5 * brownianRank * (LOG_TWO_PI + Math.log(brownianVariance))
                -0.5 * logBranchLengthSum
                -0.5 * quadraticForm / brownianVariance;
    }

    protected double edgeIncrementQuadraticForm(
            Tree tree,
            RealParameter generationTimes
    ) {

        // Q = (branch-length)^{-1} on each edge, so the quadratic form decomposes edge-by-edge
        // as sum_e (u_child - u_parent)^2 / length_e.
        double total = 0.0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            Node parent = node.getParent();

            double branchLength = node.getLength();

            if (branchLength <= 0.0 ||
                    Double.isNaN(branchLength) ||
                    Double.isInfinite(branchLength)) {
                return Double.POSITIVE_INFINITY;
            }

            double childValue = generationTimes.getValue(node.getNr());
            double parentValue = generationTimes.getValue(parent.getNr());

            double difference = childValue - parentValue;

            total += difference * difference / branchLength;
        }

        return total;
    }

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }

    @Override
    public List<String> getArguments() {
        ArrayList<String> arguments = new ArrayList<>();
        arguments.add(generationTimesInput.get().getID());
        arguments.add(brownianVarianceInput.get().getID());
        return arguments;
    }

    @Override
    public List<String> getConditions() {
        ArrayList<String> conditions = new ArrayList<>();
        conditions.add(treeInput.get().getID());
        return conditions;
    }

    @Override
    public void sample(State state, Random random) {
        // Prior is evaluated only; sampling is handled by operators.
    }
}
