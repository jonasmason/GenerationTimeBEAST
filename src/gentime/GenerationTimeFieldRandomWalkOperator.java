package gentime;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * Signed random-walk operator on the latent field {@code generationTimes}, for use when the
 * field is an autocorrelated log-rate field (paired with {@link GenerationTimeFieldPrior}).
 *
 * <p>Differences from {@link GenerationTimeRandomWalkOperator} (which was written for the
 * positive, tip-anchored generation-time trait model):
 * <ul>
 *   <li>Moves ALL non-root nodes &mdash; tips and internal nodes &mdash; because in an
 *       autocorrelated clock the tip field values are latent too (no trait data anchors them).</li>
 *   <li>Does NOT reject non-positive proposals: log-rate deviations are signed and centred at
 *       zero, so negative field values are valid.</li>
 *   <li>Never moves the root, so the field's flat level direction stays pinned at
 *       {@code u_root = 0} (matching the reference used by {@link GenerationTimeFieldPrior}).</li>
 * </ul>
 *
 * <p>The proposal is a symmetric uniform window move, so the Hastings ratio is zero.
 */
@Description("Signed uniform random-walk on the latent log-rate field over all non-root nodes.")
public class GenerationTimeFieldRandomWalkOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root node field values are updated.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent field vector indexed by BEAST tree node number.",
            Input.Validate.REQUIRED
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk window size.",
            0.1
    );

    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        double windowSize = windowSizeInput.get();

        if (windowSize <= 0.0 ||
                Double.isNaN(windowSize) ||
                Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException(
                    "windowSize must be a positive finite number."
            );
        }

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }
    }

    @Override
    public double proposal() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();
        double windowSize = windowSizeInput.get();

        int nodeCount = tree.getNodeCount();
        int rootNodeNr = tree.getRoot().getNr();

        // Uniformly choose one of the (nodeCount - 1) non-root nodes. Node numbers are the
        // contiguous range 0 .. nodeCount-1, so map an index in [0, nodeCount-1) onto that
        // range while skipping the current root number.
        int index = Randomizer.nextInt(nodeCount - 1);
        int nodeNr = (index < rootNodeNr) ? index : index + 1;

        double oldValue = generationTimes.getValue(nodeNr);
        double proposal = oldValue + windowSize * (Randomizer.nextDouble() - 0.5);

        generationTimes.setValue(nodeNr, proposal);

        // Symmetric uniform window move.
        return 0.0;
    }
}
