package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/*
 * Local Metropolis random-walk operator for clock-linked GT analyses.
 *
 * This is deliberately separate from GenerationTimeGibbsOperator.
 *
 * The original Gibbs operator is exact for the GT-only trait model, but after
 * generation time also affects the molecular clock, the full conditional includes
 * the sequence likelihood. This operator makes small local proposals and lets
 * BEAST evaluate the full posterior normally.
 *
 * Modes:
 *   beta
 *     updates the GT intercept beta
 *
 *   generationTime
 *     updates one non-root latent generation-time state u_node
 *
 * The proposal is symmetric, so the Hastings ratio is zero.
 */
public class GenerationTimeClockLinkedRandomWalkOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose non-root generation-time states may be updated.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent generation-time state vector indexed by BEAST node number.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Generation-time intercept beta.",
            Input.Validate.REQUIRED
    );

    public Input<String> modeInput = new Input<>(
            "mode",
            "Which quantity to update: 'beta' or 'generationTime'.",
            Input.Validate.REQUIRED
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk half-width.",
            0.05
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter beta;
    private String mode;
    private double windowSize;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        beta = betaInput.get();

        mode = modeInput.get();
        windowSize = windowSizeInput.get();

        if (!mode.equals("beta") && !mode.equals("generationTime")) {
            throw new IllegalArgumentException(
                    "mode must be either 'beta' or 'generationTime'."
            );
        }

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (beta.getDimension() != 1) {
            throw new IllegalArgumentException(
                    "beta must be a scalar RealParameter."
            );
        }

        if (windowSize <= 0.0 ||
                Double.isNaN(windowSize) ||
                Double.isInfinite(windowSize)) {
            throw new IllegalArgumentException(
                    "windowSize must be positive and finite."
            );
        }
    }

    @Override
    public double proposal() {

        double delta = windowSize * (2.0 * Randomizer.nextDouble() - 1.0);

        if (mode.equals("beta")) {
            beta.setValue(0, beta.getValue() + delta);
            return 0.0;
        }

        int nodeNr = drawNonRootNodeNr();

        generationTimes.setValue(
                nodeNr,
                generationTimes.getValue(nodeNr) + delta
        );

        return 0.0;
    }

    private int drawNonRootNodeNr() {

        int nodeCount = tree.getNodeCount();

        while (true) {
            int nodeNr = Randomizer.nextInt(nodeCount);
            Node node = tree.getNode(nodeNr);

            if (!node.isRoot()) {
                return nodeNr;
            }
        }
    }
}