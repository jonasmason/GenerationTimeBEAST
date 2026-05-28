package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

public class GenerationTimeRandomWalkOperator extends Operator {
	
    /*
     * Initial random-walk operator for generation-time states.
     * Generation-time updates are now handled by GenerationTimeGibbsOperator.
     */

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose internal node generation times are updated.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Generation-time state vector indexed by BEAST tree node number.",
            Input.Validate.REQUIRED
    );

    public Input<Double> windowSizeInput = new Input<>(
            "windowSize",
            "Uniform random-walk window size.",
            0.1
    );
    
    public Input<Boolean> debugInput = new Input<>(
    		"debug",
    		"Print random-walk operator diagnostics when true.",
    		false
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

        if (debugInput.get()) {
	        System.out.println(
	        		"GenerationTimeRandomWalkOperator initialised for " +
	        				tree.getInternalNodeCount() +
	        				" internal nodes."
	        );
        }
    }

    @Override
    public double proposal() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();
        double windowSize = windowSizeInput.get();

        int internalNodeCount = tree.getInternalNodeCount();

        // Choose one internal node and perturn its current generation-time state.
        
        int chosenInternalIndex = Randomizer.nextInt(internalNodeCount);
        Node node = tree.getInternalNodes().get(chosenInternalIndex);
        int nodeNr = node.getNr();

        double oldValue = generationTimes.getValue(nodeNr);
        double proposal = oldValue + windowSize * (Randomizer.nextDouble() - 0.5);

        if (proposal <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }

        generationTimes.setValue(nodeNr, proposal);

        return 0.0;
    }
}