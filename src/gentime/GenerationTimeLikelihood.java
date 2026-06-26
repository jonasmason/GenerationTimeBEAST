package gentime;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeLikelihood extends Distribution {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose tip taxa are associated with observed generation times.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimeTraitData> traitDataInput = new Input<>(
            "traitData",
            "Observed generation-time trait data.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent phylogenetic effect vector u indexed by BEAST tree node number. Root is fixed to zero.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Intercept parameter beta in y = beta + u_tip + e.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> brownianVarianceInput = new Input<>(
            "brownianVariance",
            "Brownian variance parameter for latent phylogenetic effects u.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> residualVarianceInput = new Input<>(
            "residualVariance",
            "Residual variance parameter for observed tip generation times.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimePrecisionBuilder> precisionBuilderInput = new Input<>(
            "precisionBuilder",
            "Sparse Brownian precision builder for the latent phylogenetic effect model.",
            Input.Validate.REQUIRED
    );

    public Input<Double> priorBetaMeanInput = new Input<>(
            "priorBetaMean",
            "Prior mean for beta.",
            0.0
    );

    public Input<Double> priorBetaVarianceInput = new Input<>(
            "priorBetaVariance",
            "Prior variance for beta.",
            1000.0
    );

    public Input<String> scaleInput = new Input<>(
            "scale",
            "Scale for observed generation-time model: 'raw' or 'log'.",
            "raw"
    );

    protected int leafCount;
    protected int rootNodeNr;

    protected int[] tipNodeNrs;
    protected double[] observedTipValues;
    
    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();
        RealParameter generationTimes = generationTimesInput.get();
        RealParameter beta = betaInput.get();
        RealParameter brownianVariance = brownianVarianceInput.get();
        RealParameter residualVariance = residualVarianceInput.get();
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        String scale = scaleInput.get();

        GenerationTimeTraitTransform.validateScale(scale);

        leafCount = tree.getLeafNodeCount();
        rootNodeNr = tree.getRoot().getNr();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must equal tree node count. " +
                    "Expected " + tree.getNodeCount() +
                    " but found " + generationTimes.getDimension()
            );
        }

        if (beta.getDimension() != 1) {
            throw new IllegalArgumentException("beta must be a scalar RealParameter.");
        }

        if (brownianVariance.getDimension() != 1) {
            throw new IllegalArgumentException("brownianVariance must be a scalar RealParameter.");
        }

        if (residualVariance.getDimension() != 1) {
            throw new IllegalArgumentException("residualVariance must be a scalar RealParameter.");
        }

        if (priorBetaVarianceInput.get() <= 0.0) {
            throw new IllegalArgumentException("priorBetaVariance must be positive.");
        }

        if (traitData.size() < leafCount) {
            throw new IllegalArgumentException(
                    "Trait data contains fewer entries than tree tips."
            );
        }

        if (precisionBuilder.getNodeCount() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "Precision builder node count does not match tree node count."
            );
        }

        tipNodeNrs = new int[leafCount];
        observedTipValues = new double[leafCount];

        for (int i = 0; i < leafCount; i++) {

            Node tip = tree.getNode(i);
            String taxon = tip.getID();

            if (!traitData.hasTrait(taxon)) {
                throw new IllegalArgumentException(
                        "Missing generation-time value for tree tip: " + taxon
                );
            }

            tipNodeNrs[i] = tip.getNr();

            observedTipValues[i] = GenerationTimeTraitTransform.transform(
                    traitData.getTrait(taxon),
                    scale
            );
        }
    }

    @Override
    public double calculateLogP() {

        Tree tree = treeInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        int currentRootNodeNr = tree.getRoot().getNr();

        double beta = betaInput.get().getValue();
	    double brownianVariance = brownianVarianceInput.get().getValue();
	    double residualVariance = residualVarianceInput.get().getValue();
	
	    if (brownianVariance <= 0.0 || residualVariance <= 0.0) {
	        logP = Double.NEGATIVE_INFINITY;
	        return logP;
	    }
	
	    if (Math.abs(generationTimes.getValue(currentRootNodeNr)) > 1e-10) {
	        logP = Double.NEGATIVE_INFINITY;
	        return logP;
	    }
	
	    double betaLogP = betaLogLikelihood(beta);
	    double tipLogP = tipObservationLogLikelihood(
	            tree,
	            traitData,
	            generationTimes,
	            beta,
	            residualVariance
	    );

	    double brownianLogP = brownianLatentLogLikelihood(
	            tree,
	            generationTimes,
	            brownianVariance
	    );
	
	    logP = betaLogP + tipLogP + brownianLogP;
	
	    return logP;
	}

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }
            
    protected double betaLogLikelihood(double beta) {

    	// log density: log p(β) = -0.5 log(2*pi σ²_β) - (β - μ_β)² / (2 σ²_β)
    	
        double priorMean = priorBetaMeanInput.get();
        double priorVariance = priorBetaVarianceInput.get();

        double residual = beta - priorMean;

        return -0.5 * Math.log(2.0 * Math.PI * priorVariance)
                -0.5 * residual * residual / priorVariance;
    }

    protected double tipObservationLogLikelihood(
        Tree tree,
        GenerationTimeTraitData traitData,
        RealParameter generationTimes,
        double beta,
        double residualVariance
	) {
	
    	// Sum log N(y_i; β + u_tip_i, σ²_resid) over all tips
    	
	    double total = 0.0;
	
	    //  -0.5 log(2pi σ²) constant across tips, so avoid recomputation
	    
	    double logNormalConstant =
	            -0.5 * Math.log(2.0 * Math.PI * residualVariance);
	
	    for (int i = 0; i < leafCount; i++) {
	
	        int nodeNr = tipNodeNrs[i];
	
	        double observed = observedTipValues[i];
	        double latentU = generationTimes.getValue(nodeNr);
	        double fitted = beta + latentU;
	        double residual = observed - fitted;
	        
	        // = Σ_i [ -0.5 * log(2pi σ²_resid) - (y_i - (beta + u_i))^2 / (2 σ²_resid) ]
	
	        total += logNormalConstant
	                -0.5 * residual * residual / residualVariance;
	    }
	
	    return total;
	}

    protected double brownianLatentLogLikelihood(
        Tree tree,
        RealParameter generationTimes,
        double brownianVariance
    ) {

	    double quadraticForm = edgeIncrementQuadraticForm(
	            tree,
	            generationTimes
	    );
	
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

		// Floor degenerate/non-finite branches to the precision builder's
		// minBranchLength so a 0-length branch gives a finite term, not -Inf
		// (same floor as the precision matrix Q -> normalizer stays consistent).
		if (!(branchLength > 0.0) || Double.isInfinite(branchLength)) {
			branchLength = precisionBuilderInput.get().getMinBranchLength();
		}
	
	        logBranchLengthSum += Math.log(branchLength);
	    }
	
	    int brownianRank = tree.getNodeCount() - 1;
	    
	    // full log-density of Brownian process: log p(u | σ², tree) = -0.5(n-1)log(2pi σ²) - ½ Σ log(length_e) - ½ u^T Q u / σ²
	
	    return -0.5 * brownianRank * Math.log(2.0 * Math.PI * brownianVariance)
	            -0.5 * logBranchLengthSum
	            -0.5 * quadraticForm / brownianVariance;
	}

    protected double edgeIncrementQuadraticForm(
            Tree tree,
            RealParameter generationTimes
    ) {

    	// Since Q = (branch-length)^{-1} on each edge, the quadratic form decomposes edge-by-edge as Σ_e (u_child - u_parent)² / length_e
    	
        double total = 0.0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            Node parent = node.getParent();

            double branchLength = node.getLength();

			// Same minBranchLength floor as the precision matrix Q (see above).
			if (!(branchLength > 0.0) || Double.isInfinite(branchLength)) {
				branchLength = precisionBuilderInput.get().getMinBranchLength();
			}

            double childValue = generationTimes.getValue(node.getNr());
            double parentValue = generationTimes.getValue(parent.getNr());

            double difference = childValue - parentValue;

            total += difference * difference / branchLength;
        }

        return total;
    }
    
    public String diagnosticSummary(String label) {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        int currentRootNodeNr = tree.getRoot().getNr();

        double beta = betaInput.get().getValue();
        double brownianVariance = brownianVarianceInput.get().getValue();
        double residualVariance = residualVarianceInput.get().getValue();

        double betaLogP = betaLogLikelihood(beta);

        double tipLogP = tipObservationLogLikelihood(
                tree,
                traitDataInput.get(),
                generationTimes,
                beta,
                residualVariance
        );

        double quadraticForm = edgeIncrementQuadraticForm(
                tree,
                generationTimes
        );

        double logBranchLengthSum = 0.0;
        double treeLengthSum = 0.0;
        int edgeCount = 0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                continue;
            }

            double branchLength = node.getLength();
			if (!(branchLength > 0.0) || Double.isInfinite(branchLength)) {
				branchLength = precisionBuilderInput.get().getMinBranchLength();
			}

			logBranchLengthSum += Math.log(branchLength);
			treeLengthSum += branchLength;
            edgeCount++;
        }

        double brownianLogP = brownianLatentLogLikelihood(
                tree,
                generationTimes,
                brownianVariance
        );

        double total = betaLogP + tipLogP + brownianLogP;

        return label +
                " logP=" + total +
                " betaLogP=" + betaLogP +
                " tipLogP=" + tipLogP +
                " brownianLogP=" + brownianLogP +
                " quadraticForm=" + quadraticForm +
                " logBranchLengthSum=" + logBranchLengthSum +
                " treeLengthSum=" + treeLengthSum +
                " edgeCount=" + edgeCount +
                " rootNr=" + currentRootNodeNr +
                " rootU=" + generationTimes.getValue(currentRootNodeNr) +
                " beta=" + beta +
                " brownianVariance=" + brownianVariance +
                " residualVariance=" + residualVariance;
    }

    @Override
    public List<String> getArguments() {

        List<String> arguments = new ArrayList<>();

        arguments.add(treeInput.get().getID());
        arguments.add(generationTimesInput.get().getID());
        arguments.add(betaInput.get().getID());
        arguments.add(brownianVarianceInput.get().getID());
        arguments.add(residualVarianceInput.get().getID());

        return arguments;
    }

    @Override
    public List<String> getConditions() {

        List<String> conditions = new ArrayList<>();

        conditions.add(traitDataInput.get().getID());

        return conditions;
    }

    @Override
    public void sample(State state, Random random) {
        // not used
    }
}