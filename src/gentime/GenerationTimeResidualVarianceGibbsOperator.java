package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeResidualVarianceGibbsOperator extends Operator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose tip taxa are associated with observed generation times.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Latent phylogenetic effect vector u indexed by BEAST tree node number.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Intercept parameter beta in y = beta + u_tip + e.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> residualVarianceInput = new Input<>(
            "residualVariance",
            "Residual variance parameter to update.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimeTraitData> traitDataInput = new Input<>(
            "traitData",
            "Observed generation-time tip data.",
            Input.Validate.REQUIRED
    );
    
    public Input<GenerationTimeLikelihood> gtLikelihoodInput = new Input<>(
            "gtLikelihood",
            "Generation-time likelihood to recalculate immediately after this operator writes GT state.",
            Input.Validate.REQUIRED
    );

    public Input<Double> priorAlphaInput = new Input<>(
            "priorAlpha",
            "Inverse-gamma prior shape parameter.",
            0.001
    );

    public Input<Double> priorBetaInput = new Input<>(
            "priorBeta",
            "Inverse-gamma prior scale parameter.",
            0.001
    );

    public Input<String> scaleInput = new Input<>(
            "scale",
            "Scale for observed generation-time model: 'raw' or 'log'.",
            "raw"
    );

    protected int leafCount;
    protected int[] tipNodeNrs;
    protected double[] observedTipValues;

    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();
        RealParameter beta = betaInput.get();
        RealParameter residualVariance = residualVarianceInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();

        String scale = scaleInput.get();

        GenerationTimeTraitTransform.validateScale(scale);

        leafCount = tree.getLeafNodeCount();

        if (generationTimes.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match tree node count."
            );
        }

        if (beta.getDimension() != 1) {
            throw new IllegalArgumentException("beta must be a scalar RealParameter.");
        }

        if (residualVariance.getDimension() != 1) {
            throw new IllegalArgumentException("residualVariance must be a scalar RealParameter.");
        }

        if (priorAlphaInput.get() <= 0.0 || priorBetaInput.get() <= 0.0) {
            throw new IllegalArgumentException(
                    "priorAlpha and priorBeta must both be positive."
            );
        }

        if (traitData.size() < leafCount) {
            throw new IllegalArgumentException(
                    "Trait data contains fewer entries than tree tips."
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
    public double proposal() {

    	RealParameter generationTimes = generationTimesInput.get();
    	RealParameter betaParameter = betaInput.get();
    	RealParameter residualVariance = residualVarianceInput.get();

    	/*
    	 * Force the GT likelihood to synchronise with the current BEAST state before constructing the residual-variance full conditional.
    	 *
    	 * Same as the precision refresh used by the Brownian variance Gibbs operator, but here the dependency is the tip observation likelihood:
    	 * y_tip ~ Normal(beta + u_tip, residualVariance).
    	 */
    	
    	gtLikelihoodInput.get().calculateLogP();

    	double beta = betaParameter.getValue();
    	double oldVariance = residualVariance.getValue();

    	double residualSumOfSquares = residualSumOfSquares(
    	        generationTimes,
    	        beta
    	);

        double priorAlpha = priorAlphaInput.get();
        double priorBeta = priorBetaInput.get();

        double posteriorAlpha = priorAlpha + 0.5 * leafCount;
        double posteriorBeta = priorBeta + 0.5 * residualSumOfSquares;

        double oldConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
                oldVariance,
                posteriorAlpha,
                posteriorBeta
        );

        double varianceDraw = GenerationTimeMathUtils.sampleInverseGamma(
                posteriorAlpha,
                posteriorBeta
        );

        if (varianceDraw <= 0.0 ||
                Double.isNaN(varianceDraw) ||
                Double.isInfinite(varianceDraw)) {
            return Double.NEGATIVE_INFINITY;
        }

        double newConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
                varianceDraw,
                posteriorAlpha,
                posteriorBeta
        );

        double logHastingsRatio =
                oldConditionalLogDensity - newConditionalLogDensity;

        residualVariance.setValue(0, varianceDraw);

        gtLikelihoodInput.get().calculateLogP();
        
        return logHastingsRatio;
    }
    
    protected double residualSumOfSquares(
            RealParameter generationTimes,
            double beta
    ) {

        double total = 0.0;

        for (int i = 0; i < leafCount; i++) {

            int nodeNr = tipNodeNrs[i];
            double observed = observedTipValues[i];

            double latentU = generationTimes.getValue(nodeNr);
            double fitted = beta + latentU;
            double residual = observed - fitted;

            total += residual * residual;
        }

        return total;
    }
}