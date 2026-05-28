package gentime;

import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeVarianceGibbsOperator extends Operator {
	
    /*
     * Gibbs update for the Brownian variance of the latent generation-time process.
     */
	
    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Generation-time state vector indexed by BEAST tree node number.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> brownianVarianceInput = new Input<>(
            "brownianVariance",
            "Brownian variance parameter to update.",
            Input.Validate.REQUIRED
    );

    public Input<GenerationTimePrecisionBuilder> precisionBuilderInput = new Input<>(
            "precisionBuilder",
            "Sparse Brownian precision builder.",
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

    @Override
    public void initAndValidate() {

        RealParameter generationTimes = generationTimesInput.get();
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();
        
        if (generationTimes.getDimension() != precisionBuilder.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match precision-builder node count."
            );
        }

        if (brownianVarianceInput.get().getDimension() != 1) {
            throw new IllegalArgumentException(
                    "brownianVariance must be a scalar RealParameter."
            );
        }

        if (priorAlphaInput.get() <= 0.0 || priorBetaInput.get() <= 0.0) {
            throw new IllegalArgumentException(
                    "priorAlpha and priorBeta must both be positive."
            );
        }
    }

    @Override
	public double proposal() {
    	RealParameter generationTimes = generationTimesInput.get();
	    RealParameter brownianVariance = brownianVarianceInput.get();
	    GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();
	
	    precisionBuilder.refreshFromTreeIfNeeded();
	        
	    double oldVariance = brownianVariance.getValue();
	
	    double quadraticFormTriplet = sparseQuadraticForm(
	    		generationTimes,
	            precisionBuilder
	    );
	
	    int brownianRank = precisionBuilder.getEdgeCount();
	
	    double priorAlpha = priorAlphaInput.get();
	    double priorBeta = priorBetaInput.get();
	
	    double posteriorAlpha = priorAlpha + 0.5 * brownianRank;
	    double posteriorBeta = priorBeta + 0.5 * quadraticFormTriplet;

        double oldConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
               oldVariance,
               posteriorAlpha,
               posteriorBeta
        );
	
        double varianceDraw = GenerationTimeMathUtils.sampleInverseGamma(
               posteriorAlpha,
               posteriorBeta
        );
            
	    if (varianceDraw <= 0.0 || Double.isNaN(varianceDraw) || Double.isInfinite(varianceDraw)) {
	    	return Double.NEGATIVE_INFINITY;
	    }
	
        double newConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
               varianceDraw,
               posteriorAlpha,
               posteriorBeta
        );

        double logHastingsRatio = oldConditionalLogDensity - newConditionalLogDensity;

	    brownianVariance.setValue(0, varianceDraw);

        return logHastingsRatio;
    }


    protected double sparseQuadraticForm(
            RealParameter generationTimes,
            GenerationTimePrecisionBuilder precisionBuilder
    ) {

        double total = 0.0;

        for (int k = 0; k < precisionBuilder.getUniqueTripletCount(); k++) {

            int row = precisionBuilder.getQUniqueRow(k);
            int col = precisionBuilder.getQUniqueCol(k);
            double value = precisionBuilder.getQUniqueValue(k);

            total += generationTimes.getValue(row)
                    * value
                    * generationTimes.getValue(col);
        }

        return total;
    }
    
    protected double edgeIncrementQuadraticForm(
            RealParameter generationTimes,
            GenerationTimePrecisionBuilder precisionBuilder
    ) {

        double total = 0.0;

        for (int e = 0; e < precisionBuilder.getEdgeCount(); e++) {

            int parent = precisionBuilder.getParent(e);
            int child = precisionBuilder.getChild(e);
            double length = precisionBuilder.getBranchLength(e);

            double parentValue = generationTimes.getValue(parent);
            double childValue = generationTimes.getValue(child);
            double difference = childValue - parentValue;

            total += difference * difference / length;
        }

        return total;
    }
}