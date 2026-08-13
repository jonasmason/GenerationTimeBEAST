package gentime;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;

@Description("Branch-rate model where log molecular rate depends on branch generation time plus a residual.")
public class GenerationTimeRegressionClockModel extends BranchRateModel.Base {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose branches receive generation-time-informed molecular rates.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> generationTimesInput = new Input<>(
            "generationTimes",
            "Generation-time state vector indexed by BEAST node number.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> betaInput = new Input<>(
            "beta",
            "Generation-time intercept beta; actual generation time is beta + latent generationTimes.",
            Input.Validate.REQUIRED
    );
    
    public Input<RealParameter> clockInterceptInput = new Input<>(
            "clockIntercept",
            "Intercept in the log-linear generation-time clock regression.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockSlopeInput = new Input<>(
            "clockSlope",
            "Slope of generation time in the log-linear clock regression.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> clockResidualsInput = new Input<>(
            "clockResiduals",
            "Branch-specific log-rate residuals indexed by child BEAST node number.",
            Input.Validate.REQUIRED
    );

    public Input<Boolean> includeBetaInRateInput = new Input<>(
            "includeBetaInRate",
            "If true (default), the clock predictor is beta + latent generation time. "
                    + "If false, beta is dropped and the predictor is the (root-centred) "
                    + "latent field alone, with the mean absorbed into the base rate. "
                    + "Setting false decouples beta from the clock so it is determined by "
                    + "the trait data only.",
            true
    );

    public Input<String> branchGenerationTimeModeInput = new Input<>(
            "branchGenerationTimeMode",
            "How to convert parent/child node generation times to branch predictor: 'midpoint', 'child', or 'parent'.",
            "midpoint"
    );

    public Input<Boolean> meanCenterPredictorInput = new Input<>(
            "meanCenterPredictor",
            "If true, the branch predictor is centred by subtracting its across-branch "
                    + "mean each evaluation, so the slope pivots around the data centroid "
                    + "rather than around zero. This decorrelates clockSlope from the "
                    + "base-rate/intercept level and leaves the slope depending only on the "
                    + "relative structure of the latent field. Default false (legacy behaviour).",
            false
    );

    public Input<Boolean> staticCenterInput = new Input<>(
            "staticCenter",
            "Only used when meanCenterPredictor=true. If true, the across-branch predictor mean is "
                    + "computed ONCE at initialisation and then held fixed (static centring at the "
                    + "initial data centroid) instead of being recomputed every evaluation (dynamic "
                    + "centring). Static centring subtracts a fixed constant, so it is a pure "
                    + "reparameterisation that stays valid even when the clock level/intercept is "
                    + "identified; dynamic centring is only safe when the level is non-identified. "
                    + "Default false (dynamic, legacy behaviour).",
            false
    );

    private Tree tree;
    private RealParameter generationTimes;
    private RealParameter beta;
    private RealParameter clockIntercept;
    private RealParameter clockSlope;
    private RealParameter clockResiduals;
    private boolean includeBetaInRate;

    private String branchGenerationTimeMode;
    private boolean meanCenterPredictor;
    private boolean staticCenter;
    private double frozenPredictorMean;
    private boolean frozenPredictorMeanSet = false;

    private double[] rates;
    private double[] storedRates;

    private boolean needsRecompute = true;
    private boolean storedNeedsRecompute = true;

    @Override
    public void initAndValidate() {

        tree = treeInput.get();
        generationTimes = generationTimesInput.get();
        beta = betaInput.get();
        clockIntercept = clockInterceptInput.get();
        clockSlope = clockSlopeInput.get();
        clockResiduals = clockResidualsInput.get();
        includeBetaInRate = includeBetaInRateInput.get();

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
        
        if (clockResiduals.getDimension() != tree.getNodeCount()) {
            throw new IllegalArgumentException(
                    "clockResiduals dimension must match tree node count. " +
                    "Residuals are indexed by child node number; root residual is ignored."
            );
        }

        if (clockIntercept.getDimension() != 1) {
            throw new IllegalArgumentException(
                    "clockIntercept must be a scalar RealParameter."
            );
        }

        if (clockSlope.getDimension() != 1) {
            throw new IllegalArgumentException(
                    "clockSlope must be a scalar RealParameter."
            );
        }

        branchGenerationTimeMode = branchGenerationTimeModeInput.get();
        meanCenterPredictor = meanCenterPredictorInput.get();
        staticCenter = staticCenterInput.get();

        if (!branchGenerationTimeMode.equals("midpoint") &&
                !branchGenerationTimeMode.equals("child") &&
                !branchGenerationTimeMode.equals("parent")) {
            throw new IllegalArgumentException(
                    "branchGenerationTimeMode must be 'midpoint', 'child', or 'parent'."
            );
        }

        rates = new double[tree.getNodeCount()];
        storedRates = new double[tree.getNodeCount()];

        recomputeRates();
    }

    @Override
    public double getRateForBranch(final Node node) {

        if (node.isRoot()) {
            return 0.0;
        }

        if (needsRecompute) {
            recomputeRates();
        }

        return rates[node.getNr()];
    }

    private void recomputeRates() {

        double intercept = clockIntercept.getValue();
        double slope = clockSlope.getValue();

        // Across-branch mean of the predictor, for optional mean-centring. Dynamic (default):
        // recomputed each evaluation so it tracks the current latent field. Static: computed once
        // at the first evaluation and then frozen, so it subtracts a fixed constant.
        double predictorMean = 0.0;
        if (meanCenterPredictor) {
            if (staticCenter && frozenPredictorMeanSet) {
                predictorMean = frozenPredictorMean;
            } else {
                double sum = 0.0;
                int count = 0;
                for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
                    Node node = tree.getNode(nodeNr);
                    if (node.isRoot()) {
                        continue;
                    }
                    sum += getBranchGenerationTime(node);
                    count++;
                }
                predictorMean = (count > 0) ? sum / count : 0.0;
                if (staticCenter) {
                    frozenPredictorMean = predictorMean;
                    frozenPredictorMeanSet = true;
                }
            }
        }

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {

            Node node = tree.getNode(nodeNr);

            if (node.isRoot()) {
                rates[nodeNr] = 0.0;
                continue;
            }

            double branchGenerationTime = getBranchGenerationTime(node) - predictorMean;
            double residual = clockResiduals.getValue(nodeNr);

            double logRate = intercept
                    + slope * branchGenerationTime
                    + residual;

            double rate = Math.exp(logRate);

            if (rate <= 0.0 || Double.isNaN(rate) || Double.isInfinite(rate)) {
                throw new IllegalArgumentException(
                        "Invalid branch rate for node " + nodeNr +
                        ": rate=" + rate +
                        ", logRate=" + logRate +
                        ", branchGenerationTime=" + branchGenerationTime +
                        ", residual=" + residual
                );
            }

            rates[nodeNr] = rate;
        }

        needsRecompute = false;
    }

    private double getBranchGenerationTime(final Node childNode) {

	    int childNodeNr = childNode.getNr();
	    int parentNodeNr = childNode.getParent().getNr();
	
	    double betaValue = includeBetaInRate ? beta.getValue() : 0.0;
	
	    double childGenerationTime =
	            betaValue + generationTimes.getValue(childNodeNr);
	
	    double parentGenerationTime =
	            betaValue + generationTimes.getValue(parentNodeNr);
	
	    if (branchGenerationTimeMode.equals("child")) {
	        return childGenerationTime;
	    }
	
	    if (branchGenerationTimeMode.equals("parent")) {
	        return parentGenerationTime;
	    }
	
	    return 0.5 * (parentGenerationTime + childGenerationTime);
	}

    @Override
    public boolean requiresRecalculation() {
        needsRecompute = true;
        return true;
    }

    @Override
    protected void store() {
        System.arraycopy(rates, 0, storedRates, 0, rates.length);
        storedNeedsRecompute = needsRecompute;
        super.store();
    }

    @Override
    protected void restore() {
        double[] tmpRates = rates;
        rates = storedRates;
        storedRates = tmpRates;

        needsRecompute = storedNeedsRecompute;

        super.restore();
    }
}