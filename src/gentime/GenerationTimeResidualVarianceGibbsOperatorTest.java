package gentime;

import beast.base.core.Input;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeResidualVarianceGibbsOperatorTest
        extends GenerationTimeResidualVarianceGibbsOperator {

    public Input<Boolean> debugInput = new Input<>(
            "debug",
            "Print residual-variance Gibbs diagnostics when true.",
            false
    );

    @Override
    public void initAndValidate() {

        super.initAndValidate();

        if (debugInput.get()) {
            System.out.println("GenerationTimeResidualVarianceGibbsOperatorTest initialised.");
            System.out.println("leafCount = " + leafCount);
        }
    }

    @Override
    public double proposal() {

        RealParameter generationTimes = generationTimesInput.get();
        RealParameter residualVariance = residualVarianceInput.get();

        double oldGtLogP = gtLikelihoodInput.get().calculateLogP();
        String oldGtDiagnosticSummary =
                gtLikelihoodInput.get().diagnosticSummary("beforeResidualVarianceGibbs");

        double beta = betaInput.get().getValue();
        double oldVariance = residualVariance.getValue();

        double residualSumOfSquares = residualSumOfSquares(
                generationTimes,
                beta
        );

        double priorAlpha = priorAlphaInput.get();
        double priorBeta = priorBetaInput.get();

        double posteriorAlpha = priorAlpha + 0.5 * leafCount;
        double posteriorBeta = priorBeta + 0.5 * residualSumOfSquares;

        double oldConditionalLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        oldVariance,
                        posteriorAlpha,
                        posteriorBeta
                );

        double oldTipLogDensity =
                -0.5 * leafCount * Math.log(2.0 * Math.PI * oldVariance)
                        -0.5 * residualSumOfSquares / oldVariance;

        double oldPriorLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        oldVariance,
                        priorAlpha,
                        priorBeta
                );

        double gammaDraw =
                GenerationTimeMathUtils.sampleGammaShapeScaleOne(posteriorAlpha);

        double varianceDraw = posteriorBeta / gammaDraw;

        if (varianceDraw <= 0.0 ||
                Double.isNaN(varianceDraw) ||
                Double.isInfinite(varianceDraw)) {
            return Double.NEGATIVE_INFINITY;
        }

        double newConditionalLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        varianceDraw,
                        posteriorAlpha,
                        posteriorBeta
                );

        double newTipLogDensity =
                -0.5 * leafCount * Math.log(2.0 * Math.PI * varianceDraw)
                        -0.5 * residualSumOfSquares / varianceDraw;

        double newPriorLogDensity =
                GenerationTimeMathUtils.inverseGammaLogDensity(
                        varianceDraw,
                        priorAlpha,
                        priorBeta
                );

        double logHastingsRatio =
                oldConditionalLogDensity - newConditionalLogDensity;

        double residualPosteriorRatio =
                (newTipLogDensity + newPriorLogDensity)
                        - (oldTipLogDensity + oldPriorLogDensity);

        double cancellationError =
                residualPosteriorRatio + logHastingsRatio;

        residualVariance.setValue(0, varianceDraw);

        double newGtLogP = gtLikelihoodInput.get().calculateLogP();
        String newGtDiagnosticSummary =
                gtLikelihoodInput.get().diagnosticSummary("afterResidualVarianceGibbs");

        double actualGtLogRatio = newGtLogP - oldGtLogP;
        double expectedTipLogRatio = newTipLogDensity - oldTipLogDensity;
        double gtLikelihoodMismatch = actualGtLogRatio - expectedTipLogRatio;

        if (debugInput.get()) {
            System.out.println("Residual variance Gibbs update:");
            System.out.println("beta = " + beta);
            System.out.println("residualSumOfSquares = " + residualSumOfSquares);
            System.out.println("leafCount = " + leafCount);
            System.out.println("posteriorAlpha = " + posteriorAlpha);
            System.out.println("posteriorBeta = " + posteriorBeta);
            System.out.println("gammaDraw = " + gammaDraw);
            System.out.println("varianceDraw = " + varianceDraw);
            System.out.println("oldVariance = " + oldVariance);
            System.out.println("oldTipLogDensity = " + oldTipLogDensity);
            System.out.println("newTipLogDensity = " + newTipLogDensity);
            System.out.println("oldPriorLogDensity = " + oldPriorLogDensity);
            System.out.println("newPriorLogDensity = " + newPriorLogDensity);
            System.out.println("oldConditionalLogDensity = " + oldConditionalLogDensity);
            System.out.println("newConditionalLogDensity = " + newConditionalLogDensity);
            System.out.println("residualPosteriorRatio = " + residualPosteriorRatio);
            System.out.println("logHastingsRatio = " + logHastingsRatio);
            System.out.println("cancellationError = " + cancellationError);
            System.out.println("oldGtLogP = " + oldGtLogP);
            System.out.println("newGtLogP = " + newGtLogP);
            System.out.println("actualGtLogRatio = " + actualGtLogRatio);
            System.out.println("expectedTipLogRatio = " + expectedTipLogRatio);
            System.out.println("gtLikelihoodMismatch = " + gtLikelihoodMismatch);

            if (Math.abs(gtLikelihoodMismatch) > 1.0e-8) {
                System.out.println(oldGtDiagnosticSummary);
                System.out.println(newGtDiagnosticSummary);
            }
        }

        return logHastingsRatio;
    }
}