package gentime;

import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;

/**
 * Gibbs update for the auxiliary scale parameter {@code a} of the half-Cauchy prior on the
 * Brownian standard deviation, in the inverse-gamma scale-mixture representation.
 *
 * <p>Half-Cauchy(sigma; scale = A) on sigma = sqrt(brownianVariance) is equivalent to
 * (Gelman 2006; Wand et al. 2011; the parameter-expanded prior used in MCMCglmm):
 * <pre>
 *     brownianVariance V | a ~ IG(1/2, 1/a)
 *     a                     ~ IG(1/2, 1/A^2)
 * </pre>
 * Both full conditionals are inverse-gamma, so V keeps its conjugate Gibbs (with priorAlpha=1/2
 * and priorBeta = 1/a supplied via {@code GenerationTimeVarianceGibbsOperator.priorScale}), and
 * this operator draws the other one:
 * <pre>
 *     a | V ~ IG(shape = 1, scale = 1/V + 1/A^2)
 * </pre>
 *
 * <p>{@code scaleSq} is A^2 = the square of the half-Cauchy scale (e.g. 1000 for A = sqrt(1000),
 * or 625 for A = 25). Returns the Gibbs-as-Metropolis correction (old minus new conditional
 * log-density), matching {@link GenerationTimeVarianceGibbsOperator}.
 */
public class GenerationTimeHalfCauchyHyperGibbsOperator extends Operator {

    public Input<RealParameter> brownianVarianceInput = new Input<>(
            "brownianVariance",
            "Brownian variance V; supplies the sufficient statistic 1/V for the a update.",
            Input.Validate.REQUIRED
    );

    public Input<RealParameter> hyperInput = new Input<>(
            "hyper",
            "Auxiliary half-Cauchy scale parameter a to update.",
            Input.Validate.REQUIRED
    );

    public Input<Double> scaleSqInput = new Input<>(
            "scaleSq",
            "A^2, the square of the half-Cauchy scale (1000 for A = sqrt(1000)).",
            1000.0
    );

    @Override
    public void initAndValidate() {
        if (brownianVarianceInput.get().getDimension() != 1) {
            throw new IllegalArgumentException("brownianVariance must be a scalar RealParameter.");
        }
        if (hyperInput.get().getDimension() != 1) {
            throw new IllegalArgumentException("hyper (a) must be a scalar RealParameter.");
        }
        if (scaleSqInput.get() <= 0.0) {
            throw new IllegalArgumentException("scaleSq (A^2) must be positive.");
        }
    }

    @Override
    public double proposal() {

        RealParameter hyper = hyperInput.get();

        double v = brownianVarianceInput.get().getValue();
        double a2 = scaleSqInput.get();

        if (v <= 0.0 || Double.isNaN(v) || Double.isInfinite(v)) {
            return Double.NEGATIVE_INFINITY;
        }

        // a | V ~ IG(shape = 1, scale = 1/V + 1/A^2)
        double posteriorShape = 1.0;
        double posteriorScale = 1.0 / v + 1.0 / a2;

        double oldA = hyper.getValue();

        double oldConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
                oldA, posteriorShape, posteriorScale);

        double draw = GenerationTimeMathUtils.sampleInverseGamma(posteriorShape, posteriorScale);

        if (draw <= 0.0 || Double.isNaN(draw) || Double.isInfinite(draw)) {
            return Double.NEGATIVE_INFINITY;
        }

        double newConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
                draw, posteriorShape, posteriorScale);

        double logHastingsRatio = oldConditionalLogDensity - newConditionalLogDensity;

        hyper.setValue(0, draw);

        return logHastingsRatio;
    }
}
