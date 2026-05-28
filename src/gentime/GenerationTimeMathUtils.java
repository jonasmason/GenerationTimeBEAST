package gentime;

import beast.base.util.Randomizer;

final class GenerationTimeMathUtils {

    private GenerationTimeMathUtils() {
        // Utility class; do not instantiate.
    }

    static double inverseGammaLogDensity(
            double x,
            double alpha,
            double beta
    ) {

        if (x <= 0.0 ||
                Double.isNaN(x) ||
                Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }

        return alpha * Math.log(beta)
                - logGamma(alpha)
                - (alpha + 1.0) * Math.log(x)
                - beta / x;
    }

    static double sampleInverseGamma(
            double alpha,
            double beta
    ) {

        double gammaDraw = sampleGammaShapeScaleOne(alpha);
        return beta / gammaDraw;
    }

    static double sampleGammaShapeScaleOne(double shape) {

        if (shape <= 0.0) {
            throw new IllegalArgumentException(
                    "Gamma shape must be positive."
            );
        }

        if (shape < 1.0) {
            double u = Randomizer.nextDouble();
            return sampleGammaShapeScaleOne(shape + 1.0) *
                    Math.pow(u, 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);

        while (true) {

            double x = Randomizer.nextGaussian();
            double v = 1.0 + c * x;

            if (v <= 0.0) {
                continue;
            }

            v = v * v * v;

            double u = Randomizer.nextDouble();

            if (u < 1.0 - 0.0331 * x * x * x * x) {
                return d * v;
            }

            if (Math.log(u) <
                    0.5 * x * x +
                            d * (1.0 - v + Math.log(v))) {
                return d * v;
            }
        }
    }

    static double logGamma(double x) {

        double[] coefficients = {
                76.18009172947146,
                -86.50532032941677,
                24.01409824083091,
                -1.231739572450155,
                0.001208650973866179,
                -0.000005395239384953
        };

        double y = x;
        double tmp = x + 5.5;

        tmp -= (x + 0.5) * Math.log(tmp);

        double ser = 1.000000000190015;

        for (double coefficient : coefficients) {
            y += 1.0;
            ser += coefficient / y;
        }

        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }
}