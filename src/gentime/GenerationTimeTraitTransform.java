package gentime;

final class GenerationTimeTraitTransform {

    private GenerationTimeTraitTransform() {
        // Utility class; do not instantiate.
    }

    static double transform(
            double value,
            String scale
    ) {

        if (scale.equals("raw")) {
            return value;
        }

        if (scale.equals("log")) {

            if (value <= 0.0) {
                throw new IllegalArgumentException(
                        "Cannot log-transform non-positive generation time: " + value
                );
            }

            return Math.log(value);
        }

        throw new IllegalArgumentException(
                "Unknown generation-time scale: " + scale +
                        ". Expected 'raw' or 'log'."
        );
    }

    static void validateScale(String scale) {

        if (!scale.equals("raw") && !scale.equals("log")) {
            throw new IllegalArgumentException(
                    "scale must be either 'raw' or 'log', but found: " + scale
            );
        }
    }
}