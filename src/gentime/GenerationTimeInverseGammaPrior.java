package gentime;

import beast.base.core.Input;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;

import java.util.List;

public class GenerationTimeInverseGammaPrior extends Distribution {

    public Input<RealParameter> parameterInput = new Input<>(
            "parameter",
            "Positive scalar parameter with inverse-gamma prior.",
            Input.Validate.REQUIRED
    );

    public Input<Double> alphaInput = new Input<>(
            "alpha",
            "Inverse-gamma shape parameter.",
            Input.Validate.REQUIRED
    );

    public Input<Double> betaInput = new Input<>(
            "beta",
            "Inverse-gamma scale parameter.",
            Input.Validate.REQUIRED
    );
    
    public Input<RealParameter> scaleInput = new Input<>(
            "scale", "Optional: beta = 1/scale (a parameter), for the half-Cauchy scale mixture. Overrides beta.",
            (RealParameter) null
    );

    @Override
    public void initAndValidate() {
        RealParameter parameter = parameterInput.get();

        if (parameter.getDimension() != 1) {
            throw new IllegalArgumentException(
                    "GenerationTimeInverseGammaPrior currently expects a scalar RealParameter."
            );
        }

        if (alphaInput.get() <= 0.0 || betaInput.get() <= 0.0) {
            throw new IllegalArgumentException(
                    "Inverse-gamma alpha and beta must both be positive."
            );
        }
    }

    @Override
    public double calculateLogP() {
        double x = parameterInput.get().getValue();
        double alpha = alphaInput.get();
        double beta = betaInput.get();
        if (scaleInput.get() != null) beta = 1.0 / scaleInput.get().getValue();

        if (x <= 0.0 || Double.isNaN(x) || Double.isInfinite(x)) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }

        logP = GenerationTimeMathUtils.inverseGammaLogDensity(
                x,
                alpha,
                beta
        );

        return logP;
    }

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }
    
    @Override
    public List<String> getArguments() {
        java.util.ArrayList<String> arguments = new java.util.ArrayList<>();
        arguments.add(parameterInput.get().getID());
        return arguments;
    }

    @Override
    public List<String> getConditions() {
        return new java.util.ArrayList<>();
    }

    @Override
    public void sample(State state, java.util.Random random) {
        // Prior is evaluated only; sampling is handled by Gibbs operators.
    }
}