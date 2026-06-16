---
editor_options: 
  markdown: 
    wrap: 72
---

# GenerationTimeInverseGammaPrior.java explained

## What this file does

Provides a BEAST `Distribution` (prior) for a scalar positive parameter
whose prior distribution is InverseGamma(alpha, beta). It is used in the
GT project to specify priors on `brownianVariance` and
`residualVariance`.

In BEAST's computation graph, every term in the posterior is a
`Distribution` subclass. The framework calls `calculateLogP()` on each
distribution during every MCMC step to evaluate the posterior.

------------------------------------------------------------------------

## Background: BEAST's `Distribution`

`Distribution` is BEAST's abstract base class for any term that
contributes to the posterior log-density. Subclasses must implement:

-    `calculateLogP()` : returns the log-probability and stores it in
    `this.logP`

-   `getArguments()` : returns the list of state-node IDs that are the
    random variables under this distribution

-   `getConditions()` : returns the list of IDs that are treated as
    fixed conditioning quantities

-   `sample(State, Random)` : draws a sample; not used for priors
    evaluated by Gibbs operators but required by the interface

-   `initAndValidate()` : startup validation

------------------------------------------------------------------------

## Line-by-line walkthrough

``` java
package gentime;

import beast.base.core.Input;
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;

import java.util.List;
```

-   `Distribution` : the BEAST abstract base for posterior terms.
-   `State` : the BEAST container holding all state nodes (parameters,
    trees).
-   `RealParameter` : BEAST's type for a real-valued parameter vector.
-   `List` : Java's ordered-collection interface, used for the return
    types of `getArguments()` and `getConditions()`.

------------------------------------------------------------------------

``` java
public class GenerationTimeInverseGammaPrior extends Distribution {
```

`extends Distribution` makes this a proper BEAST distribution that the
framework knows how to call.

------------------------------------------------------------------------

``` java
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
```

Three XML inputs.

------------------------------------------------------------------------

### `initAndValidate()`

``` java
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
```

The two guards check:

1\. That the parameter is a scalar (dimension 1) -\> the class doesn't
support vector inverse-gamma priors.

2\. That the hyperparameters are positive -\> the inverse-gamma
distribution is undefined for non-positive shape or scale.

------------------------------------------------------------------------

### `calculateLogP()`

``` java
    @Override
    public double calculateLogP() {
        double x = parameterInput.get().getValue();
        double alpha = alphaInput.get();
        double beta = betaInput.get();
```

`RealParameter.getValue()` returns the scalar value (the first element
if dimension \> 1, but we've guaranteed dimension = 1 above).

------------------------------------------------------------------------

``` java
        if (x <= 0.0 || Double.isNaN(x) || Double.isInfinite(x)) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }
```

The inverse-gamma density is only defined for x \> 0. Writing −∞ to
`this.logP` and returning it causes BEAST to reject any proposal that
produces an invalid variance value. The field `logP` is inherited from
`Distribution`; BEAST reads it after calling `calculateLogP()`.

------------------------------------------------------------------------

``` java
        logP = GenerationTimeMathUtils.inverseGammaLogDensity(
                x,
                alpha,
                beta
        );

        return logP;
    }
```

Delegates to the math utility. Storing in `logP` (the inherited field)
allows BEAST's internal bookkeeping to cache the value correctly.

------------------------------------------------------------------------

### `requiresRecalculation()`

``` java
    @Override
    protected boolean requiresRecalculation() {
        return true;
    }
```

Tells BEAST that the log-probability must be recomputed every MCMC step.
Returning `true` always is the safest default. BEAST uses this flag to
decide whether it can skip calling `calculateLogP()` for this
distribution. If a parameter changes every Gibbs step, recalculation is
always needed.

------------------------------------------------------------------------

### `getArguments()` and `getConditions()`

``` java
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
```

BEAST requires these methods to map the distribution to state nodes. The
"argument" is the random variable (the variance parameter); the
"conditions" list is empty because this is a marginal prior with no
conditioning variables. `getID()` returns the `id="..."` attribute the
XML assigned to the parameter.

------------------------------------------------------------------------

### `sample()`

``` java
    @Override
    public void sample(State state, java.util.Random random) {
        // Prior is evaluated only; sampling is handled by Gibbs operators.
    }
```

`Distribution.sample()` is called by BEAST when initialising the state
from prior draws. Sampling from this prior is instead handled by the
dedicated Gibbs operators (`GenerationTimeVarianceGibbsOperator` and
`GenerationTimeResidualVarianceGibbsOperator`), so this implementation
is intentionally left empty.
