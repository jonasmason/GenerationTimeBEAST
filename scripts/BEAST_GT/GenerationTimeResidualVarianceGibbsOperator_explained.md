---
editor_options: 
  markdown: 
    wrap: 72
---

# GenerationTimeResidualVarianceGibbsOperator.java explained

## What this file does

A Gibbs operator for the residual variance σ²_residual. At each step it
draws a new residual variance from its exact inverse-gamma full
conditional given the current latent effects u, intercept β, and
observed tip generation times.

------------------------------------------------------------------------

## Background: the full conditional

The observation model is:

```         
y_i = β + u_{tip_i} + ε_i,   ε_i ~ N(0, σ²_resid)
```

Given β, u, and y, the sufficient statistic is the residual sum of
squares:

```         
RSS = Σ_i (y_i − β − u_{tip_i})²
```

With an inverse-gamma prior σ²_resid \~ IG(α, β_prior), the full
conditional is:

```         
p(σ²_resid | β, u, y) = IG(α + n/2,  β_prior + RSS/2)
```

where n is the number of tips.

------------------------------------------------------------------------

## Comparison with `GenerationTimeVarianceGibbsOperator`

The structure is nearly identical to the Brownian variance operator:

1\. Compute a sufficient statistic (RSS here; quadratic form there).

2\. Update the inverse-gamma parameters.

3\. Draw from the full conditional.

4\. Return the log Hastings ratio.

The only structural differences are:

-   The sufficient statistic is RSS over tips rather than u\^T Q u over
    edges.

-   This operator calls `gtLikelihood.calculateLogP()` at the start and
    end (the Brownian variance operator doesn't).

-   This operator needs the tree, trait data, and the β parameter; the
    Brownian variance operator does not.

------------------------------------------------------------------------

## Line-by-line walkthrough

### Inputs

``` java
    public Input<Tree> treeInput = ...
    public Input<RealParameter> generationTimesInput = ...
    public Input<RealParameter> betaInput = ...
    public Input<RealParameter> residualVarianceInput = ...
    public Input<GenerationTimeTraitData> traitDataInput = ...
    public Input<GenerationTimeLikelihood> gtLikelihoodInput = ...
    public Input<Double> priorAlphaInput = ...   (default 0.001)
    public Input<Double> priorBetaInput = ...    (default 0.001)
    public Input<String> scaleInput = ...        (default "raw")
```

Eight inputs. The `gtLikelihood` input is actively used here (unlike in
the Brownian variance operator).

------------------------------------------------------------------------

### Fields

``` java
    protected int leafCount;
    protected int[] tipNodeNrs;
    protected double[] observedTipValues;
```

Same pattern as in `GenerationTimeLikelihood`: caches the node numbers
and transformed observed values for tips so they are not re-fetched
every step.

------------------------------------------------------------------------

### `initAndValidate()`

``` java
    @Override
    public void initAndValidate() {
        // ... validate all inputs ...
        leafCount = tree.getLeafNodeCount();

        tipNodeNrs = new int[leafCount];
        observedTipValues = new double[leafCount];

        for (int i = 0; i < leafCount; i++) {
            Node tip = tree.getNode(i);
            String taxon = tip.getID();
            if (!traitData.hasTrait(taxon)) { throw ... }
            tipNodeNrs[i] = tip.getNr();
            observedTipValues[i] = GenerationTimeTraitTransform.transform(
                    traitData.getTrait(taxon), scale
            );
        }
    }
```

Fills `tipNodeNrs` and `observedTipValues` once at startup. The loop
iterates i from 0 to leafCount-1, using BEAST's convention that leaf
nodes are numbered first. `tip.getNr()` returns the integer node number
that BEAST uses internally to index into the `generationTimes`
parameter.

------------------------------------------------------------------------

### `proposal()`

``` java
    @Override
    public double proposal() {
        RealParameter generationTimes = generationTimesInput.get();
        RealParameter betaParameter = betaInput.get();
        RealParameter residualVariance = residualVarianceInput.get();
```

Retrieves the current state via BEAST's `Input.get()`.

------------------------------------------------------------------------

``` java
        /*
         * Force the GT likelihood to synchronise with the current BEAST state before
         * constructing the residual-variance full conditional.
         */
        gtLikelihoodInput.get().calculateLogP();
```

Calls `calculateLogP()` on the GT likelihood before computing the RSS.
This ensures BEAST's internal state is fully synched. In particular, any
dirty flags set by preceding operators are cleared. This defensive call
is the reason `gtLikelihoodInput` exists in this class (contrast with
`GenerationTimeVarianceGibbsOperator`, which declares the input but
never calls it -\> dead-code, still needs editing).

------------------------------------------------------------------------

``` java
        double beta = betaParameter.getValue();
        double oldVariance = residualVariance.getValue();
```

`getValue()` on a scalar (dimension-1) `RealParameter` returns its
single component as a primitive `double`.

------------------------------------------------------------------------

``` java
        double residualSumOfSquares = residualSumOfSquares(
                generationTimes, beta
        );
```

Calls the helper below to compute RSS = Σ_i (y_i − β − u\_{tip_i})².

------------------------------------------------------------------------

``` java
        double priorAlpha = priorAlphaInput.get();
        double priorBeta = priorBetaInput.get();
        double posteriorAlpha = priorAlpha + 0.5 * leafCount;
        double posteriorBeta = priorBeta + 0.5 * residualSumOfSquares;
```

Conjugate update of the inverse-gamma parameters: - Posterior shape =
prior shape + n/2 (n = number of tips) - Posterior scale = prior scale +
RSS/2

------------------------------------------------------------------------

``` java
        double oldConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
                oldVariance, posteriorAlpha, posteriorBeta
        );
```

Log density of the current variance under the posterior full
conditional. Needed for the Hastings ratio.

------------------------------------------------------------------------

``` java
        double varianceDraw = GenerationTimeMathUtils.sampleInverseGamma(
                posteriorAlpha, posteriorBeta
        );

        if (varianceDraw <= 0.0 ||
                Double.isNaN(varianceDraw) ||
                Double.isInfinite(varianceDraw)) {
            return Double.NEGATIVE_INFINITY;
        }
```

Draws the new variance. The guard returns −∞ to reject gracefully on
numerical failure.

------------------------------------------------------------------------

``` java
        double newConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
                varianceDraw, posteriorAlpha, posteriorBeta
        );

        double logHastingsRatio = oldConditionalLogDensity - newConditionalLogDensity;
```

Hastings ratio for the Gibbs step. As in the Brownian variance operator,
this makes the total acceptance = 0 (always accept).

------------------------------------------------------------------------

``` java
        residualVariance.setValue(0, varianceDraw);

        gtLikelihoodInput.get().calculateLogP();

        return logHastingsRatio;
    }
```

Writes the new variance. The second `calculateLogP()` call updates the
GT likelihood's internal cache to reflect the new variance. This ensures
BEAST's bookkeeping is consistent with the new state. Finally, the
Hastings ratio is returned.

------------------------------------------------------------------------

### `residualSumOfSquares()`

``` java
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
```

Computes RSS = Σ_i (y_i − β − u\_{tip_i})² by iterating over all tips.
For each tip: - `tipNodeNrs[i]` gives the BEAST node number of tip i. -
`generationTimes.getValue(nodeNr)` reads u\_{tip_i} from the current
state. - `fitted = β + u_{tip_i}` is the model's prediction at this
tip. - `residual = y_i − fitted`. - `residual * residual` is added to
the running sum.

This method is `protected` (not `private`) so that tests or subclasses
can invoke it directly.
