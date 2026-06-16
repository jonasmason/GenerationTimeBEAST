---
editor_options: 
  markdown: 
    wrap: 72
---

# GenerationTimeVarianceGibbsOperator.java explained

## What this file does

A Gibbs operator for the Brownian variance σ²_brownian. At each step it
draws a new variance from its exact inverse-gamma full conditional given
the current latent effects u and the tree.

------------------------------------------------------------------------

## Background: the full conditional

The Brownian prior on the latent effects is:

```         
p(u | σ²_brown, tree) ∝ (σ²_brown)^{-(n-1)/2}
                          exp(−u^T Q u / (2 σ²_brown))
```

where Q is the Brownian precision matrix (at unit variance) and n−1 is
the number of non-root nodes (the "rank" of the Brownian process). With
an inverse-gamma prior σ²_brown \~ IG(α, β), the full conditional is:

```         
p(σ²_brown | u, tree) = IG(α + (n-1)/2,  β + u^T Q u / 2)
```

This is a conjugate update: shape and scale simply absorb sufficient
statistics from the data.

The Hastings ratio for this Gibbs step is computed analytically: log
q(old \| new) / q(new \| old) = log p(old \| full cond) − log p(new \|
full cond)

When BEAST adds the log posterior ratio, the acceptance probability
becomes 0 (always accept), as expected for a Gibbs move.

------------------------------------------------------------------------

## Line-by-line walkthrough

### Imports and class declaration

``` java
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeVarianceGibbsOperator extends Operator {
```

`Operator` is BEAST's base class for all MCMC operators.
`extends Operator` means BEAST knows to call `proposal()` on this class
and to include it in the operator schedule.

------------------------------------------------------------------------

### Inputs

``` java
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
```

**Note:** `gtLikelihoodInput` is declared REQUIRED and wired in the XML
but is never actually called anywhere in `proposal()`, this will still
need to be addressed (dead code, either the call should be added or the
input removed).

``` java
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
```

Hyperparameters of the inverse-gamma prior. Default values of 0.001 give
a weakly informative (nearly flat) prior.

------------------------------------------------------------------------

### `initAndValidate()`

``` java
    @Override
    public void initAndValidate() {
        RealParameter generationTimes = generationTimesInput.get();
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        if (generationTimes.getDimension() != precisionBuilder.getNodeCount()) {
            throw new IllegalArgumentException(
                    "generationTimes dimension must match precision-builder node count."
            );
        }
        if (brownianVarianceInput.get().getDimension() != 1) { throw ... }
        if (priorAlphaInput.get() <= 0.0 || priorBetaInput.get() <= 0.0) { throw ... }
    }
```

Validates that `generationTimes` has the same number of components as
there are nodes in the tree (one latent value per node, including the
root which is fixed to zero), and that the variance is scalar and the
hyperparameters are positive.

------------------------------------------------------------------------

### `proposal()`

``` java
    @Override
    public double proposal() {
        RealParameter generationTimes = generationTimesInput.get();
        RealParameter brownianVariance = brownianVarianceInput.get();
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();

        precisionBuilder.refreshFromTreeIfNeeded();
```

Ensures the precision builder is up to date with the current tree before
computing the quadratic form. Unlike `GenerationTimeGibbsOperator`, this
operator checks the tree through the precision builder rather than
through its own bookkeeping.

------------------------------------------------------------------------

``` java
        double oldVariance = brownianVariance.getValue();
```

Stores the current variance before the draw. This is used to compute the
Hastings ratio (old conditional density / new conditional density).

------------------------------------------------------------------------

``` java
        double quadraticFormTriplet = sparseQuadraticForm(
                generationTimes, precisionBuilder
        );
```

Computes u\^T Q u, where Q is the Brownian precision matrix at unit
variance (i.e., σ² = 1). The full quadratic form is u\^T (Q/σ²) u =
(u\^T Q u) / σ², but we factor out σ² and absorb it into the posterior
scale parameter.

------------------------------------------------------------------------

``` java
        int brownianRank = precisionBuilder.getEdgeCount();
```

`edgeCount = nodeCount − 1` is the effective rank of the Brownian
precision matrix (the root is pinned, so there are n−1 free parameters).
This is the "n−1" in the posterior shape parameter.

------------------------------------------------------------------------

``` java
        double priorAlpha = priorAlphaInput.get();
        double priorBeta = priorBetaInput.get();
        double posteriorAlpha = priorAlpha + 0.5 * brownianRank;
        double posteriorBeta = priorBeta + 0.5 * quadraticFormTriplet;
```

Conjugate update: - Posterior shape = prior shape + (n−1)/2 - Posterior
scale = prior scale + (u\^T Q u) / 2

This is the closed-form update for the inverse-gamma conjugate prior
given a Gaussian likelihood with known mean (zero in Brownian motion
centred at root).

------------------------------------------------------------------------

``` java
        double oldConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
               oldVariance, posteriorAlpha, posteriorBeta
        );
```

Log density of the current (old) variance under the posterior full
conditional. Needed to compute the Hastings ratio.

------------------------------------------------------------------------

``` java
        double varianceDraw = GenerationTimeMathUtils.sampleInverseGamma(
               posteriorAlpha, posteriorBeta
        );
```

Draws a new variance from the exact full conditional (= the Gibbs draw).

------------------------------------------------------------------------

``` java
        if (varianceDraw <= 0.0 ||
                Double.isNaN(varianceDraw) ||
                Double.isInfinite(varianceDraw)) {
            return Double.NEGATIVE_INFINITY;
        }
```

The inverse-gamma sampler should always return a positive finite value,
but we guard against numerical edge cases. Returning −∞ causes BEAST to
reject the proposal harmlessly.

------------------------------------------------------------------------

``` java
        double newConditionalLogDensity = GenerationTimeMathUtils.inverseGammaLogDensity(
               varianceDraw, posteriorAlpha, posteriorBeta
        );

        double logHastingsRatio = oldConditionalLogDensity - newConditionalLogDensity;
```

The Hastings ratio for a Gibbs step with proposal distribution = full
conditional is:

```         
log q(old | new) / q(new | old) = log p(old | full cond) − log p(new | full cond)
```

This makes the total acceptance probability exactly 0 (always accept),
because BEAST's computation adds: (new total logP − old total logP) +
logHastingsRatio = 0

------------------------------------------------------------------------

``` java
        brownianVariance.setValue(0, varianceDraw);

        return logHastingsRatio;
    }
```

Writes the new variance into the BEAST state and returns the Hastings
ratio. BEAST will automatically recalculate all affected distributions
(GT likelihood, Brownian variance prior) when evaluating the acceptance
probability.

------------------------------------------------------------------------

### `sparseQuadraticForm()`

``` java
    protected double sparseQuadraticForm(
            RealParameter generationTimes,
            GenerationTimePrecisionBuilder precisionBuilder
    ) {
        double total = 0.0;
        for (int k = 0; k < precisionBuilder.getUniqueTripletCount(); k++) {
            int row = precisionBuilder.getQUniqueRow(k);
            int col = precisionBuilder.getQUniqueCol(k);
            double value = precisionBuilder.getQUniqueValue(k);
            total += generationTimes.getValue(row) * value * generationTimes.getValue(col);
        }
        return total;
    }
```

Computes u\^T Q u as Σ_k u\_{row_k} × Q\_{row_k, col_k} × u\_{col_k}
where the sum is over unique triplets of Q. This is the sparse
matrix-vector-matrix product for a symmetric matrix stored in triplet
form.

Note: the triplets include both (i,j) and (j,i) entries with the same
value, so the off-diagonal contributions are counted twice (once for
each symmetric triplet). This is correct because the full matrix Q
includes both symmetric entries.

------------------------------------------------------------------------

### `edgeIncrementQuadraticForm()` (not used in `proposal()`)

``` java
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
```

An alternative computation of u\^T Q u by iterating over edges directly:
u\^T Q u = Σ_e (u_child − u_parent)² / length_e. This is equivalent to
`sparseQuadraticForm` (both compute the same quantity, just from
different representations of Q). This method is available for
diagnostics and testing but is not called automatically. The
`proposal()` uses `sparseQuadraticForm` because the precision builder
already has the unique triplets computed.
