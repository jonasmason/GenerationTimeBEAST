---
editor_options: 
  markdown: 
    wrap: 72
---

# GenerationTimeLikelihood.java explained

## What this file does

Computes the complete log-probability of the generation-time model. This
is the quantity BEAST needs to evaluate when deciding whether to accept
or reject any MCMC proposal that touches generation-time parameters. It
extends BEAST's `Distribution` class so BEAST knows to include it in the
posterior sum.

The model this class evaluates is:

```         
y_i  = β + u_tip_i + ε_i,      ε_i ~ N(0, σ²_resid)     [observation]
u    ~ BrownianMotion(tree, σ²_brownian)                    [Brownian latent]
β    ~ N(μ_β, σ²_β)                                        [intercept prior]
```

where y_i is the observed generation time for tip i, u_tip_i is the
latent phylogenetic effect at that tip, and β is an intercept.

`calculateLogP()` returns the sum of three terms:

1\. The log prior on β.

2\. The log likelihood of observed tip values given (β, u, σ²_resid).

3\. The log density of the latent Brownian process given (u, tree,
σ²_brownian).

------------------------------------------------------------------------

## Line-by-line walkthrough

### Imports and class declaration

``` java
import beast.base.inference.Distribution;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;
```

`Distribution` is the base class; `State` is needed only for the
`sample()` signature; `RealParameter` is BEAST's type for real-valued
parameters.

``` java
public class GenerationTimeLikelihood extends Distribution {
```

Declares this as a BEAST `Distribution` so it is included in the
posterior.

------------------------------------------------------------------------

### Inputs

``` java
    public Input<Tree> treeInput = ...
    public Input<GenerationTimeTraitData> traitDataInput = ...
    public Input<RealParameter> generationTimesInput = ...
    public Input<RealParameter> betaInput = ...
    public Input<RealParameter> brownianVarianceInput = ...
    public Input<RealParameter> residualVarianceInput = ...
    public Input<GenerationTimePrecisionBuilder> precisionBuilderInput = ...
    public Input<Double> priorBetaMeanInput = ...   (default 0.0)
    public Input<Double> priorBetaVarianceInput = ... (default 1000.0)
    public Input<String> scaleInput = ...           (default "raw")
```

All the quantities this distribution depends on. The
`precisionBuilderInput` is present primarily to wire the computation
graph correctly; the likelihood itself does not use the sparse precision
matrix — it computes the Brownian log density by iterating over edges
directly.

------------------------------------------------------------------------

### Fields

``` java
    protected int leafCount;
    protected int rootNodeNr;
    protected int[] tipNodeNrs;
    protected double[] observedTipValues;
```

`protected` (not `private`) so that any subclass can access these in a
diagnostic/test override. `tipNodeNrs[i]` stores the BEAST node number
for tip i, and `observedTipValues[i]` stores its (possibly
log-transformed) observed generation time. These are filled once in
`initAndValidate()` and then read many times per MCMC step.

------------------------------------------------------------------------

### `initAndValidate()`

``` java
    @Override
    public void initAndValidate() {
        // ... retrieve inputs, validate dimensions ...
        leafCount = tree.getLeafNodeCount();
        rootNodeNr = tree.getRoot().getNr();
```

`@Override` is a Java annotation telling the compiler to override a
method from the parent class. The compiler then verifies the signature
matches; if it doesn't, a compile-time error is generated (useful for
catching typos in method names).

Caches counts that won't change while the topology is fixed. `getNr()`
returns the integer node number that BEAST uses internally to index
nodes.

------------------------------------------------------------------------

``` java
        for (int i = 0; i < leafCount; i++) {
            Node tip = tree.getNode(i);
            String taxon = tip.getID();
            // ...
            tipNodeNrs[i] = tip.getNr();
            observedTipValues[i] = GenerationTimeTraitTransform.transform(
                    traitData.getTrait(taxon), scale
            );
        }
```

In BEAST 2, leaf nodes are conventionally numbered 0 to leafCount−1. For
each tip we record its BEAST node number (to look up latent values
later) and its transformed observed value (transformed once at startup,
not on every step).

------------------------------------------------------------------------

### `calculateLogP()`

Called by BEAST on every step where any dependency changes.

``` java
    @Override
    public double calculateLogP() {

        Tree tree = treeInput.get();
        // ...
        int currentRootNodeNr = tree.getRoot().getNr();
```

The root node number can change when the tree topology changes (root may
be renumbered), so it is retrieved fresh every call rather than relying
on the cached `rootNodeNr`.

------------------------------------------------------------------------

``` java
        if (brownianVariance <= 0.0 || residualVariance <= 0.0) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }
```

A hard constraint: variance parameters must be positive. If either is
non-positive (e.g., due to a rogue operator proposal), return −∞ so the
proposal is rejected.

------------------------------------------------------------------------

``` java
        if (Math.abs(generationTimes.getValue(currentRootNodeNr)) > 1e-10) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }
```

The root latent value is constrained to zero (to anchor the
non-identifiable Brownian process). If the Gibbs operator or a restore
left it non-zero, return −∞. The tolerance 1e-10 guards against
floating-point rounding.

------------------------------------------------------------------------

``` java
        double betaLogP = betaLogLikelihood(beta);
        double tipLogP = tipObservationLogLikelihood(...);
        double brownianLogP = brownianLatentLogLikelihood(...);
        logP = betaLogP + tipLogP + brownianLogP;
        return logP;
    }
```

Sum three independent contributions. These are separated into named
methods for clarity and testability.

------------------------------------------------------------------------

### `betaLogLikelihood()`

``` java
    protected double betaLogLikelihood(double beta) {
        double priorMean = priorBetaMeanInput.get();
        double priorVariance = priorBetaVarianceInput.get();
        double residual = beta - priorMean;
        return -0.5 * Math.log(2.0 * Math.PI * priorVariance)
                -0.5 * residual * residual / priorVariance;
    }
```

Log density of N(beta; priorMean, priorVariance):

```         
log p(β) = -½ log(2π σ²_β) - (β - μ_β)² / (2 σ²_β)
```

------------------------------------------------------------------------

### `tipObservationLogLikelihood()`

``` java
    protected double tipObservationLogLikelihood(...) {

        double total = 0.0;
        double logNormalConstant = -0.5 * Math.log(2.0 * Math.PI * residualVariance);

        for (int i = 0; i < leafCount; i++) {
            int nodeNr = tipNodeNrs[i];
            double observed = observedTipValues[i];
            double latentU = generationTimes.getValue(nodeNr);
            double fitted = beta + latentU;
            double residual = observed - fitted;
            total += logNormalConstant - 0.5 * residual * residual / residualVariance;
        }
        return total;
    }
```

Sums log N(y_i; β + u_tip_i, σ²_resid) over all tips. The term
`-½ log(2π σ²)`is precomputed outside the loop (as it's constant across
tips) to avoid recomputing a logarithm on every iteration.
`generationTimes.getValue(nodeNr)` fetches the current latent effect at
that node from BEAST's state vector.

------------------------------------------------------------------------

### `brownianLatentLogLikelihood()`

``` java
    protected double brownianLatentLogLikelihood(
            Tree tree,
            RealParameter generationTimes,
            double brownianVariance
    ) {
        double quadraticForm = edgeIncrementQuadraticForm(tree, generationTimes);

        if (Double.isInfinite(quadraticForm) || Double.isNaN(quadraticForm)) {
            return Double.NEGATIVE_INFINITY;
        }
```

The Brownian log density has a quadratic form u\^T Q u / σ² where Q is
the Brownian precision matrix. Because Q = (branch-length)\^{-1} on each
edge, the quadratic form decomposes edge-by-edge as Σ_e (u_child -
u_parent)² / length_e.

------------------------------------------------------------------------

``` java
        double logBranchLengthSum = 0.0;

        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            if (node.isRoot()) { continue; }
            double branchLength = node.getLength();
            if (branchLength <= 0.0 || ...) { return Double.NEGATIVE_INFINITY; }
            logBranchLengthSum += Math.log(branchLength);
        }
```

Computes Σ_e log(branch_e). This appears in the log-determinant of the
Brownian covariance matrix: log\|Σ_brownian\| = (n-1) log σ² + Σ_e
log(length_e). Checking for non-positive or non-finite branch lengths is
a defensive guard.

------------------------------------------------------------------------

``` java
        int brownianRank = tree.getNodeCount() - 1;

        return -0.5 * brownianRank * Math.log(2.0 * Math.PI * brownianVariance)
                -0.5 * logBranchLengthSum
                -0.5 * quadraticForm / brownianVariance;
    }
```

Full log-density of the Brownian process (n−1 non-trivial dimensions,
root fixed to zero):

```         
log p(u | σ², tree) = -½(n-1)log(2π σ²) - ½ Σ log(length_e)
                      - ½ u^T Q u / σ²
```

`brownianRank = n − 1` is the number of non-root nodes (the effective
rank of the Brownian precision matrix).

------------------------------------------------------------------------

### `edgeIncrementQuadraticForm()`

``` java
    protected double edgeIncrementQuadraticForm(Tree tree, RealParameter generationTimes) {
        double total = 0.0;
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            if (node.isRoot()) { continue; }
            Node parent = node.getParent();
            double branchLength = node.getLength();
            if (branchLength <= 0.0 || ...) { return Double.POSITIVE_INFINITY; }
            double childValue = generationTimes.getValue(node.getNr());
            double parentValue = generationTimes.getValue(parent.getNr());
            double difference = childValue - parentValue;
            total += difference * difference / branchLength;
        }
        return total;
    }
```

Computes u\^T Q u = Σ_e (u_child - u_parent)² / length_e directly from
the tree structure. Each edge e with parent node p and child node c
contributes (u_c - u_p)² × (1/length_e). Returns +∞ for an invalid
branch length (signals −∞ log-density to the caller).

------------------------------------------------------------------------

### `diagnosticSummary()`

``` java
    public String diagnosticSummary(String label) { ... }
```

A debugging helper that returns a single formatted string summarising
all likelihood components, parameter values, and tree statistics (called
optionally).

------------------------------------------------------------------------

### `getArguments()`, `getConditions()`, `sample()`

``` java
    @Override
    public List<String> getArguments() {
        List<String> arguments = new ArrayList<>();
        arguments.add(treeInput.get().getID());
        arguments.add(generationTimesInput.get().getID());
        arguments.add(betaInput.get().getID());
        arguments.add(brownianVarianceInput.get().getID());
        arguments.add(residualVarianceInput.get().getID());
        return arguments;
    }

    @Override
    public List<String> getConditions() {
        List<String> conditions = new ArrayList<>();
        conditions.add(traitDataInput.get().getID());
        return conditions;
    }

    @Override
    public void sample(State state, Random random) {
        // not used
    }
```

`getArguments()` lists the state nodes that are the "random variables"
of this distribution (tree, latent effects, β, and both variances).
`getConditions()` lists the observed data, which is fixed and conditions
the distribution. `sample()` is empty because sampling is handled by the
Gibbs operators.
