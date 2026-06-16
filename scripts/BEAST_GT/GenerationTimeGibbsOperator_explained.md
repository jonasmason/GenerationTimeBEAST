---
editor_options: 
  markdown: 
    wrap: 72
---

# GenerationTimeGibbsOperator.java explained

## What this file does

The main Gibbs operator. At each step it draws jointly from the exact
posterior full conditional of (β, u) — the intercept and the full vector
of latent phylogenetic effects given the current tree, variance
parameters, and observed tip generation times.

------------------------------------------------------------------------

## Background: the model and the Gibbs step

The model is:

```         
y_i  = β + u_{tip_i} + ε_i      ε_i ~ N(0, σ²_resid)    [observation]
u    ~ Brownian(tree, σ²_brown)                            [Brownian prior]
β    ~ N(μ_β, σ²_β)                                       [intercept prior]
```

The joint full conditional of θ = (β, u₁, u₂, ..., u\_{n-1}) (excluding
the root u which is fixed to 0) is a (2n-1)-dimensional Gaussian. Its
precision matrix is:

```         
C = Q_prior(β) + Q_brownian(tree) + Q_likelihood(tips)
```

where:

-   `Q_prior(β)` adds `1/σ²_β` to `C[0,0]`

-   `Q_brownian` is the Brownian precision matrix restricted to non-root
    nodes

-   `Q_likelihood` adds `1/σ²_resid` to each tip diagonal and the β
    row/column

Sampling from this Gaussian directly requires a full Cholesky solve of a
(2n-1)×(2n-1) system, which is what this operator does.

### The prior-draw trick (Matheron / perturbation method)

Rather than computing the posterior mean explicitly and then adding
noise, the code uses the "prior-draw trick":

1.  **Draw θ\*** from the prior: β\* \~ N(μ_β, σ²_β), u\* \~
    Brownian(tree, σ²_brown)
2.  **Simulate y\***: for each tip i, draw y\**i = β\* + u\**{tip_i} +
    ε\*\_i
3.  **Compute residual**: r_i = y_i − y\*\_i (observed minus simulated)
4.  **Solve**: C δ = Q_likelihood · r where the RHS is Σ_i
    (r_i/σ²_resid) × A_i\^T
5.  **Final draw**: θ = θ\* + δ

This is equivalent to sampling from the exact posterior but avoids
computing the posterior mean explicitly.

### BEAST acceptance accounting

Because θ is drawn from the exact posterior, acceptance probability = 1.
BEAST computes `new_logP − old_logP + Hastings_ratio = 0`. The operator
achieves this by returning `oldGtLogP − newGtLogP` as the Hastings
ratio, which exactly cancels the GT likelihood change that BEAST
computes internally.

------------------------------------------------------------------------

## Line-by-line walkthrough

### Inputs

``` java
    public Input<Tree> treeInput = ...
    public Input<RealParameter> generationTimesInput = ...  // latent u vector, indexed by node nr
    public Input<RealParameter> betaInput = ...             // scalar β
    public Input<RealParameter> brownianVarianceInput = ... // scalar σ²_brown
    public Input<RealParameter> residualVarianceInput = ... // scalar σ²_resid
    public Input<GenerationTimeTraitData> traitDataInput = ...
    public Input<GenerationTimePrecisionBuilder> precisionBuilderInput = ...
    public Input<GenerationTimeLikelihood> gtLikelihoodInput = ...
    public Input<Double> priorBetaMeanInput = ...  (default 0.0)
    public Input<Double> priorBetaVarianceInput = ... (default 1000.0)
    public Input<String> scaleInput = ...          (default "raw")
```

Nine XML inputs. The `generationTimes` parameter has dimension equal to
the total node count; the root entry is fixed to 0 and is never updated.

------------------------------------------------------------------------

### Dimension and bookkeeping fields

``` java
    private int nodeCount;          // total BEAST node count
    private int leafNodeCount;      // number of tips
    private int latentNodeCount;    // nodeCount - 1 (all except root)
    private int thetaCount;         // 1 + latentNodeCount (β plus all latent u)
    private int rootNodeNr;         // BEAST node number of the root
```

The "theta vector" θ = (β, u₁, ..., u\_{n-1}) has length `thetaCount`.
The β component is θ[0]; the latent effect for BEAST node nr is θ[1 +
latentIndexByNodeNr[nr]].

``` java
    private int[] tipLatentIndices;         // tipLatentIndices[i]  = latent index of tip i
    private int[] latentNodeNrs;            // latentNodeNrs[j]     = BEAST node nr of latent index j
    private double[] observedTipValues;     // transformed observed generation times
    private int[] latentIndexByNodeNr;      // latentIndexByNodeNr[nr] = latent index (-1 for root)
    private boolean[] isLatentNode;         // true for every non-root node
```

Two-way mapping between BEAST node numbers and latent vector indices,
needed when copying values into/out of the BEAST `RealParameter`.

------------------------------------------------------------------------

### Latent precision fields

``` java
    private int nLatentTriplets;
    private int[] qLatentRow;    // latent-indexed row of each precision triplet
    private int[] qLatentCol;    // latent-indexed column
    private double[] qLatentValue;
```

The Brownian precision triplets re-indexed to use latent indices (0 to
latentNodeCount-1) instead of BEAST node numbers. Root row/column are
dropped because the root latent value is fixed.

------------------------------------------------------------------------

### Temporary fields - cleaner as local variables, still needs doing

``` java
    private double[] lastGibbsDraw;
    private double[] lastPriorDraw;
    private double[] lastDelta;
    private double[] lastDeltaRhs;
```

------------------------------------------------------------------------

### Version tracking

``` java
    private long seenPrecisionStructureVersion = -1L;
    private long seenPrecisionNumericVersion = -1L;
```

The Gibbs operator tracks which version of the precision builder it last
used. When these differ from the precision builder's current versions,
the operator rebuilds its internal structures. Initialised to -1 (no
version seen yet) so the first call always triggers a build.

------------------------------------------------------------------------

### CSC fields

``` java
    private int cscNonZeros;
    private int[] cscColPtr;
    private int[] cscRowIdx;
    private double[] cscValues;

    private int cscBetaBetaIndex;
    private int[] cscBetaTipIndex;
    private int[] cscTipBetaIndex;
    private int[] cscTipDiagonalIndex;
    private int[] cscLatentPrecisionIndex;
```

The posterior precision matrix C stored in Compressed Sparse Column
(CSC) format. The `*Index` fields are shortcuts into `cscValues` that
allow `updatePosteriorPrecisionCscValues()` to set specific entries
without searching through the CSC structure on every step.

``` java
    private CSparseJSolver csparseJSolver;
```

The Cholesky solver instance, rebuilt whenever the CSC pattern changes.

------------------------------------------------------------------------

### `initAndValidate()`

``` java
    @Override
    public void initAndValidate() {
        // ... validate all inputs: dimensions match, variance positive, etc. ...

        buildNodeBookkeeping();
        buildLatentPrecisionTriplets();
        buildPosteriorPrecisionCscPattern();
        rebuildSparseSolversForCurrentPattern();

        seenPrecisionStructureVersion = precisionBuilder.getStructureVersion();
        seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();
    }
```

The four build steps run in order:

1\. Build node \<-\> latent-index mappings.

2\. Translate precision triplets to latent indices.

3\. Compute the CSC sparsity pattern of C.

4\. Run CSparseJ symbolic analysis (expensive, once per topology).

Finally, record the current precision-builder versions so later calls
can detect changes.

------------------------------------------------------------------------

### `proposal()`

``` java
    @Override
    public double proposal() {

        refreshPrecisionStructuresIfNeeded();
```

Before anything else, check whether the tree has changed since the last
call (BEAST may have run other operators between two calls to this one).

------------------------------------------------------------------------

``` java
        double brownianVariance = brownianVarianceInput.get().getValue();
        double residualVariance = residualVarianceInput.get().getValue();

        if (brownianVariance <= 0.0 || residualVariance <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
```

Guard: return −∞ (reject) if either variance is non-positive. This
should not happen if the variance Gibbs operators are working correctly,
but the guard prevents a crash in the Cholesky factorisation.

------------------------------------------------------------------------

``` java
        lastPriorDraw = drawPriorTheta(brownianVariance);
        lastDeltaRhs = buildDeltaRhs(lastPriorDraw, residualVariance);
        lastDelta = solveGaussianSystem(lastDeltaRhs, brownianVariance, residualVariance);

        lastGibbsDraw = new double[thetaCount];
        for (int i = 0; i < thetaCount; i++) {
            lastGibbsDraw[i] = lastPriorDraw[i] + lastDelta[i];
        }
```

Steps of the prior-draw trick:

1\. θ\* \<- `drawPriorTheta`

2\. RHS \<- `buildDeltaRhs` (computes the likelihood residual weighted
by precision)

3\. δ \<- `solveGaussianSystem` (solves C δ = RHS)

4\. θ = θ\* + δ

------------------------------------------------------------------------

``` java
        double oldGtLogP = gtLikelihoodInput.get().calculateLogP();
```

Records the GT log-probability before the update. This is needed to
compute the Hastings ratio that ensures acceptance.

------------------------------------------------------------------------

``` java
        betaInput.get().setValue(0, lastGibbsDraw[0]);
```

`RealParameter.setValue(index, value)` sets one component of the
parameter vector. The β parameter has dimension 1 so `index = 0`.

------------------------------------------------------------------------

``` java
        RealParameter generationTimes = generationTimesInput.get();
        generationTimes.setValue(rootNodeNr, 0.0);
```

Explicitly enforces the root constraint (u_root = 0). This is a safety
guard: the Gibbs draw never sets the root, but an explicit reset doesn't
hurt.

------------------------------------------------------------------------

``` java
        for (int latentIndex = 0; latentIndex < latentNodeCount; latentIndex++) {
            int nodeNr = latentNodeNrs[latentIndex];
            generationTimes.setValue(nodeNr, lastGibbsDraw[1 + latentIndex]);
        }
```

Writes every non-root latent effect back to the BEAST `RealParameter`.
The mapping `latentNodeNrs[latentIndex]` converts latent indices
(compact 0-based) to BEAST node numbers (which may have gaps if the root
is not the last node).

------------------------------------------------------------------------

``` java
        double newGtLogP = gtLikelihoodInput.get().calculateLogP();
        return oldGtLogP - newGtLogP;
    }
```

Records the new GT log-probability after the update. The returned value
`oldGtLogP − newGtLogP` is the Hastings ratio:

BEAST computes acceptance = (new_total_logP − old_total_logP) +
returned_ratio = (newGtLogP − oldGtLogP) + (oldGtLogP − newGtLogP) = 0

So the proposal is always accepted.

------------------------------------------------------------------------

### `refreshPrecisionStructuresIfNeeded()`

``` java
    private void refreshPrecisionStructuresIfNeeded() {
        GenerationTimePrecisionBuilder precisionBuilder = precisionBuilderInput.get();
        precisionBuilder.refreshFromTreeIfNeeded();

        boolean structureChanged =
                precisionBuilder.getStructureVersion() != seenPrecisionStructureVersion;
        boolean numericChanged =
                precisionBuilder.getNumericVersion() != seenPrecisionNumericVersion;
```

First tells the precision builder to check the tree for changes. Then
checks whether the builder's versions have advanced past what this
operator last saw.

------------------------------------------------------------------------

``` java
        if (structureChanged) {
            buildNodeBookkeeping();
            buildLatentPrecisionTriplets();
            buildPosteriorPrecisionCscPattern();
            rebuildSparseSolversForCurrentPattern();
            seenPrecisionStructureVersion = precisionBuilder.getStructureVersion();
            seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();
            return;
        }

        if (numericChanged) {
            buildLatentPrecisionTriplets();
            seenPrecisionNumericVersion = precisionBuilder.getNumericVersion();
        }
    }
```

-   Topology change -\> full rebuild (all four steps, including
    expensive symbolic Cholesky analysis).

-   Branch-length-only change -\> only rebuild the triplet values (the
    CSC pattern is unchanged; the numeric values will be updated in
    `updatePosteriorPrecisionCscValues` before each solve).

------------------------------------------------------------------------

### `buildNodeBookkeeping()`

``` java
private void buildNodeBookkeeping() {

        Tree tree = treeInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();

        nodeCount = tree.getNodeCount();
        leafNodeCount = tree.getLeafNodeCount();
        rootNodeNr = tree.getRoot().getNr();

        latentNodeCount = nodeCount - 1;
        thetaCount = 1 + latentNodeCount;

        tipLatentIndices = new int[leafNodeCount];
        observedTipValues = new double[leafNodeCount];

        latentNodeNrs = new int[latentNodeCount];
        latentIndexByNodeNr = new int[nodeCount];
        isLatentNode = new boolean[nodeCount];

        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {
            latentIndexByNodeNr[nodeNr] = -1;
            isLatentNode[nodeNr] = false;
        }

        int latentIndex = 0;

        for (int nodeNr = 0; nodeNr < nodeCount; nodeNr++) {

            if (nodeNr == rootNodeNr) {
                continue;
            }

            latentNodeNrs[latentIndex] = nodeNr;
            latentIndexByNodeNr[nodeNr] = latentIndex;
            isLatentNode[nodeNr] = true;

            latentIndex++;
        }

        if (latentIndex != latentNodeCount) {
            throw new IllegalStateException(
                    "Expected " + latentNodeCount +
                            " latent nodes but found " + latentIndex
            );
        }

        for (int i = 0; i < leafNodeCount; i++) {

            Node tip = tree.getNode(i);
            String taxon = tip.getID();
            int nodeNr = tip.getNr();

            if (!isLatentNode[nodeNr]) {
                throw new IllegalStateException("Tip node is unexpectedly not latent: " + nodeNr);
            }

            tipLatentIndices[i] = latentIndexByNodeNr[nodeNr];

            observedTipValues[i] = GenerationTimeTraitTransform.transform(
                    traitData.getTrait(taxon),
                    scaleInput.get()
            );
        }
    }
```

Iterates all BEAST node numbers 0 to `nodeCount-1`, skipping the root,
and assigns a compact latent index to each non-root node. Also fills
`tipLatentIndices` and `observedTipValues` by looking up each tip's
taxon name in the trait data.

Key detail: `tree.getNode(i)` for i \< leafNodeCount returns tip nodes
by BEAST's convention that leaf nodes are numbered first.

------------------------------------------------------------------------

### `buildLatentPrecisionTriplets()`

``` java
    private void buildLatentPrecisionTriplets() {
        int maxTriplets = precisionBuilder.getUniqueTripletCount();
        qLatentRow = new int[maxTriplets];
        qLatentCol = new int[maxTriplets];
        qLatentValue = new double[maxTriplets];
        nLatentTriplets = 0;

        for (int k = 0; k < precisionBuilder.getUniqueTripletCount(); k++) {
            int rowNode = precisionBuilder.getQUniqueRow(k);
            int colNode = precisionBuilder.getQUniqueCol(k);
            double value = precisionBuilder.getQUniqueValue(k);

            if (isLatentNode[rowNode] && isLatentNode[colNode]) {
                qLatentRow[nLatentTriplets] = latentIndexByNodeNr[rowNode];
                qLatentCol[nLatentTriplets] = latentIndexByNodeNr[colNode];
                qLatentValue[nLatentTriplets] = value;
                nLatentTriplets++;
            }
        }
    }
```

Filters the full precision matrix to keep only non-root entries (both
row and column must be latent nodes). Converts BEAST node numbers to
compact latent indices. This step is repeated whenever branch lengths
change because the precision values change even if the sparsity pattern
doesm't.

------------------------------------------------------------------------

### `drawPriorTheta()`

``` java
    private double[] drawPriorTheta(double brownianVariance) {
        double[] theta = new double[thetaCount];
        double betaPriorMean = priorBetaMeanInput.get();
        double betaPriorSd = Math.sqrt(priorBetaVarianceInput.get());
        theta[0] = betaPriorMean + betaPriorSd * Randomizer.nextGaussian();

        Tree tree = treeInput.get();
        simulateBrownianFromNode(tree.getRoot(), 0.0, brownianVariance, theta);

        return theta;
    }
```

Draws β\* from N(μ_β, σ²_β), then simulates the Brownian process
top-down starting at the root (which has latent value 0). The simulation
propagates down the tree recursively.

------------------------------------------------------------------------

### `simulateBrownianFromNode()`

``` java
    private void simulateBrownianFromNode(
            Node node, double parentValue, double brownianVariance, double[] theta
    ) {
        for (int i = 0; i < node.getChildCount(); i++) {
            Node child = node.getChild(i);
            int childNodeNr = child.getNr();
            double branchLength = child.getLength();
            if (branchLength < 0.0) { throw ... }

            double childValue = parentValue
                    + Math.sqrt(brownianVariance * branchLength)
                    * Randomizer.nextGaussian();

            if (isLatentNode[childNodeNr]) {
                int latentIndex = latentIndexByNodeNr[childNodeNr];
                theta[1 + latentIndex] = childValue;
            }

            simulateBrownianFromNode(child, childValue, brownianVariance, theta);
        }
    }
```

For each child of the current node: draws an increment \~ N(0, σ² ×
length) and adds it to the parent value. `Math.sqrt(σ² × length)` is the
standard deviation of the Brownian increment over one branch.
`Randomizer.nextGaussian()` returns a standard normal draw. The child
value is stored and then the recursion continues depth-first.

------------------------------------------------------------------------

### `buildDeltaRhs()`

``` java
    private double[] buildDeltaRhs(double[] priorTheta, double residualVariance) {
        double[] rhs = new double[thetaCount];
        double residualPrecision = 1.0 / residualVariance;

        for (int i = 0; i < leafNodeCount; i++) {
            int latentIndex = tipLatentIndices[i];
            double observedY = observedTipValues[i];
            double simulatedY = priorTheta[0]
                    + priorTheta[1 + latentIndex]
                    + Math.sqrt(residualVariance) * Randomizer.nextGaussian();
            double residual = observedY - simulatedY;

            rhs[0] += residualPrecision * residual;
            rhs[1 + latentIndex] += residualPrecision * residual;
        }
        return rhs;
    }
```

For each tip i:

-    `simulatedY` = β\* + u\*\_{tip_i} + ε\*\_i (simulated observation
    from the prior draw)

-   `residual` = observed - simulated

-   Adds `precision × residual` to both the β component (index 0) and
    the corresponding latent-effect component.

The design matrix A for tip i has A[i, 0] = 1 (for β) and A[i,
1+latentIndex] = 1 (for u\_{tip_i}). So `A^T r` for this tip contributes
`r_i` to positions 0 and 1+latentIndex. Dividing by σ²_resid gives the
right-hand side of C δ = A\^T r / σ².

------------------------------------------------------------------------

### `buildPosteriorPrecisionCscPattern()`

Builds the sparsity pattern of C as a `TreeMap<Long, Boolean>`. A
`TreeMap` in Java keeps entries sorted by key: here the keys encode
(column, row) pairs, so iterating the map visits entries in column-major
order.

``` java
        TreeMap<Long, Boolean> pattern = new TreeMap<>();

        addCscPatternEntry(pattern, 0, 0);   // β-β entry
        for (int i = 0; i < leafNodeCount; i++) {
            int thetaIndex = 1 + tipLatentIndices[i];
            addCscPatternEntry(pattern, 0, thetaIndex);       // β-u_tip off-diagonal
            addCscPatternEntry(pattern, thetaIndex, 0);       // u_tip-β off-diagonal
            addCscPatternEntry(pattern, thetaIndex, thetaIndex);  // u_tip diagonal
        }
        for (int k = 0; k < nLatentTriplets; k++) {
            int thetaRow = 1 + qLatentRow[k];
            int thetaCol = 1 + qLatentCol[k];
            addCscPatternEntry(pattern, thetaRow, thetaCol);   // Brownian block entries
        }
```

Three groups of pattern entries:

1\. The (0,0) β-β entry.

2\. For each tip: two off-diagonal β-u entries and the tip diagonal.

3\. For each Brownian precision triplet: the (row, col) entry.

Entries that appear multiple times (e.g., a tip node's diagonal appears
once in group 2 and possibly once in group 3) are deduplicated
automatically by the `TreeMap`.

After the pattern is built, it is converted to CSC arrays and the
`*Index` shortcut fields are populated using `getCscIndex()`.

------------------------------------------------------------------------

### `updatePosteriorPrecisionCscValues()`

``` java
    private void updatePosteriorPrecisionCscValues(
            double brownianVariance, double residualVariance
    ) {
        Arrays.fill(cscValues, 0.0);   // reset all values to zero

        double invResid = 1.0 / residualVariance;
        double invPhylo = 1.0 / brownianVariance;
        double priorBetaVariance = priorBetaVarianceInput.get();

        cscValues[cscBetaBetaIndex] +=
                leafNodeCount * invResid + 1.0 / priorBetaVariance;

        for (int i = 0; i < leafNodeCount; i++) {
            cscValues[cscBetaTipIndex[i]] += invResid;
            cscValues[cscTipBetaIndex[i]] += invResid;
            cscValues[cscTipDiagonalIndex[i]] += invResid;
        }

        for (int k = 0; k < nLatentTriplets; k++) {
            cscValues[cscLatentPrecisionIndex[k]] += invPhylo * qLatentValue[k];
        }
    }
```

Fills numeric values into the pre-built CSC array. The `*Index` fields
are pre-computed integer offsets into `cscValues`, so no searching is
needed here. This method runs every MCMC step, so should be fast.

-   β-β entry: n_tips/σ²_resid (likelihood contribution) + 1/σ²_β
    (prior).
-   Off-diagonal β-u_tip and u_tip-β: 1/σ²_resid each.
-   Tip diagonal: 1/σ²_resid (likelihood) + Q\_{tip,tip}/σ²_brown
    (Brownian prior).
-   Brownian off-diagonal: Q\_{i,j}/σ²_brown.

`Arrays.fill(array, value)` is Java's standard idiom for zero-filling an
array.

------------------------------------------------------------------------

### `solveGaussianSystem()`

``` java
    private double[] solveGaussianSystem(
            double[] rhs, double brownianVariance, double residualVariance
    ) {
        updatePosteriorPrecisionCscValues(brownianVariance, residualVariance);
        csparseJSolver.factorize(cscValues);
        return csparseJSolver.solve(rhs);
    }
```

Three steps: fill the numeric values, numerically factorise C = L L\^T,
then solve C δ = rhs. The symbolic analysis (pattern) was computed once
by `rebuildSparseSolversForCurrentPattern()` and is reused here.

------------------------------------------------------------------------

### CSC key helpers

``` java
    private long cscKey(int row, int column) {
        return (((long) column) << 32) | (row & 0xffffffffL);
    }
    private int cscColumnFromKey(long key) { return (int) (key >>> 32); }
    private int cscRowFromKey(long key) { return (int) key; }
```

Packs (column, row) into a 64-bit long with column in the upper 32 bits
and row in the lower 32 bits. When sorted in ascending order, entries
are sorted first by column then by row, the order needed to build CSC
format. This is the same technique as `tripletKey` in
`GenerationTimePrecisionBuilder`, but with column and row swapped.
