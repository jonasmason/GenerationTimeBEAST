package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;

/**
 * Scale-invariant variant of {@link GenerationTimeVarianceGibbsOperator} that MUST be paired
 * with {@link GenerationTimeUnitHeightLikelihood} under the alpha=0 timescale.
 *
 * The parent Gibbs step draws brownianVariance from its inverse-gamma full conditional using
 * the sufficient statistic SS = u^T Q u formed from the (raw) precision builder:
 *
 *     brownianVariance | u ~ IG(alpha + rank/2, beta + 0.5 * SS)
 *
 * Under the unit-height parameterisation the effective precision is H * Q (branch lengths are
 * length_e / H), so the correct sufficient statistic is H * SS and the full conditional is:
 *
 *     s2 | u ~ IG(alpha + rank/2, beta + 0.5 * H * SS)
 *
 * We therefore override the quadratic form to return H * (raw quadratic form). Everything else
 * in the parent proposal (rank, prior, IG draw, Hastings ratio) is reused unchanged, and the
 * draw stays consistent with GenerationTimeUnitHeightLikelihood.
 */
public class GenerationTimeUnitHeightVarianceGibbsOperator
        extends GenerationTimeVarianceGibbsOperator {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose root height normalises the field variance (unit-height scheme).",
            Input.Validate.REQUIRED
    );

    @Override
    protected double sparseQuadraticForm(
            RealParameter generationTimes,
            GenerationTimePrecisionBuilder precisionBuilder
    ) {
        double h = treeInput.get().getRoot().getHeight();

        double raw = super.sparseQuadraticForm(generationTimes, precisionBuilder);

        return h * raw;
    }
}
