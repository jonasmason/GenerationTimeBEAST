package gentime;

import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;

/**
 * Scale-invariant variant of {@link GenerationTimeLikelihood} for use under the alpha=0
 * (fix_base_rate) timescale, where the tree height is ESTIMATED in substitution units.
 *
 * In the parent class the Brownian field density uses raw branch lengths, so the field
 * variance parameter's meaning is tied to the absolute tree scale:
 *
 *     increment variance on edge e = brownianVariance * length_e
 *
 * When the tree height H is a free parameter (alpha=0), length_e scales with H, so what the
 * data constrain is brownianVariance * H, not brownianVariance alone. Because the clock uses
 * slope * field, this couples the slope to brownianVariance and creates the slope<->variance
 * ridge observed in the alpha=0 runs.
 *
 * This class removes the coupling by defining the field on a UNIT-HEIGHT tree: the effective
 * branch lengths are length_e / H, so the variance parameter becomes scale-invariant and now
 * directly represents the tip-level phylogenetic variance (matching the IG(4, 2.4) prior mean).
 *
 * Derivation: the unit-height density with parameter s2 equals the parent density evaluated
 * with effective variance v = s2 / H (all H-dependent normalising terms included). So the only
 * change needed is to pass brownianVariance / H into the inherited density.
 *
 * Uses the SAME (raw) precision builder as the parent; nothing else changes.
 */
public class GenerationTimeUnitHeightLikelihood extends GenerationTimeLikelihood {

    @Override
    protected double brownianLatentLogLikelihood(
            Tree tree,
            RealParameter generationTimes,
            double brownianVariance
    ) {
        double h = tree.getRoot().getHeight();

        if (h <= 0.0 || Double.isNaN(h) || Double.isInfinite(h)) {
            return Double.NEGATIVE_INFINITY;
        }

        // v = s2 / H  ->  parent density with v == unit-height density with s2
        return super.brownianLatentLogLikelihood(
                tree,
                generationTimes,
                brownianVariance / h
        );
    }
}
