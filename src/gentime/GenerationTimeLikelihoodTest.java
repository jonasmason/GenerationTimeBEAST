package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;

public class GenerationTimeLikelihoodTest extends GenerationTimeLikelihood {

    public Input<Boolean> debugInput = new Input<>(
            "debug",
            "Print likelihood diagnostics when true.",
            false
    );

    @Override
    public void initAndValidate() {

        super.initAndValidate();

        if (!debugInput.get()) {
            return;
        }

        Tree tree = treeInput.get();
        GenerationTimeTraitData traitData = traitDataInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        String scale = scaleInput.get();

        System.out.println("GenerationTimeLikelihoodTest initialised.");
        System.out.println("Model: y_tip ~ Normal(beta + u_tip, residualVariance)");
        System.out.println("Prior: beta ~ Normal(priorBetaMean, priorBetaVariance)");
        System.out.println("Prior: u ~ Brownian(0, brownianVariance), root u fixed to 0");
        System.out.println("Tree node count = " + tree.getNodeCount());
        System.out.println("Leaf count = " + leafCount);
        System.out.println("Root node number = " + rootNodeNr);
        System.out.println("generationTimes dimension = " + generationTimes.getDimension());

        for (int i = 0; i < leafCount; i++) {

            Node tip = tree.getNode(i);
            String taxon = tip.getID();

            System.out.println(
                    "Tip node " + i +
                            " taxon=" + taxon +
                            " observedGT=" + traitData.getTrait(taxon) +
                            " transformedObservedGT=" +
                            GenerationTimeTraitTransform.transform(
                                    traitData.getTrait(taxon),
                                    scale
                            ) +
                            " latentU=" + generationTimes.getValue(tip.getNr())
            );
        }
    }

    @Override
    public double calculateLogP() {

        double value = super.calculateLogP();

        if (!debugInput.get()) {
            return value;
        }

        Tree tree = treeInput.get();
        RealParameter generationTimes = generationTimesInput.get();

        double beta = betaInput.get().getValue();
        double brownianVariance = brownianVarianceInput.get().getValue();
        double residualVariance = residualVarianceInput.get().getValue();

        double betaLogP = betaLogLikelihood(beta);

        double tipLogP = tipObservationLogLikelihood(
                tree,
                traitDataInput.get(),
                generationTimes,
                beta,
                residualVariance
        );

        double brownianLogP = brownianLatentLogLikelihood(
                tree,
                generationTimes,
                brownianVariance
        );

        System.out.println("GenerationTimeLikelihoodTest diagnostics:");
        System.out.println("beta = " + beta);
        System.out.println("brownianVariance = " + brownianVariance);
        System.out.println("residualVariance = " + residualVariance);
        System.out.println("betaLogP = " + betaLogP);
        System.out.println("tipLogP = " + tipLogP);
        System.out.println("brownianLogP = " + brownianLogP);
        System.out.println("totalLogP = " + value);

        return value;
    }
}