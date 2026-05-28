package gentime;

import beast.base.core.Input;

public class GenerationTimePrecisionBuilderTest extends GenerationTimePrecisionBuilder {

    public Input<Boolean> debugInput = new Input<>(
            "debug",
            "Print sparse precision builder diagnostics when true.",
            false
    );

    @Override
    public void initAndValidate() {

        super.initAndValidate();

        if (!debugInput.get()) {
            return;
        }

        printInitialDiagnostics();
        printPrecisionTriplets();
        printUniquePrecisionTriplets();
    }

    private void printInitialDiagnostics() {

        System.out.println("GenerationTimePrecisionBuilderTest initialised.");
        System.out.println("nodeCount = " + nodeCount);
        System.out.println("edgeCount = " + edgeCount);

        for (int i = 0; i < edgeCount; i++) {
            System.out.println(
                    "edge " + i +
                            ": parent=" + parent[i] +
                            " child=" + child[i] +
                            " length=" + branchLength[i]
            );
        }
    }
}