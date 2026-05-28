package gentime;

import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.CalculationNode;

import java.util.HashMap;
import java.util.Map;

public class GenerationTimeTraitData extends CalculationNode {
	
    /*
     * Parses and stores observed tip generation-time values from the XML trait string.
     * Validates that every tree tip has exactly one observed generation-time value.
     */
	
	public Input<String> traitStringInput = new Input<>(
            "value",
            "Trait values in the form taxon=value,taxon=value,...",
            Input.Validate.REQUIRED
    );
    
    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose tip taxa should match the generation-time trait values.",
            Input.Validate.REQUIRED
    );
    
    public Input<Boolean> debugInput = new Input<>(
    		"debug",
    		"Print generation-time trait data diagnostics when true.",
    		false
    );

    private Map<String, Double> traitByTaxon;

    @Override
    public void initAndValidate() {

        traitByTaxon = new HashMap<>();

        String traitString = traitStringInput.get();

        String[] entries = traitString.split(",");

        for (String entry : entries) {

            String trimmedEntry = entry.trim();

            if (trimmedEntry.isEmpty()) {
                continue;
            }

            String[] parts = trimmedEntry.split("=");

            if (parts.length != 2) {
                throw new IllegalArgumentException(
                        "Could not parse generation-time entry: " + trimmedEntry +
                        ". Expected format taxon=value."
                );
            }

            String taxon = parts[0].trim();
            double value = Double.parseDouble(parts[1].trim());

            if (traitByTaxon.containsKey(taxon)) {
                throw new IllegalArgumentException(
                        "Duplicate generation-time value for taxon: " + taxon
                );
            }

            traitByTaxon.put(taxon, value);
        }

        if (traitByTaxon.isEmpty()) {
            throw new IllegalArgumentException(
                    "No generation-time trait values were parsed."
            );
        }

        if (debugInput.get()) {	
        	System.out.println("Parsed " + traitByTaxon.size() + " generation-time values.");
        }
        
        Tree tree = treeInput.get();

        int missingCount = 0;

        for (int i = 0; i < tree.getLeafNodeCount(); i++) {

            Node tip = tree.getNode(i);
            String taxon = tip.getID();

            if (!traitByTaxon.containsKey(taxon)) {
                missingCount++;
                System.err.println(
                        "Missing generation-time value for tree tip: " + taxon
                );
            }
        }

        if (missingCount > 0) {
            throw new IllegalArgumentException(
                    "Missing generation-time values for " + missingCount + " tree tips."
            );
        }
    }

    public boolean hasTrait(String taxon) {
        return traitByTaxon.containsKey(taxon);
    }

    public double getTrait(String taxon) {
        if (!traitByTaxon.containsKey(taxon)) {
            throw new IllegalArgumentException(
                    "No generation-time value found for taxon: " + taxon
            );
        }

        return traitByTaxon.get(taxon);
    }

    public int size() {
        return traitByTaxon.size();
    }
    
}