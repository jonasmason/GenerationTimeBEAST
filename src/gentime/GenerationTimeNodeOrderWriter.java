package gentime;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Writes a node-order / clade mapping CSV for the (fixed) tree, using BEAST's own
 * node numbering, so downstream R can map each logged generationTimes.N column to
 * the clade it represents.
 *
 * This is the production replacement for the mapping that used to be emitted only
 * by the *Test Java classes (via writePosteriorMean diagnostics). It has no test
 * dependency and is safe to leave on in normal runs: it implements
 * StateNodeInitialiser purely so that BEAST instantiates it and calls
 * initAndValidate(); it contributes no state nodes and changes no state.
 *
 * It writes exactly the columns the diagnostic needs:
 *     beast_node_nr, node_id, is_leaf, is_root, latent_index, clade_key
 * where beast_node_nr = node.getNr() (so the logged parameter index N maps to
 * beast_node_nr = N - 1), and clade_key is the "|"-joined, lexicographically
 * sorted set of descendant leaf labels (a single leaf label for tips). This
 * matches the R clade key built from sort(tree$tip.label[...]).
 *
 * Place in the MCMC run as:
 *   <init spec="gentime.GenerationTimeNodeOrderWriter" tree="@tree" outputPath="..."/>
 */
public class GenerationTimeNodeOrderWriter extends BEASTObject implements StateNodeInitialiser {

    public Input<Tree> treeInput = new Input<>(
            "tree",
            "Tree whose BEAST node numbering and clades are exported.",
            Input.Validate.REQUIRED
    );

    public Input<String> outputPathInput = new Input<>(
            "outputPath",
            "Absolute path of the CSV file to write.",
            Input.Validate.REQUIRED
    );

    private Tree tree;
    private String outputPath;

    @Override
    public void initAndValidate() {
        tree = treeInput.get();
        outputPath = outputPathInput.get();
        writeNodeOrder();
    }

    private void writeNodeOrder() {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("beast_node_nr,node_id,is_leaf,is_root,latent_index,clade_key");
            final int nodeCount = tree.getNodeCount();
            for (int nr = 0; nr < nodeCount; nr++) {
                final Node node = tree.getNode(nr);
                final String id = node.getID() == null ? "" : node.getID();
                writer.println(
                        nr + "," + id + "," + node.isLeaf() + "," + node.isRoot()
                                + "," + nr + "," + cladeKey(node)
                );
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write node-order CSV to " + outputPath, e);
        }
    }

    private String cladeKey(final Node node) {
        final List<String> labels = new ArrayList<>();
        collectLeafLabels(node, labels);
        Collections.sort(labels);
        return String.join("|", labels);
    }

    private void collectLeafLabels(final Node node, final List<String> labels) {
        if (node.isLeaf()) {
            labels.add(node.getID());
            return;
        }
        for (final Node child : node.getChildren()) {
            collectLeafLabels(child, labels);
        }
    }

    // ---- StateNodeInitialiser: no-op (this object only writes a diagnostic file) ----

    @Override
    public void initStateNodes() {
        // contributes no state initialisation
    }

    @Override
    public void getInitialisedStateNodes(final List<StateNode> stateNodes) {
        // contributes no state nodes
    }
}
