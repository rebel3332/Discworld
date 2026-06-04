# visualize_neat_genome.py
# =========================================================
# UNIVERSAL NEAT GENOME VISUALIZER
# =========================================================
# Supports:
# - best genome .pkl
# - NEAT checkpoints
# - feedforward / recurrent
# - automatic input detection
# - legends
# - disabled links
# =========================================================

import os
import pickle
import graphviz
import neat

# =========================================================
# FILES
# =========================================================

GENOME_FILE = "neat_simple_best_genome.pkl" #None #"best_simple_genome.pkl"


CHECKPOINT_FILE = None #"neat-simple-checkpoint-36" #None
# Example:
# CHECKPOINT_FILE = "neat-simple-checkpoint-12"

OUTPUT_NAME = "neat_graph"

# =========================================================
# INPUT LEGEND
# =========================================================

INPUT_LABELS_9 = {

    -1: "wall_left",
    -2: "wall_front",
    -3: "wall_right",

    -4: "enemy_left",
    -5: "enemy_front",
    -6: "enemy_right",

    -7: "hp",

    -8: "sin(angle)",
    -9: "cos(angle)"
}

# Old history mode (72 inputs)
INPUT_LABELS_72 = {}

frame = 0

for history in range(8):

    prefix = f"t-{7-history}"

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_wall_left"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_wall_front"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_wall_right"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_enemy_left"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_enemy_front"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_enemy_right"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_sin"
    frame += 1

    INPUT_LABELS_72[-(frame+1)] = f"{prefix}_cos"
    frame += 1

# =========================================================
# OUTPUT LEGEND
# =========================================================

OUTPUT_LABELS = {

    0: "move_front_back",
    1: "move_left_right",
    2: "look_delta",
    3: "shoot"
}

# =========================================================
# LOAD
# =========================================================

def load_genome():

    # -----------------------------------------------------
    # BEST GENOME
    # -----------------------------------------------------

    if GENOME_FILE and os.path.exists(GENOME_FILE):

        print(f"📦 Loading genome: {GENOME_FILE}")

        with open(GENOME_FILE, "rb") as f:

            genome = pickle.load(f)

        return genome

    # -----------------------------------------------------
    # CHECKPOINT
    # -----------------------------------------------------

    if CHECKPOINT_FILE:

        print(f"📦 Loading checkpoint: {CHECKPOINT_FILE}")

        population = neat.Checkpointer.restore_checkpoint(
            CHECKPOINT_FILE
        )

        best = max(

            population.population.values(),

            key=lambda g:
                g.fitness
                if g.fitness is not None
                else -999999
        )

        return best

    raise Exception("No genome source found")


# =========================================================
# DETECT INPUTS
# =========================================================

def detect_input_labels(genome):

    input_nodes = [

        n for n in genome.nodes.keys()

        if n < 0
    ]

    count = len(input_nodes)

    print(f"Detected inputs: {count}")

    if count == 9:
        return INPUT_LABELS_9

    if count == 72:
        return INPUT_LABELS_72

    labels = {}

    for n in input_nodes:

        labels[n] = f"input_{abs(n)}"

    return labels


# =========================================================
# VISUALIZE
# =========================================================

def visualize_genome(genome):

    input_labels = detect_input_labels(genome)

    dot = graphviz.Digraph(
        format="png"
    )

    dot.attr(rankdir="LR")

    # =====================================================
    # INPUTS
    # =====================================================

    for node in sorted(input_labels.keys()):

        dot.node(

            str(node),

            input_labels[node],

            shape="box",

            style="filled",

            fillcolor="lightgray"
        )

    # =====================================================
    # OUTPUTS
    # =====================================================

    for node, label in OUTPUT_LABELS.items():

        dot.node(

            str(node),

            label,

            shape="circle",

            style="filled",

            fillcolor="lightblue"
        )

    # =====================================================
    # HIDDEN
    # =====================================================

    for node in genome.nodes.keys():

        if node in OUTPUT_LABELS:
            continue

        if node < 0:
            continue

        dot.node(

            str(node),

            f"H{node}",

            shape="circle",

            style="filled",

            fillcolor="white"
        )

    # =====================================================
    # CONNECTIONS
    # =====================================================

    for conn_key, conn in genome.connections.items():

        input_node, output_node = conn_key

        enabled = conn.enabled

        weight = conn.weight

        color = "green" if weight > 0 else "red"

        width = str(
            min(
                5,
                0.3 + abs(weight)
            )
        )

        style = "solid" if enabled else "dashed"

        label = f"{weight:.2f}"

        dot.edge(

            str(input_node),

            str(output_node),

            color=color,

            penwidth=width,

            style=style,

            label=label
        )

    # =====================================================
    # LEGEND
    # =====================================================

    with dot.subgraph(name="cluster_legend") as c:

        c.attr(label="Legend")

        c.node(
            "legend_pos",
            "+ weight",
            shape="plaintext"
        )

        c.node(
            "legend_neg",
            "- weight",
            shape="plaintext"
        )

        c.node(
            "legend_dis",
            "disabled",
            shape="plaintext"
        )

    # =====================================================
    # SAVE
    # =====================================================

    dot.render(
        OUTPUT_NAME,
        cleanup=True
    )

    print(
        f"\n✅ Saved graph: "
        f"{OUTPUT_NAME}.png"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    genome = load_genome()

    visualize_genome(genome)
