# neat_visualizer.py

import pickle
import math

import matplotlib.pyplot as plt
import networkx as nx


# =========================================================
# CONFIG
# =========================================================

GENOME_FILE = "best_runtime_genome.pkl"

SHOW_DISABLED = False

SHOW_LABELS = True

NODE_SIZE = 1200

FONT_SIZE = 8

EDGE_ALPHA = 0.7

CURVE_RAD = 0.2


# =========================================================
# LOAD GENOME
# =========================================================

import glob
import os
import neat


def load_latest_checkpoint_genome():

    checkpoints = glob.glob("neat-checkpoint-*")

    if not checkpoints:
        raise FileNotFoundError(
            "Не найден ни best_runtime_genome.pkl, "
            "ни один neat-checkpoint-*"
        )

    latest = max(
        checkpoints,
        key=lambda p: int(
            os.path.basename(p).split("-")[-1]
        )
    )

    print(f"📦 Loading checkpoint: {latest}")

    population = neat.Checkpointer.restore_checkpoint(
        latest
    )

    best_genome = max(
        population.population.values(),
        key=lambda g: g.fitness
        if g.fitness is not None
        else float("-inf")
    )

    print(
        f"🏆 Best genome id={best_genome.key} "
        f"fitness={best_genome.fitness}"
    )

    return best_genome


if os.path.exists(GENOME_FILE):

    print(f"📄 Loading genome: {GENOME_FILE}")

    with open(GENOME_FILE, "rb") as f:

        genome = pickle.load(f)

else:

    print(
        f"⚠️ {GENOME_FILE} not found. "
        f"Using latest checkpoint..."
    )

    genome = load_latest_checkpoint_genome()

# =========================================================
# GRAPH
# =========================================================

G = nx.DiGraph()

# =========================================================
# INPUT LABELS
# =========================================================

INPUT_LABELS = []

STACK_SIZE = 4

for t in range(STACK_SIZE):

    INPUT_LABELS += [

        f"wall_L_t-{t}",
        f"wall_C_t-{t}",
        f"wall_R_t-{t}",

        f"enemy_L_t-{t}",
        f"enemy_C_t-{t}",
        f"enemy_R_t-{t}",

        f"hp_t-{t}",

        f"sin_t-{t}",
        f"cos_t-{t}",
    ]


OUTPUT_LABELS = [

    "move_fb",
    "move_lr",
    "look",
    "shoot"
]


# =========================================================
# NODE MAP
# =========================================================

input_keys = genome.connections.keys()

# =========================================================
# INPUT IDS
# =========================================================

input_ids = []

output_ids = []

hidden_ids = set()

for conn_key in genome.connections:

    a, b = conn_key

    if a < 0:

        input_ids.append(a)

    if b >= 0:

        hidden_ids.add(b)

    if a >= 0:

        hidden_ids.add(a)

input_ids = sorted(list(set(input_ids)))

# neat-python outputs are usually:
# 0..N

output_ids = [0, 1, 2, 3]

hidden_ids = list(hidden_ids)

for o in output_ids:

    if o in hidden_ids:

        hidden_ids.remove(o)


# =========================================================
# POSITIONS
# =========================================================

pos = {}

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

for i, node_id in enumerate(input_ids):

    pos[node_id] = (-1, -i)

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

for i, node_id in enumerate(output_ids):

    pos[node_id] = (1, -i * 5)

# ---------------------------------------------------------
# HIDDEN
# ---------------------------------------------------------

for i, node_id in enumerate(hidden_ids):

    angle = i * 0.3

    pos[node_id] = (

        math.cos(angle) * 0.2,

        math.sin(angle) * 10
    )

# =========================================================
# NODE LABELS
# =========================================================

labels = {}

for idx, node_id in enumerate(input_ids):

    if idx < len(INPUT_LABELS):

        labels[node_id] = INPUT_LABELS[idx]

    else:

        labels[node_id] = str(node_id)

for idx, node_id in enumerate(output_ids):

    if idx < len(OUTPUT_LABELS):

        labels[node_id] = OUTPUT_LABELS[idx]

    else:

        labels[node_id] = str(node_id)

for node_id in hidden_ids:

    labels[node_id] = f"H{node_id}"

# =========================================================
# ADD NODES
# =========================================================

for node_id in labels:

    G.add_node(node_id)

# =========================================================
# EDGES
# =========================================================

edge_colors = []

edge_widths = []

for conn_key, conn in genome.connections.items():

    if not conn.enabled and not SHOW_DISABLED:

        continue

    a, b = conn_key

    G.add_edge(a, b)

    w = conn.weight

    # -----------------------------------------------------
    # COLOR
    # -----------------------------------------------------

    if not conn.enabled:

        color = "gray"

    elif w >= 0:

        color = "green"

    else:

        color = "red"

    edge_colors.append(color)

    # -----------------------------------------------------
    # WIDTH
    # -----------------------------------------------------

    edge_widths.append(

        0.5 + abs(w) * 2.0
    )

# =========================================================
# DRAW
# =========================================================

# =========================================================
# NODE COLORS
# =========================================================

node_colors = []

for node in G.nodes():

    if node in input_ids:

        node_colors.append("lightblue")

    elif node in output_ids:

        node_colors.append("lightgreen")

    else:

        node_colors.append("orange")

# =========================================================
# DRAW
# =========================================================

plt.figure(figsize=(22, 14))

# ---------------------------------------------------------
# NODES
# ---------------------------------------------------------

nx.draw_networkx_nodes(

    G,
    pos,

    node_size=NODE_SIZE,

    node_color=node_colors
)

# ---------------------------------------------------------
# EDGES
# ---------------------------------------------------------

nx.draw_networkx_edges(

    G,
    pos,

    edge_color=edge_colors,

    width=edge_widths,

    alpha=EDGE_ALPHA,

    arrows=True,

    connectionstyle=f"arc3,rad={CURVE_RAD}"
)

# ---------------------------------------------------------
# LABELS
# ---------------------------------------------------------

if SHOW_LABELS:

    nx.draw_networkx_labels(

        G,
        pos,

        labels,

        font_size=FONT_SIZE
    )

# =========================================================
# EDGE WEIGHTS
# =========================================================

edge_labels = {}

for conn_key, conn in genome.connections.items():

    if not conn.enabled and not SHOW_DISABLED:

        continue

    a, b = conn_key

    edge_labels[(a, b)] = f"{conn.weight:.2f}"

nx.draw_networkx_edge_labels(

    G,
    pos,

    edge_labels=edge_labels,

    font_size=6
)

# =========================================================
# LEGEND
# =========================================================

from matplotlib.lines import Line2D

legend_elements = [

    Line2D(
        [0],
        [0],

        color='green',

        lw=3,

        label='Positive weight'
    ),

    Line2D(
        [0],
        [0],

        color='red',

        lw=3,

        label='Negative weight'
    ),

    Line2D(
        [0],
        [0],

        marker='o',

        color='w',

        label='Input neuron',

        markerfacecolor='lightblue',

        markersize=12
    ),

    Line2D(
        [0],
        [0],

        marker='o',

        color='w',

        label='Hidden neuron',

        markerfacecolor='orange',

        markersize=12
    ),

    Line2D(
        [0],
        [0],

        marker='o',

        color='w',

        label='Output neuron',

        markerfacecolor='lightgreen',

        markersize=12
    ),
]

plt.legend(

    handles=legend_elements,

    loc='upper left'
)

# =========================================================
# TITLE
# =========================================================

plt.title(

    "NEAT Genome Visualization\n"
    "Green = positive | Red = negative"
)

plt.axis("off")

plt.tight_layout()



# =========================================================
# INPUT LEGEND
# =========================================================

legend_text = """

INPUT LEGEND

wall_L   = left wall sensor
wall_C   = center wall sensor
wall_R   = right wall sensor

enemy_L  = left enemy sensor
enemy_C  = center enemy sensor
enemy_R  = right enemy sensor

hp       = normalized hp

sin      = sin(angle)
cos      = cos(angle)

t-0      = current frame
t-1      = previous frame
t-2      = older frame
t-3      = oldest frame
"""

plt.gcf().text(

    0.82,     # x
    0.5,      # y

    legend_text,

    fontsize=10,

    verticalalignment='center',

    bbox=dict(

        facecolor='white',

        alpha=0.8,

        edgecolor='black'
    )
)

plt.show()