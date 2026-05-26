# Ego-graph plotting.
#
# For a chosen concept, draws its radius-1 ego graph with a concentric-shell
# layout: neighbours are placed on up to three rings sorted by edge weight
# (co-occurrence strength), so visual proximity to the ego mirrors semantic
# strength. Node shape and colour encode the structural role (core / bridge /
# stable / peripheral); unclassified neighbours fall back to a neutral grey.
#
# The palette and shape constants live in `viz_palette` to keep this module
# free of style decisions and to avoid duplicating colour definitions across
# visualisation modules.

from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D

from scripts.viz_palette import (
    EGO_NODE_COLOR,
    EGO_OTHER_COLOR,
    EGO_ROLE_FONT_SIZES,
    EGO_ROLE_SHAPES,
    EGO_ROLE_SIZES,
    ROLE_COLORS,
)


def compute_concentric_positions(ego, neighbors, graph, rng):
    # Places `ego` at the origin and distributes its `neighbors` on up to
    # three concentric rings according to edge-weight percentiles:
    #   - ring 1 (r=1.0): top 33% by weight (strong ties)
    #   - ring 2 (r=2.0): middle 33%
    #   - ring 3 (r=3.2): bottom 33%
    # Within each ring, nodes are spaced angularly with small random jitter
    # to avoid perfectly symmetric layouts. Falls back to a single ring
    # when no weights are available.
    pos = {ego: np.array([0.0, 0.0])}

    weights = {}
    for n in neighbors:
        edge = graph.get_edge_data(ego, n) or {}
        weights[n] = edge.get("weight", None)

    has_weights = any(w is not None for w in weights.values())

    if has_weights:
        valid_weights = [w for w in weights.values() if w is not None]
        min_w = min(valid_weights)
        weights = {n: (w if w is not None else min_w) for n, w in weights.items()}

        sorted_nodes = sorted(neighbors, key=lambda n: weights[n], reverse=True)
        n_total = len(sorted_nodes)
        cut1 = max(1, n_total // 3)
        cut2 = max(cut1 + 1, 2 * n_total // 3)

        shells = {
            n: (1 if i < cut1 else (2 if i < cut2 else 3))
            for i, n in enumerate(sorted_nodes)
        }
        radii = {1: 1.0, 2: 2.0, 3: 3.2}
    else:
        shells = {n: 1 for n in neighbors}
        radii  = {1: 1.8}

    ring_nodes = defaultdict(list)
    for n, ring in shells.items():
        ring_nodes[ring].append(n)

    for ring, nodes_in_ring in ring_nodes.items():
        r = radii[ring]
        n_ring = len(nodes_in_ring)
        base_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        jitter = rng.uniform(-0.15, 0.15, size=n_ring)
        angles = base_angles + jitter
        for node, angle in zip(nodes_in_ring, angles):
            pos[node] = np.array([r * np.cos(angle), r * np.sin(angle)])

    return pos


def ego_node_color(node, ego_node, role_lookup):
    # Colour for a node in the ego graph: the ego node gets its own colour;
    # classified neighbours get the role colour from ROLE_COLORS;
    # everything else gets the grey "other" colour.
    if node == ego_node:
        return EGO_NODE_COLOR
    return ROLE_COLORS.get(role_lookup.get(node), EGO_OTHER_COLOR)


def ego_node_font_size(node, ego_node, role_lookup):
    # Label font size for a node in the ego graph. Ego gets the largest;
    # classified neighbours scale by role; unclassified ones get the
    # smallest size so they recede visually.
    if node == ego_node:
        return 22
    return EGO_ROLE_FONT_SIZES.get(role_lookup.get(node), 10)


def plot_ego_graph(G, ego_node, tags_ranking_df, pos_seed=0, ego_type=''):
    # Plots the radius-1 ego graph of a node with a concentric-shell layout.
    #
    # Neighbours are grouped into 3 concentric rings based on edge weight
    # (co-occurrence strength). Strong ties go to the inner ring, weak ties
    # to the outer one. If no weights are available, all neighbours go on
    # a single ring.
    #
    # Node shapes and colours reflect structural roles:
    #   Ego node:    star, terracotta orange, always at the centre
    #   Core:        square, dark blue
    #   Bridge:      diamond, dark red
    #   Stable:      triangle, sage green
    #   Peripheral:  circle, dusty purple
    #   Other:       circle, grey
    #
    # Labels are sized proportionally to role importance (wordcloud style)
    # and placed with adjust_text to avoid overlaps.
    role_lookup = tags_ranking_df.set_index("tag")["concept_type"].to_dict()
    ego_G       = nx.ego_graph(G, ego_node, radius=1)
    other_nodes = [n for n in ego_G.nodes if n != ego_node]

    rng = np.random.default_rng(pos_seed)
    pos = compute_concentric_positions(ego_node, other_nodes, ego_G, rng)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Edges: emphasise spokes from ego (darker), de-emphasise neighbour-to-neighbour
    ego_edges   = [(u, v) for u, v in ego_G.edges()
                   if u == ego_node or v == ego_node]
    other_edges = [(u, v) for u, v in ego_G.edges()
                   if u != ego_node and v != ego_node]

    nx.draw_networkx_edges(
        ego_G, pos, edgelist=ego_edges,
        width=0.9, edge_color="#999999", alpha=0.8, ax=ax
    )
    nx.draw_networkx_edges(
        ego_G, pos, edgelist=other_edges,
        width=0.5, edge_color="#cccccc", alpha=0.4, ax=ax
    )

    # Ego node
    nx.draw_networkx_nodes(
        ego_G, pos, nodelist=[ego_node],
        node_size=700, node_shape="*",
        node_color=EGO_NODE_COLOR,
        edgecolors="black", linewidths=1.2, ax=ax
    )

    # Other nodes
    classified_others   = [n for n in other_nodes if role_lookup.get(n) in EGO_ROLE_SHAPES]
    unclassified_others = [n for n in other_nodes if role_lookup.get(n) not in EGO_ROLE_SHAPES]

    nx.draw_networkx_nodes(
        ego_G, pos, nodelist=unclassified_others,
        node_size=180, node_shape="o",
        node_color=EGO_OTHER_COLOR,
        edgecolors="none", alpha=0.85, ax=ax
    )

    for role, shape in EGO_ROLE_SHAPES.items():
        nodelist = [n for n in classified_others if role_lookup.get(n) == role]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            ego_G, pos, nodelist=nodelist,
            node_size=EGO_ROLE_SIZES[role], node_shape=shape,
            node_color=ROLE_COLORS[role],
            edgecolors="none", ax=ax
        )

    # Wordcloud-style labels, sized by role
    texts = []
    for n in ego_G.nodes:
        x, y      = pos[n]
        color     = ego_node_color(n, ego_node, role_lookup)
        font_size = ego_node_font_size(n, ego_node, role_lookup)

        t = ax.text(
            x, y, str(n).upper(),
            fontsize=font_size, fontweight="bold", color=color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.35)
        )
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        texts.append(t)

    adjust_text(
        texts, ax=ax,
        expand_text=(2.0, 2.5),
        expand_points=(1.8, 2.0),
        force_text=(1.8, 2.2),
        force_points=(0.4, 0.6),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
        lim=500,
        only_move={"texts": "xy", "points": "xy"},
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=EGO_NODE_COLOR,
               markersize=14, markeredgecolor="black", label=f"ego ({ego_node})"),
    ] + [
        Line2D([0], [0], marker=EGO_ROLE_SHAPES[role], color="w",
               markerfacecolor=ROLE_COLORS[role], markersize=10, label=role)
        for role in EGO_ROLE_SHAPES
        if any(role_lookup.get(n) == role for n in other_nodes)
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=EGO_OTHER_COLOR,
               markersize=8, label="other", alpha=0.85)
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_elements),
        fontsize=9, framealpha=0.9,
        title="Parola - Profilo", title_fontsize=10,
    )

    if ego_type:
        ax.set_title(
            "Parola " + ego_type.capitalize() + ': ' + ego_node.upper() + " - voci associate",
            loc="left", fontweight="bold", fontsize=13
        )
    else:
        ax.set_title(
            ego_node.upper() + " - voci associate",
            loc="left", fontweight="bold", fontsize=13
        )

    ax.axis("off")
    plt.tight_layout()
    plt.show()
