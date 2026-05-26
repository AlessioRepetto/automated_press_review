# Community visualisation.
#
# Communities with up to `size_threshold` nodes are rendered as labelled
# graphs (spring layout, size proportional to combined_score); larger
# communities are rendered as wordclouds (word size by score, colour by
# role with intensity proportional to score).
#
# The original notebook read the community title from a module-level
# `community_results` global. Here it is passed as an explicit parameter
# to keep the function pure and the data flow visible in the notebook.

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from wordcloud import WordCloud

from scripts.viz_palette import (
    COMM_OTHER_COLOR,
    COMM_OTHER_SHAPE,
    COMM_OTHER_SIZE,
    COMM_ROLE_NODE_SIZES,
    COMM_ROLE_SHAPES,
    ROLE_COLORS,
)


def community_node_color(node, role_lookup):
    # Returns the colour of a community node. Classified nodes get the
    # role colour, unclassified ones get the grey "other" colour.
    return ROLE_COLORS.get(role_lookup.get(node), COMM_OTHER_COLOR)


def community_wordcloud_color_func(comm_score_dict, comm_role_dict,
                                     min_s, max_s):
    # Returns a colour function compatible with the wordcloud library.
    # The returned callable maps a word to an interpolated hue: light
    # at low scores, full role colour at high scores. Built as a factory
    # so the lookups and bounds are captured cleanly without a closure
    # over an outer function's locals (the lookups are explicit args).
    def normalise(v):
        return (v - min_s) / (max_s - min_s + 1e-9)

    def color_fn(word, font_size, position, orientation,
                 random_state=None, **kwargs):
        role     = comm_role_dict.get(word, "unclassified")
        score    = comm_score_dict.get(word, min_s)
        base_hex = ROLE_COLORS.get(role, "#BDBDBD")
        base_color  = np.array(mcolors.to_rgb(base_hex))
        light_color = base_color + (1 - base_color) * 0.6
        alpha = 0.4 + 0.6 * normalise(score)
        color = light_color + alpha * (base_color - light_color)
        return mcolors.to_hex(np.clip(color, 0, 1))

    return color_fn


def plot_community_graph(comm_idx, comm, tags_ranking_df, concepts_graph,
                          score_lookup, role_lookup, title, pos_seed=0):
    # Renders a community as a graph. Layout: spring with edge-weight bias.
    # Node shape and colour reflect the role; node size is proportional
    # to combined_score; unclassified nodes are de-emphasised.
    subgraph = concepts_graph.subgraph(comm).copy()
    pos = nx.spring_layout(subgraph, k=1.2, seed=pos_seed, weight="weight")

    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_edges(
        subgraph, pos, width=0.8, edge_color="#cccccc", alpha=0.7, ax=ax
    )

    for role, shape in COMM_ROLE_SHAPES.items():
        nodelist = [n for n in subgraph.nodes if role_lookup.get(n) == role]
        if not nodelist:
            continue
        sizes = [
            COMM_ROLE_NODE_SIZES[role] * (0.5 + score_lookup.get(n, 0.0))
            for n in nodelist
        ]
        nx.draw_networkx_nodes(
            subgraph, pos, nodelist=nodelist,
            node_size=sizes, node_shape=shape,
            node_color=ROLE_COLORS[role],
            edgecolors="none", ax=ax
        )

    other_nodes = [n for n in subgraph.nodes
                   if role_lookup.get(n) not in COMM_ROLE_SHAPES]
    if other_nodes:
        nx.draw_networkx_nodes(
            subgraph, pos, nodelist=other_nodes,
            node_size=COMM_OTHER_SIZE, node_shape=COMM_OTHER_SHAPE,
            node_color=COMM_OTHER_COLOR,
            edgecolors="none", alpha=0.7, ax=ax
        )

    texts = []
    for n in subgraph.nodes:
        x, y  = pos[n]
        color = community_node_color(n, role_lookup)
        is_classified = role_lookup.get(n) in COMM_ROLE_SHAPES
        t = ax.text(
            x, y, str(n).upper(),
            fontsize=13 if is_classified else 9,
            fontweight="bold", color=color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.4)
        )
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
        texts.append(t)

    adjust_text(
        texts,
        expand_text=(1.4, 1.5),
        expand_points=(1.2, 1.3),
        force_text=(1.0, 1.2),
        force_points=(0.3, 0.5),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)
    )

    legend_elements = [
        Line2D([0], [0], marker=COMM_ROLE_SHAPES[role], color="w",
               markerfacecolor=ROLE_COLORS[role], markersize=10, label=role)
        for role in COMM_ROLE_SHAPES
        if any(role_lookup.get(n) == role for n in subgraph.nodes)
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COMM_OTHER_COLOR,
               markersize=8, label="other")
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_elements),
        fontsize=9, framealpha=0.9,
        title="Node role", title_fontsize=10,
    )
    ax.set_title(
        f"Tema portante {comm_idx} - {title.upper()}  ({len(comm)} parole)",
        loc="left", fontweight="bold", fontsize=13
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_community_wordcloud(comm_idx, comm, score_lookup, role_lookup, title):
    # Renders a community as a wordcloud when the node count is too large
    # for a readable graph plot. Word size encodes combined_score; colour
    # encodes role with intensity proportional to score.
    comm_score_dict = {
        n: score_lookup.get(n, 0.0)
        for n in comm
        if score_lookup.get(n, 0.0) > 0
    }

    if not comm_score_dict:
        print(f"Tema portante {comm_idx}: no scored words, skipping wordcloud.")
        return

    comm_role_dict = {n: role_lookup.get(n, "unclassified") for n in comm}
    min_s = min(comm_score_dict.values())
    max_s = max(comm_score_dict.values())

    color_fn = community_wordcloud_color_func(
        comm_score_dict, comm_role_dict, min_s, max_s
    )

    wc = WordCloud(
        width=900, height=500,
        background_color="white",
        collocations=False,
        max_words=200,
        color_func=color_fn,
    ).generate_from_frequencies(comm_score_dict)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Tema portante {comm_idx} - {title.upper()}  ({len(comm)} parole)",
        loc="left", fontweight="bold", fontsize=13
    )

    legend_elements = [
        Patch(facecolor=ROLE_COLORS[role], label=role)
        for role in ROLE_COLORS
        if any(comm_role_dict.get(n) == role for n in comm)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_elements),
        fontsize=10, framealpha=0.9,
        title="Node role", title_fontsize=11,
    )

    plt.subplots_adjust(bottom=0.08)
    plt.tight_layout()
    plt.show()


def plot_community(comm_idx, comm, tags_ranking_df, concepts_graph,
                    community_results, size_threshold=30, pos_seed=0):
    # Visualises a community as a graph (if nodes <= size_threshold) or
    # as a wordcloud (otherwise). Both branches delegate to dedicated
    # top-level helpers so the rendering style is easy to swap out and
    # the colour/wordcloud functions are not buried in closures.
    #
    # `community_results` is the dict produced by analyse_top_communities,
    # passed in explicitly so this function stays pure (no module globals).
    role_lookup  = tags_ranking_df.set_index("tag")["concept_type"].to_dict()
    score_lookup = tags_ranking_df.set_index("tag")["combined_score"].to_dict()
    title        = community_results[comm_idx]["title"]

    if len(comm) <= size_threshold:
        plot_community_graph(
            comm_idx, comm, tags_ranking_df, concepts_graph,
            score_lookup, role_lookup, title, pos_seed=pos_seed
        )
    else:
        plot_community_wordcloud(
            comm_idx, comm, score_lookup, role_lookup, title
        )
