# Daily wordcloud of the top concepts.
#
# Word size encodes `combined_score` (the global ranking signal); colour
# encodes the structural role with intensity scaled by `relevance_score`.
#
# Note on the score used by the colour function: the original notebook
# normalises `relevance_score` using the min/max bounds of `combined_score`.
# The two scores live on different scales, so the resulting alpha is not
# a true [0, 1] normalisation. In practice the rendering is still readable
# because the score distributions are roughly aligned, and changing the
# behaviour would change the look of every previous daily output. The
# original behaviour is preserved here intentionally.

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from wordcloud import WordCloud

from scripts.viz_palette import ROLE_COLORS


def make_wordcloud_color_func(role_dict, score_dict, min_s, max_s):
    # Builds the color_func consumed by the wordcloud library.
    # Factory pattern so the lookups and bounds are captured as closure
    # state without polluting the module namespace.
    def normalize_scores(v):
        return (v - min_s) / (max_s - min_s + 1e-9)

    def color_func(word, font_size, position, orientation,
                   random_state=None, **kwargs):
        role  = role_dict.get(word, "peripheral")
        score = score_dict.get(word, min_s)

        base_hex = ROLE_COLORS.get(role, "#BDBDBD")
        base_color  = np.array(mcolors.to_rgb(base_hex))
        light_color = base_color + (1 - base_color) * 0.6  # lighter version of the same hue

        alpha = normalize_scores(score)
        alpha = 0.4 + 0.6 * alpha  # minimum alpha 0.4 to avoid words that are too faded
        color = light_color + alpha * (base_color - light_color)
        return mcolors.to_hex(np.clip(color, 0, 1))

    return color_func


def plot_daily_wordcloud(top_nodes, show_legend=True):
    # Plots the daily wordcloud from the `top_nodes` DataFrame.
    # Expects columns: tag, concept_type, relevance_score, combined_score.
    #
    # show_legend: when True (default) a role legend is drawn below the
    #   wordcloud. The HTML report pipeline passes show_legend=False, since
    #   the report already explains the roles in its own legend section.
    role_dict  = top_nodes.set_index("tag")["concept_type"].to_dict()
    score_dict = top_nodes.set_index("tag")["relevance_score"].to_dict()

    # min_s and max_s are computed on combined_score (the wordcloud frequency input)
    # and are used by color_func to normalize color intensity.
    min_s, max_s = top_nodes["combined_score"].min(), top_nodes["combined_score"].max()

    color_func = make_wordcloud_color_func(role_dict, score_dict, min_s, max_s)

    # Frequencies for word size = combined_score
    score_dict_size = top_nodes.set_index("tag")["combined_score"].to_dict()

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        collocations=False,
        max_words=200,
        color_func=color_func,
    ).generate_from_frequencies(score_dict_size)

    # Plot with legend
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    if show_legend:
        legend_elements = [
            Patch(facecolor=color, label=role)
            for role, color in ROLE_COLORS.items()
            if role in role_dict.values()
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_elements),
            fontsize=10,
            framealpha=0.9,
            title="Tipo Parola",
            title_fontsize=11,
        )
        plt.subplots_adjust(bottom=0.08)

    plt.tight_layout()
    plt.show()
