# Framing visualisations.
#
# Two thin utilities used in the cross-source analysis of section 25:
#
# 1. `describe_cat_column_stacked`: a single horizontal stacked bar that
#    shows the relative coverage of a category (typically `source`) within
#    a cluster, making coverage asymmetry across outlets immediately visible.
#
# 2. `display_framing_table`: pivots the long-form framing DataFrame
#    (source, term, tfidf_score) into a rank x source table and prints it
#    row by row, so the reader can compare how each outlet lexically frames
#    the same story at the same rank.

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns


# Plots a single horizontal stacked bar showing the relative frequency (%)
# of each value in color_map found in column c, plus an 'Other' bucket
# for values not in the map. Labels are shown in a legend below the chart.
#
# Parameters:
#   data         : DataFrame to analyse
#   c            : column name
#   color_map    : {category_value: hex_color} - subset of column values
#   custom_title : optional chart title
#   show_title   : when True (default) the title is drawn. The HTML report
#                  pipeline passes show_title=False, since the report shows
#                  the cluster title in its own section heading.
def describe_cat_column_stacked(data, c, color_map, custom_title="",
                                show_title=True):

    total = data[c].notna().sum()

    # Build shares for mapped values + residual "Other"
    shares = {}
    for val in color_map:
        shares[val] = (data[c] == val).sum() / total * 100

    other_pct = 100 - sum(shares.values())

    # -- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 1.6))

    legend_handles = []
    left = 0
    for val, pct in shares.items():
        bar = ax.barh(0, pct, left=left, color=color_map[val], height=0.5)
        legend_handles.append(
            mpatches.Patch(color=color_map[val], label=f"{val} ({pct:.1f}%)")
        )
        left += pct

    if other_pct > 0.05:
        ax.barh(0, other_pct, left=left, color="#CCCCCC", height=0.5)
        legend_handles.append(
            mpatches.Patch(color="#CCCCCC", label=f"Other ({other_pct:.1f}%)")
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("")
    ax.set_yticks([])
    sns.despine(left=True)

    if show_title:
        title = custom_title if custom_title else f"Distribution of values in column '{c}'"
        plt.suptitle(title, fontsize=13, fontweight="bold", ha="left", x=0, y=1.12)

    # Legend below the chart, wrapping automatically across multiple rows
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        fontsize=8,
        frameon=False,
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.0,
        handletextpad=0.4
    )

    n_rows = math.ceil(len(legend_handles) / 4)  # 4 = ncol
    bottom_margin = 0.2 + n_rows * 0.12
    plt.subplots_adjust(top=0.78, bottom=0.25)
    plt.show()


def display_framing_table(framing_df, top_n=5):
    # Pivot framing into a source x rank table for quick visual contrast.
    # Prints in text format ordered by rank, one row per position,
    # so each row compares sources horizontally at the same rank.
    if framing_df.empty:
        print("   (insufficient data)")
        return

    framing_df = framing_df.copy()
    framing_df["rank"] = framing_df.groupby("source").cumcount() + 1
    framing_df = framing_df[framing_df["rank"] <= top_n]

    pivot = framing_df.pivot(index="source", columns="rank", values="term").fillna("-")
    pivot.columns = [f"#{c}" for c in pivot.columns]

    # Alternatively: display(pivot) for tabular output
    # Here textual printout: after transposing, each column is a source
    pivot = pivot.T
    for c in pivot.columns:
        output_string = c.upper() + ": " + ", ".join(pivot[c].tolist())
        print(output_string)
