# Shared visual palette and constants.
#
# Centralising these here avoids forward-reference issues in the original
# notebook, where ROLE_COLORS was defined late but referenced by upstream
# rendering helpers, and ensures every visualisation module reads from
# the same source of truth.

import seaborn as sns


# =============================================================================
# Role palette - used by all visualisations
# =============================================================================
ROLE_COLORS = {
    "core":       "#355C7D",  # dark blue: central, structurally important
    "bridge":     "#8C2F39",  # dark red: connector between communities
    "stable":     "#6A9A86",  # sage green: compact thematic clusters
    "peripheral": "#6F5A8A",  # dusty purple: marginal nodes
}


# Role highlight palette: same hues as ROLE_COLORS but lighter.
# Used both for the rectangle-background style (use_bold_style=False) and
# as the text colour for the bold-text style (use_bold_style=True), so the
# semantic mapping role -> visual identity stays consistent across modes.
ROLE_HIGHLIGHT_COLORS = {
    "core":         "#AEC6E8",
    "bridge":       "#E8A8AD",
    "stable":       "#B8D8CC",
    "peripheral":   "#C9C0D8",
    "generic":      "#E0D8C0",
    "unclassified": "#E0E0E0",
}


# =============================================================================
# Ego-graph rendering
# =============================================================================
EGO_NODE_COLOR   = "#C97C3A"   # terracotta - reserved for the ego node
EGO_OTHER_COLOR  = "#858383"   # grey - for unclassified neighbours

EGO_ROLE_SHAPES = {
    "core":       "s",
    "bridge":     "d",
    "stable":     "^",
    "peripheral": "o",
}
EGO_ROLE_SIZES = {
    "core":       320,
    "bridge":     300,
    "stable":     280,
    "peripheral": 240,
}
EGO_ROLE_FONT_SIZES = {
    "core":       17,
    "bridge":     16,
    "stable":     15,
    "peripheral": 13,
}


# =============================================================================
# Community visualisation
# =============================================================================
# Sizes are intentionally larger than EGO_ROLE_SIZES because the community
# graph is plotted on its own (no ego star), so nodes need to be more visible.
COMM_ROLE_SHAPES = {
    "core":       "s",
    "bridge":     "d",
    "stable":     "^",
    "peripheral": "o",
}
COMM_ROLE_NODE_SIZES = {
    "core":       500,
    "bridge":     400,
    "stable":     350,
    "peripheral": 200,
}
COMM_OTHER_COLOR = "#858383"
COMM_OTHER_SHAPE = "o"
COMM_OTHER_SIZE  = 150


# =============================================================================
# Source palette
# =============================================================================
def set_palette(n_colors=10):
    # Defines a custom color palette to use for journal/source names.
    # Falls back to seaborn's husl palette beyond 20 colours.
    if n_colors <= 20:
        custom_palette = [
            "#6F5A8A",  # dusty purple
            "#3F5E7A",  # muted blue
            "#2E6C80",  # subdued teal blue
            "#6F8F3A",  # desaturated olive green
            "#D4B866",  # sandy yellow
            "#C97C3A",  # terracotta orange
            "#8C3F3F",  # brownish red
            "#2E4F63",  # muted deep blue
            "#C76A72",  # dusty salmon red
            "#9A5A83",  # muted magenta
            "#565084",  # soft dark purple
            "#8F5A86",  # desaturated purple
            "#C45A5F",  # soft red
            "#D1B04A",  # soft ochre
            "#6A9A86",  # sage green
            "#3F7DA6",  # muted light blue
            "#C87A55",  # soft warm orange
            "#B56183",  # muted fuchsia
            "#5E4E82",  # soft vintage purple
            "#D39A3A",  # soft ochre orange
        ]
        return custom_palette[:n_colors]
    else:
        return sns.color_palette("husl", n_colors)
