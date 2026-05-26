# Fuzzy classification of node roles.
#
# Five structural roles:
#   generic     high frequency, high degree, low clustering, shallow core
#               (cross-cutting hubs)
#   core        embedded in the densest subgraph, globally reachable,
#               structurally prestigious
#   bridge      high betweenness, low constraint, low clustering
#               (connector between communities)
#   stable      narrow thematic cluster, confined connections, does not
#               act as a bridge
#   peripheral  structurally marginal - low degree, low core, globally
#               unreachable
#
# Thresholds of the membership functions are derived empirically from
# the percentile distribution of each metric in the current corpus.
# Classification is therefore robust to daily corpora with different
# sizes and densities.
#
# Role assignment rule gives priority to `generic`: if the generic score
# exceeds its minimum threshold, the node is generic regardless of the
# other roles. This prevents cross-cutting hubs from being classified as
# core or bridge. Otherwise, the non-generic role with the highest score
# is selected, provided it is above the role-specific minimum.


ROLE_COLS = ["generic", "core", "bridge", "stable", "peripheral"]


# =============================================================================
# Membership functions
# =============================================================================
# All functions operate on percentile-ranked values in [0, 1].
# Thresholds are empirically derived from each day's corpus, so
# "high" always means "in the top N% of today's distribution".

def clamp01(v):
    return max(0.0, min(1.0, v))


def make_high(t0, t1):
    # Ramp from 0 at t0 to 1 at t1
    def high(x):
        if x <= t0: return 0.0
        if x >= t1: return 1.0
        return (x - t0) / (t1 - t0)
    return high


def make_low(t0, t1):
    # Ramp from 1 at t0 to 0 at t1 (inverted)
    def low(x):
        if x <= t0: return 1.0
        if x >= t1: return 0.0
        return (t1 - x) / (t1 - t0)
    return low


def make_not_high(t0, t1):
    # Complement of make_high
    _high = make_high(t0, t1)
    def not_high(x):
        return 1.0 - _high(x)
    return not_high


def compute_thresholds(metrics_norm, cols):
    # Computes percentile transition thresholds for each metric.
    # These anchor points ensure that role definitions are relative to
    # today's corpus structure and not to fixed absolute values.
    thresholds = {}
    for col in cols:
        series = metrics_norm[col]
        thresholds[col] = {
            "p20": series.quantile(0.20),
            "p30": series.quantile(0.30),
            "p40": series.quantile(0.40),
            "p50": series.quantile(0.50),
            "p65": series.quantile(0.65),
            "p80": series.quantile(0.80),
            "p90": series.quantile(0.90),
        }
    return thresholds


# =============================================================================
# Scoring and assignment
# =============================================================================

def compute_role_scores(row, mf, use_avg_neigh_deg):
    # Computes the five fuzzy role scores for a single node (one row of
    # the metrics DataFrame), using the precomputed membership functions
    # `mf` (a dict of dicts: metric -> shape -> callable).
    # use_avg_neigh_deg controls whether avg_neigh_deg is available;
    # when False the metric is treated as zero (its role contribution
    # collapses but the score weights stay comparable across days).
    freq   = float(row["frequency"])
    deg    = float(row["degree"])
    betw   = float(row["betweenness"])
    harm   = float(row["harmonic"])
    eig    = float(row["eigen"])
    clust  = float(row["clustering"])
    constr = float(row["constraint"])
    core   = float(row["core_n"])
    avg_nd = float(row["avg_neigh_deg"]) if use_avg_neigh_deg else 0.0

    # GENERIC: high frequency + high degree + low clustering + shallow core
    generic = clamp01(
        0.25 * mf["frequency"]["high"](freq) +
        0.25 * mf["cluster_freq"]["high"](float(row["cluster_freq"])) +
        0.25 * mf["degree"]["high"](deg) +
        0.15 * mf["clustering"]["low_soft"](clust) +
        0.10 * mf["core_n"]["not_high_soft"](core)
    )

    # CORE: embedded in the densest subgraph, reachable, prestigious
    core_role = clamp01(
        0.45 * mf["core_n"]["very_high"](core) +
        0.25 * mf["harmonic"]["high"](harm) +
        0.15 * mf["eigen"]["high"](eig) +
        0.10 * mf["frequency"]["not_high_soft"](freq) +
        0.05 * mf["cluster_freq"]["not_high_soft"](float(row["cluster_freq"]))
    )

    # BRIDGE: high betweenness + low constraint + low clustering + not ubiquitous.
    # Moderate cluster_freq: a true bridge crosses a few distinct topics,
    # not all topics (which would indicate generic).
    bridge = clamp01(
        0.40 * mf["betweenness"]["high"](betw) +
        0.25 * mf["constraint"]["low"](constr) +
        0.20 * mf["clustering"]["low_soft"](clust) +
        0.10 * mf["frequency"]["not_high_soft"](freq) +
        0.05 * mf["cluster_freq"]["not_high_soft"](float(row["cluster_freq"]))
    )

    # STABLE: narrow thematic cluster, confined connections, not a bridge
    stable = clamp01(
        0.35 * mf["clustering"]["high"](clust) +
        0.30 * mf["constraint"]["high"](constr) +
        0.20 * mf["betweenness"]["low_soft"](betw) +
        0.15 * mf["cluster_freq"]["low"](float(row["cluster_freq"]))
    )

    # PERIPHERAL: structurally marginal - low degree, low core, unreachable
    peripheral = clamp01(
        0.40 * mf["core_n"]["low"](core) +
        0.35 * mf["degree"]["low"](deg) +
        0.25 * mf["harmonic"]["low"](harm)
    )

    return {
        "generic":    generic,
        "core":       core_role,
        "bridge":     bridge,
        "stable":     stable,
        "peripheral": peripheral,
    }


def assign_role_from_scores(row, role_cols, min_scores):
    # Selects the structural role for a node, given its five fuzzy scores.
    # Generic has priority over core and bridge: if the generic score is
    # above its minimum threshold, the node is generic regardless of the
    # other roles. This prevents cross-cutting hubs from being labelled
    # core or bridge. Otherwise, the non-generic role with the highest
    # score wins, provided it is above the role-specific minimum.
    scores_dict = {r: row[r] for r in role_cols}

    gh_score = scores_dict["generic"]
    if gh_score >= min_scores.get("generic", 0.55):
        return "generic"

    top_role, top_score = max(
        {r: scores_dict[r] for r in role_cols if r != "generic"}.items(),
        key=lambda kv: kv[1]
    )
    threshold = min_scores.get(top_role, 0.55)
    return top_role if top_score >= threshold else "unclassified"


def classify_nodes(metrics_norm, corr, corr_threshold=0.85, min_scores=None):
    # Assigns a structural role to each node through fuzzy scoring.
    #
    # The membership-function thresholds are empirically derived from the
    # percentile distribution of each metric in the current corpus, so
    # classification is robust to different corpora on different days.
    #
    # Parameters:
    #   metrics_norm     DataFrame of percentile-ranked metrics, one row per node
    #   corr             Spearman correlation matrix
    #   corr_threshold   if |rho(avg_neigh_deg, eigen)| exceeds this threshold
    #                    avg_neigh_deg is discarded
    #   min_scores       minimum fuzzy score by role for assignment,
    #                    default 0.55 for all roles if not specified
    #
    # Returns the input DataFrame with role scores and the 'concept_type' column.
    if min_scores is None:
        min_scores = {r: 0.55 for r in ROLE_COLS}

    rho_avg_eigen     = abs(corr.loc["avg_neigh_deg", "eigen"])
    use_avg_neigh_deg = rho_avg_eigen <= corr_threshold

    if use_avg_neigh_deg:
        print(f"avg_neigh_deg retained  (rho with eigen = {rho_avg_eigen:.3f})")
    else:
        print(f"avg_neigh_deg dropped   (rho with eigen = {rho_avg_eigen:.3f} > {corr_threshold})")

    classification_cols = [
        "frequency", "cluster_freq", "degree", "betweenness", "harmonic",
        "eigen", "clustering", "constraint", "core_n"
    ]
    if use_avg_neigh_deg:
        classification_cols.insert(2, "avg_neigh_deg")

    thr = compute_thresholds(metrics_norm, classification_cols)

    mf = {}
    for col in classification_cols:
        t = thr[col]
        mf[col] = {
            "high":         make_high(t["p65"], t["p80"]),
            "very_high":    make_high(t["p80"], t["p90"]),
            "low":          make_low(t["p20"], t["p30"]),
            "not_high":     make_not_high(t["p65"], t["p80"]),
            "high_soft":    make_high(t["p50"], t["p65"]),
            "not_high_soft":make_not_high(t["p40"], t["p65"]),
            "low_soft":     make_low(t["p30"], t["p50"]),
        }

    scores = metrics_norm.apply(
        lambda row: compute_role_scores(row, mf, use_avg_neigh_deg),
        axis=1, result_type="expand"
    )
    result = metrics_norm.copy()
    for col in ROLE_COLS:
        result[col] = scores[col].values

    result["concept_type"] = result.apply(
        lambda row: assign_role_from_scores(row, ROLE_COLS, min_scores),
        axis=1
    )
    return result
