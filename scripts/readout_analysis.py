"""Readout analysis: map expression data to protocols/days and rank best readouts.

Stepwise analysis:
  1. Aggregate expression values per protocol × gene × day
  2. Compute temporal trajectories and fold changes
  3. Rank genes by cross-protocol consistency, fold change, and coverage
  4. Identify optimal time points per marker
  5. Export summary tables and figures

Usage:
    python3 scripts/readout_analysis.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from data_layer.gene_panel import TARGET_GENES, ALL_TARGET_GENES, resolve_alias

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/readout_analysis")

EXPECTED_DIRECTION: dict[str, str] = {}
for gene in TARGET_GENES["pluripotency"]:
    EXPECTED_DIRECTION[gene] = "down"
for gene in TARGET_GENES["definitive_endoderm"]:
    EXPECTED_DIRECTION[gene] = "transient_up"
for gene in TARGET_GENES["hepatic_progenitor"]:
    EXPECTED_DIRECTION[gene] = "up"
for gene in TARGET_GENES["mature_hepatocyte"]:
    EXPECTED_DIRECTION[gene] = "up"
for gene in TARGET_GENES["fetal_hepatocyte"]:
    EXPECTED_DIRECTION[gene] = "transient_up"
for gene in TARGET_GENES["cholangiocyte"]:
    EXPECTED_DIRECTION[gene] = "variable"
for gene in TARGET_GENES["mesenchymal"]:
    EXPECTED_DIRECTION[gene] = "low"
EXPECTED_DIRECTION["AFP"] = "transient_up"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect("data/db/protocols.db")
    conn.row_factory = sqlite3.Row
    return conn


# ═══════════════════════════════════════════
# STEP 1: Aggregate expression per protocol × gene × day
# ═══════════════════════════════════════════

def step1_aggregate(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Step 1: Aggregating expression values per protocol × gene × day")

    df = pd.read_sql_query(
        """SELECT ev.protocol_id, p.pmc_id, p.doi, pr.protocol_arm,
                  ev.gene_symbol, ev.value, ev.unit, ev.time_point_day,
                  ev.condition_label, ev.source_type, ev.confidence
           FROM expression_values ev
           JOIN papers p ON p.id = ev.paper_id
           LEFT JOIN protocols pr ON pr.id = ev.protocol_id
           WHERE ev.protocol_id IS NOT NULL""",
        conn,
    )

    df["gene_symbol"] = df["gene_symbol"].apply(resolve_alias)
    df = df[df["gene_symbol"].isin(ALL_TARGET_GENES)]

    agg = (
        df.groupby(["protocol_id", "pmc_id", "doi", "protocol_arm",
                     "gene_symbol", "time_point_day", "unit"])
        .agg(mean_value=("value", "mean"),
             n_replicates=("value", "count"),
             std_value=("value", "std"),
             confidence=("confidence", "max"))
        .reset_index()
    )

    logger.info(f"  {len(agg)} aggregated measurements across "
                f"{agg['protocol_id'].nunique()} protocols, "
                f"{agg['gene_symbol'].nunique()} genes")

    temporal = agg[agg["time_point_day"].notna()].copy()
    n_temporal_protos = temporal["protocol_id"].nunique()
    logger.info(f"  {len(temporal)} measurements with temporal data "
                f"({n_temporal_protos} protocols)")

    return agg


# ═══════════════════════════════════════════
# STEP 2: Compute temporal trajectories and fold changes
# ═══════════════════════════════════════════

def step2_trajectories(agg: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 2: Computing temporal trajectories and fold changes")

    temporal = agg[agg["time_point_day"].notna()].copy()
    temporal = temporal.sort_values(["protocol_id", "gene_symbol", "time_point_day"])

    protos_with_multiday = (
        temporal.groupby("protocol_id")["time_point_day"]
        .nunique()
        .loc[lambda x: x >= 2]
        .index
    )
    temporal = temporal[temporal["protocol_id"].isin(protos_with_multiday)]

    trajectories = []
    for (pid, gene), grp in temporal.groupby(["protocol_id", "gene_symbol"]):
        grp = grp.sort_values("time_point_day")
        days = grp["time_point_day"].values
        vals = grp["mean_value"].values

        if len(days) < 2 or np.all(np.isnan(vals)):
            continue

        first_val = vals[0]
        last_val = vals[-1]
        max_val = np.nanmax(vals)
        min_val = np.nanmin(vals)
        peak_day = days[np.nanargmax(vals)]

        pseudocount = 1.0
        if first_val > 0 and last_val > 0:
            fc_first_last = last_val / first_val
            log2fc = np.log2(fc_first_last)
        elif first_val == 0 and last_val > 0:
            fc_first_last = last_val / pseudocount
            log2fc = np.log2(fc_first_last)
        elif first_val > 0 and last_val == 0:
            fc_first_last = pseudocount / first_val
            log2fc = np.log2(fc_first_last)
        else:
            fc_first_last = 1.0
            log2fc = 0.0

        dynamic_range = max_val - min_val if not np.isnan(max_val) else 0

        direction = EXPECTED_DIRECTION.get(gene, "up")
        if direction == "down":
            follows_expected = log2fc < -0.5
        elif direction == "up":
            follows_expected = log2fc > 0.5
        elif direction == "transient_up":
            follows_expected = peak_day < days[-1] and max_val > first_val * 1.5
        elif direction == "low":
            follows_expected = max_val < np.nanmean(vals) * 2
        else:
            follows_expected = True

        pmc_id = grp["pmc_id"].iloc[0]
        trajectories.append({
            "protocol_id": pid,
            "pmc_id": pmc_id,
            "gene_symbol": gene,
            "first_day": days[0],
            "last_day": days[-1],
            "n_timepoints": len(days),
            "first_value": first_val,
            "last_value": last_val,
            "max_value": max_val,
            "min_value": min_val,
            "peak_day": peak_day,
            "fold_change": fc_first_last,
            "log2fc": log2fc,
            "dynamic_range": dynamic_range,
            "expected_direction": direction,
            "follows_expected": follows_expected,
            "unit": grp["unit"].iloc[0],
        })

    traj_df = pd.DataFrame(trajectories)
    logger.info(f"  {len(traj_df)} trajectories across "
                f"{traj_df['protocol_id'].nunique()} protocols, "
                f"{traj_df['gene_symbol'].nunique()} genes")

    return traj_df


# ═══════════════════════════════════════════
# STEP 3: Rank readouts by consistency, fold change, coverage
# ═══════════════════════════════════════════

def step3_rank_readouts(traj_df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 3: Ranking readouts")

    gene_stats = []
    for gene, grp in traj_df.groupby("gene_symbol"):
        n_protocols = grp["protocol_id"].nunique()
        n_follow_expected = grp["follows_expected"].sum()
        consistency = n_follow_expected / n_protocols if n_protocols > 0 else 0

        median_abs_log2fc = np.nanmedian(np.abs(grp["log2fc"]))
        mean_abs_log2fc = np.nanmean(np.abs(grp["log2fc"]))
        median_dynamic_range = np.nanmedian(grp["dynamic_range"])

        total_protocols_with_any_data = agg[agg["gene_symbol"] == gene]["protocol_id"].nunique()

        direction = EXPECTED_DIRECTION.get(gene, "up")
        category = "unknown"
        for cat, genes in TARGET_GENES.items():
            if gene in genes:
                category = cat
                break

        # Weight dynamic markers (up/down/transient) higher than stable ones
        direction_weight = {"up": 1.0, "down": 0.9, "transient_up": 0.95,
                            "low": 0.3, "variable": 0.3}
        dw = direction_weight.get(direction, 0.5)

        composite_score = (
            consistency * 0.30 * dw +
            min(median_abs_log2fc / 5, 1.0) * 0.35 +
            min(n_protocols / 20, 1.0) * 0.20 +
            min(median_dynamic_range / 1000, 1.0) * 0.15
        )

        gene_stats.append({
            "gene_symbol": gene,
            "category": category,
            "expected_direction": direction,
            "n_protocols_temporal": n_protocols,
            "n_protocols_total": total_protocols_with_any_data,
            "consistency": consistency,
            "n_follow_expected": int(n_follow_expected),
            "median_abs_log2fc": median_abs_log2fc,
            "mean_abs_log2fc": mean_abs_log2fc,
            "median_dynamic_range": median_dynamic_range,
            "composite_score": composite_score,
        })

    ranking = pd.DataFrame(gene_stats).sort_values("composite_score", ascending=False)
    ranking["rank"] = range(1, len(ranking) + 1)

    logger.info("  Top 10 readouts:")
    for _, r in ranking.head(10).iterrows():
        logger.info(f"    {r['rank']:2d}. {r['gene_symbol']:10s} "
                     f"score={r['composite_score']:.3f} "
                     f"consistency={r['consistency']:.0%} "
                     f"|log2FC|={r['median_abs_log2fc']:.1f} "
                     f"n={r['n_protocols_temporal']}")

    return ranking


# ═══════════════════════════════════════════
# STEP 4: Identify optimal time points per marker
# ═══════════════════════════════════════════

def step4_optimal_timepoints(traj_df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 4: Identifying optimal time points per marker")

    temporal = agg[agg["time_point_day"].notna()].copy()

    results = []
    for gene in sorted(traj_df["gene_symbol"].unique()):
        gene_data = temporal[temporal["gene_symbol"] == gene].copy()
        if gene_data.empty:
            continue

        direction = EXPECTED_DIRECTION.get(gene, "up")

        day_stats = (
            gene_data.groupby("time_point_day")
            .agg(
                mean_value=("mean_value", "mean"),
                std_value=("mean_value", "std"),
                n_protocols=("protocol_id", "nunique"),
            )
            .reset_index()
        )
        day_stats = day_stats[day_stats["n_protocols"] >= 2]

        if day_stats.empty:
            continue

        cv = day_stats["std_value"] / day_stats["mean_value"].replace(0, np.nan)
        day_stats["cv"] = cv

        if direction == "up":
            best_idx = day_stats["mean_value"].idxmax()
        elif direction == "down":
            best_idx = day_stats["mean_value"].idxmin()
        elif direction == "transient_up":
            best_idx = day_stats["mean_value"].idxmax()
        else:
            best_idx = day_stats["n_protocols"].idxmax()

        if best_idx is None or pd.isna(best_idx):
            continue

        best = day_stats.loc[best_idx]

        for _, row in day_stats.iterrows():
            is_optimal = row["time_point_day"] == best["time_point_day"]
            results.append({
                "gene_symbol": gene,
                "expected_direction": direction,
                "day": int(row["time_point_day"]),
                "mean_value": row["mean_value"],
                "std_value": row["std_value"],
                "cv": row["cv"],
                "n_protocols": int(row["n_protocols"]),
                "is_optimal_day": is_optimal,
            })

    tp_df = pd.DataFrame(results)

    optimal = tp_df[tp_df["is_optimal_day"]].sort_values("gene_symbol")
    logger.info(f"  Optimal time points for {len(optimal)} genes:")
    for _, r in optimal.iterrows():
        logger.info(f"    {r['gene_symbol']:10s} day {r['day']:3d} "
                     f"(mean={r['mean_value']:.1f}, n={r['n_protocols']})")

    return tp_df


# ═══════════════════════════════════════════
# STEP 5: Generate figures and export
# ═══════════════════════════════════════════

def step5_export(agg: pd.DataFrame, traj_df: pd.DataFrame,
                 ranking: pd.DataFrame, timepoints: pd.DataFrame) -> None:
    logger.info("Step 5: Generating figures and exporting results")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR = Path("manuscript/figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(OUT_DIR / "readout_ranking.csv", index=False)
    traj_df.to_csv(OUT_DIR / "trajectories.csv", index=False)
    timepoints.to_csv(OUT_DIR / "timepoint_analysis.csv", index=False)
    logger.info(f"  Tables saved to {OUT_DIR}/")

    # --- Figure A: Readout ranking bar chart ---
    fig, ax = plt.subplots(figsize=(10, 8))
    top = ranking.head(25).copy()
    top = top.sort_values("composite_score")

    cat_colors = {
        "pluripotency": "#9467bd",
        "definitive_endoderm": "#ff7f0e",
        "hepatic_progenitor": "#2ca02c",
        "mature_hepatocyte": "#d62728",
        "fetal_hepatocyte": "#e377c2",
        "cholangiocyte": "#8c564b",
        "mesenchymal": "#7f7f7f",
    }
    colors = [cat_colors.get(c, "#1f77b4") for c in top["category"]]

    bars = ax.barh(range(len(top)), top["composite_score"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene_symbol"], fontsize=10)
    ax.set_xlabel("Composite readout score", fontsize=12)
    ax.set_title("Top 25 readout genes ranked by composite score", fontsize=13)

    from matplotlib.patches import Patch
    legend_items = []
    for cat, color in cat_colors.items():
        label = cat.replace("_", " ").title()
        if cat in top["category"].values:
            legend_items.append(Patch(facecolor=color, label=label))
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_readout_ranking.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "readout_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure B: Temporal heatmap of key markers ---
    temporal = agg[agg["time_point_day"].notna()].copy()
    top10_genes = ranking.head(15)["gene_symbol"].tolist()

    pivot = (
        temporal[temporal["gene_symbol"].isin(top10_genes)]
        .groupby(["gene_symbol", "time_point_day"])["mean_value"]
        .mean()
        .reset_index()
        .pivot(index="gene_symbol", columns="time_point_day", values="mean_value")
    )

    day_cols = sorted([d for d in pivot.columns if d <= 42])
    pivot = pivot[[c for c in day_cols if c in pivot.columns]]
    gene_order = [g for g in top10_genes if g in pivot.index]
    pivot = pivot.loc[gene_order]

    pivot_norm = pivot.div(pivot.max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = LinearSegmentedColormap.from_list("hepato", ["#f7fbff", "#08519c"])
    im = ax.imshow(pivot_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot_norm.columns)))
    ax.set_xticklabels([f"D{int(d)}" for d in pivot_norm.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_norm.index)))
    ax.set_yticklabels(pivot_norm.index, fontsize=10)
    ax.set_xlabel("Protocol day", fontsize=12)
    ax.set_title("Expression dynamics of top readout genes (row-normalized)", fontsize=13)

    plt.colorbar(im, ax=ax, label="Relative expression (0–1)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_expression_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "expression_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure C: Consistency vs fold change scatter ---
    fig, ax = plt.subplots(figsize=(9, 7))
    for cat, color in cat_colors.items():
        mask = ranking["category"] == cat
        sub = ranking[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub["median_abs_log2fc"], sub["consistency"],
            s=sub["n_protocols_temporal"] * 8,
            c=color, alpha=0.7, edgecolors="white", linewidth=0.5,
            label=cat.replace("_", " ").title(),
        )

    from adjustText import adjust_text
    texts = []
    for _, r in ranking.iterrows():
        texts.append(ax.text(
            r["median_abs_log2fc"], r["consistency"],
            r["gene_symbol"], fontsize=7, ha="center", va="bottom",
            zorder=15,
        ))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
                expand=(2.5, 3.0), force_text=(2.0, 2.5),
                force_points=(1.5, 1.5), force_objects=(0.5, 0.5))
    for child in ax.get_children():
        if hasattr(child, 'arrowstyle') or (hasattr(child, '_arrow_transmuter')):
            child.set_zorder(5)

    ax.set_xlabel("Median |log₂ fold change|", fontsize=12)
    ax.set_ylabel("Consistency (fraction following expected pattern)", fontsize=12)
    ax.set_title("Readout quality: consistency vs. dynamic range", fontsize=13)
    ax.legend(fontsize=9, loc="upper right", labelspacing=1.2, handletextpad=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_readout_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "readout_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure D: Key marker trajectories (multi-panel) ---
    key_markers = {
        "Pluripotency": ["POU5F1", "NANOG"],
        "Endoderm": ["SOX17", "FOXA2"],
        "Hepatic prog.": ["HNF4A", "AFP"],
        "Mature hepatocyte": ["ALB", "CYP3A4"],
    }
    marker_colors = {
        "POU5F1": "#9467bd", "NANOG": "#c5b0d5",
        "SOX17": "#ff7f0e", "FOXA2": "#ffbb78",
        "HNF4A": "#2ca02c", "AFP": "#98df8a",
        "ALB": "#d62728", "CYP3A4": "#ff9896",
    }

    temporal_multi = agg[
        (agg["time_point_day"].notna()) &
        (agg["protocol_id"].isin(
            temporal.groupby("protocol_id")["time_point_day"]
            .nunique().loc[lambda x: x >= 3].index
        ))
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for idx, (panel_name, genes) in enumerate(key_markers.items()):
        ax = axes.flat[idx]
        for gene in genes:
            gene_data = temporal_multi[temporal_multi["gene_symbol"] == gene]
            if gene_data.empty:
                continue

            day_means = (
                gene_data.groupby("time_point_day")["mean_value"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            day_means = day_means[day_means["time_point_day"] <= 42]

            within_study = gene_data.groupby(["protocol_id", "time_point_day"])["mean_value"].mean().reset_index()
            max_per_proto = within_study.groupby("protocol_id")["mean_value"].max()
            norm_vals = []
            for _, row in within_study.iterrows():
                mx = max_per_proto.get(row["protocol_id"], 1)
                norm_vals.append(row["mean_value"] / mx if mx > 0 else 0)
            within_study["norm_value"] = norm_vals

            day_norm = within_study.groupby("time_point_day")["norm_value"].agg(["mean", "std"]).reset_index()
            day_norm = day_norm[day_norm["time_point_day"] <= 42]

            color = marker_colors.get(gene, "#333")
            ax.plot(day_norm["time_point_day"], day_norm["mean"],
                    color=color, linewidth=2, label=gene, marker="o", markersize=4)
            ax.fill_between(
                day_norm["time_point_day"],
                day_norm["mean"] - day_norm["std"],
                day_norm["mean"] + day_norm["std"],
                color=color, alpha=0.15,
            )

        ax.set_title(panel_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative expression" if idx % 2 == 0 else "")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

    axes[1, 0].set_xlabel("Protocol day")
    axes[1, 1].set_xlabel("Protocol day")
    fig.suptitle("Expression trajectories of key differentiation markers\n(cross-protocol mean ± SD, within-study normalized)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_expression_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "expression_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Summary stats for manuscript ---
    summary_path = OUT_DIR / "summary_for_manuscript.txt"
    with open(summary_path, "w") as f:
        f.write("=== Expression Readout Analysis Summary ===\n\n")

        n_vals = len(agg)
        n_protocols = agg["protocol_id"].nunique()
        n_genes = agg["gene_symbol"].nunique()
        n_temporal = temporal["protocol_id"].nunique()
        f.write(f"Total aggregated measurements: {n_vals:,}\n")
        f.write(f"Protocols with expression data: {n_protocols}\n")
        f.write(f"Genes in panel: {n_genes}\n")
        f.write(f"Protocols with temporal (multi-day) data: {n_temporal}\n\n")

        f.write("Top 10 readout genes (by composite score):\n")
        for _, r in ranking.head(10).iterrows():
            f.write(f"  {r['rank']:2d}. {r['gene_symbol']:10s} "
                     f"category={r['category']:22s} "
                     f"score={r['composite_score']:.3f} "
                     f"consistency={r['consistency']:.0%} "
                     f"|log2FC|={r['median_abs_log2fc']:.2f} "
                     f"n_protos={r['n_protocols_temporal']}\n")

        f.write("\nOptimal time points for key markers:\n")
        optimal = timepoints[timepoints["is_optimal_day"]].sort_values("gene_symbol")
        for _, r in optimal.iterrows():
            f.write(f"  {r['gene_symbol']:10s} → day {r['day']:3d} "
                     f"(mean={r['mean_value']:.1f}, CV={r['cv']:.2f}, "
                     f"n_protocols={r['n_protocols']})\n")

        top_up = ranking[ranking["expected_direction"] == "up"].head(5)
        top_down = ranking[ranking["expected_direction"] == "down"].head(3)
        top_transient = ranking[ranking["expected_direction"] == "transient_up"].head(3)

        f.write("\nBest 'activation' readouts (should increase):\n")
        for _, r in top_up.iterrows():
            f.write(f"  {r['gene_symbol']} (consistency={r['consistency']:.0%}, "
                     f"|log2FC|={r['median_abs_log2fc']:.2f})\n")

        f.write("\nBest 'silencing' readouts (should decrease):\n")
        for _, r in top_down.iterrows():
            f.write(f"  {r['gene_symbol']} (consistency={r['consistency']:.0%}, "
                     f"|log2FC|={r['median_abs_log2fc']:.2f})\n")

        f.write("\nBest 'transient' readouts (peak then decline):\n")
        for _, r in top_transient.iterrows():
            f.write(f"  {r['gene_symbol']} (consistency={r['consistency']:.0%}, "
                     f"|log2FC|={r['median_abs_log2fc']:.2f})\n")

    logger.info(f"  Summary written to {summary_path}")
    logger.info("Done.")


def run(db=None):
    """Run readout analysis. Accepts optional PipelineDB instance."""
    if db is not None:
        conn = db._conn
        own_conn = False
    else:
        conn = get_db()
        own_conn = True
    try:
        agg = step1_aggregate(conn)
        traj_df = step2_trajectories(agg)
        ranking = step3_rank_readouts(traj_df, agg)
        timepoints = step4_optimal_timepoints(traj_df, agg)
        step5_export(agg, traj_df, ranking, timepoints)
    finally:
        if own_conn:
            conn.close()


def main():
    run()


if __name__ == "__main__":
    main()
