"""Static charts for the EDA report (matplotlib, saved as PNG).

Conventions used in every chart:
- Churn RATE charts use one colour (blue); colour carries no meaning there.
- When churned vs retained customers are compared, churned is always orange
  and retained is always blue, in every chart.
- Bars are labelled directly with the rate and group size n, because a
  45% rate on 20 customers means much less than 45% on 2,000.
- Whiskers show the 95% Wilson confidence interval.
- A dashed grey line marks the overall churn rate for reference.
"""

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed (works on servers/CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn_platform.analysis import CATEGORICAL_FEATURES

BLUE = "#2a78d6"        # retained / single-series
ORANGE = "#eb6834"      # churned
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10,
})


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_churn_distribution(summary: dict, path) -> None:
    labels = ["Retained", "Churned"]
    counts = [summary["retained"], summary["churned"]]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(labels, counts, color=[BLUE, ORANGE], height=0.5)
    for i, n in enumerate(counts):
        ax.text(n + 60, i, f"{n:,}  ({_pct(n / summary['customers'])})", va="center", color=INK)
    ax.set_xlim(0, max(counts) * 1.3)
    ax.set_xlabel("Customers")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Churn distribution (n = {summary['customers']:,})")
    _save(fig, path)


def plot_rate_bars(table: pd.DataFrame, column: str, title: str, overall_rate: float, path) -> None:
    """Horizontal churn-rate bars with CI whiskers and n labels. Rows are drawn top to bottom in table order."""
    t = table.iloc[::-1]  # barh draws the first row at the bottom; reverse so the table order reads top-down
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 1.4))
    ax.barh(y, t["churn_rate"], color=BLUE, height=0.5)
    ax.errorbar(t["churn_rate"], y,
                xerr=[t["churn_rate"] - t["ci_low"], t["ci_high"] - t["churn_rate"]],
                fmt="none", ecolor=INK_SECONDARY, elinewidth=1, capsize=3)
    for yi, (rate, hi, n) in enumerate(zip(t["churn_rate"], t["ci_high"], t["customers"])):
        ax.text(hi + 0.01, yi, f"{_pct(rate)}  (n={n:,})", va="center", color=INK)
    ax.axvline(overall_rate, color=INK_SECONDARY, linestyle="--", linewidth=1)
    # Label the reference line just above the plot area; the title gets extra padding to make room.
    ax.text(overall_rate, 1.01, f"overall {_pct(overall_rate)}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(y, t[column])
    ax.set_xlim(0, min(1.0, t["ci_high"].max() + 0.2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Churn rate (whiskers: 95% confidence interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=20)
    _save(fig, path)


def plot_numeric_distributions(df: pd.DataFrame, path) -> None:
    """Density histograms of each numeric column, churned vs retained.

    density=True scales each group to the same area, so the SHAPES can be
    compared even though there are ~2.8x more retained customers.
    """
    specs = [("tenure", "Tenure (months)"), ("MonthlyCharges", "Monthly charges ($)"),
             ("TotalCharges", "Total charges ($, 11 blanks excluded)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (col, label) in zip(axes, specs):
        values = df[col].dropna()
        bins = np.histogram_bin_edges(values, bins=30)
        for flag, name, color in [(0, "Retained", BLUE), (1, "Churned", ORANGE)]:
            part = df.loc[df["churned"] == flag, col].dropna()
            ax.hist(part, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                    label=f"{name} (median {part.median():,.1f})")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Density (each group scaled to equal area)")
    fig.suptitle("Numeric distributions by churn status", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_all_categorical_rates(cat_rates: pd.DataFrame, overall_rate: float, path) -> None:
    """Small multiples: churn rate for each value of every categorical feature."""
    ncols = 4
    nrows = int(np.ceil(len(CATEGORICAL_FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), sharex=True)
    for ax, feature in zip(axes.flat, CATEGORICAL_FEATURES):
        t = cat_rates[cat_rates["feature"] == feature].sort_values("churn_rate")
        ax.barh(t["value"], t["churn_rate"], color=BLUE, height=0.5)
        for yi, (rate, n) in enumerate(zip(t["churn_rate"], t["customers"])):
            ax.text(rate + 0.01, yi, f"{_pct(rate)} (n={n:,})", va="center", fontsize=8, color=INK)
        ax.axvline(overall_rate, color=INK_SECONDARY, linestyle="--", linewidth=1)
        # Same y-range for every panel so 2-value and 4-value features get equally thick bars.
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(feature, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[len(CATEGORICAL_FEATURES):]:
        ax.set_visible(False)
    axes.flat[0].set_xlim(0, 0.75)
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(f"Churn rate by every categorical feature (dashed line = overall {_pct(overall_rate)})",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_segment_heatmap(seg: pd.DataFrame, path) -> None:
    """Churn rate for each Contract x InternetService combination."""
    rows = ["Month-to-month", "One year", "Two year"]
    cols = ["Fiber optic", "DSL", "No"]
    rate = seg.pivot(index="Contract", columns="InternetService", values="churn_rate").loc[rows, cols]
    n = seg.pivot(index="Contract", columns="InternetService", values="customers").loc[rows, cols]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUES)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(rate.values, cmap=cmap, vmin=0, vmax=rate.values.max(), aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = rate.values[i, j]
            ink = "white" if value > 0.6 * rate.values.max() else INK
            ax.text(j, i, f"{_pct(value)}\nn={n.values[i, j]:,}", ha="center", va="center", color=ink, fontsize=9)
    ax.set_xticks(range(len(cols)), [f"Internet: {c}" for c in cols])
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, format=matplotlib.ticker.PercentFormatter(1.0), label="Churn rate")
    ax.set_title("Churn rate by contract and internet service")
    _save(fig, path)


# --- Phase 3: model evaluation charts ---------------------------------------

# Fixed colour per model (never by rank). The dummy baseline is grey because
# it is a reference line, not a competitor.
MODEL_COLORS = {"dummy_most_frequent": INK_SECONDARY, "logreg": BLUE, "logreg_balanced": ORANGE}
MODEL_LABELS = {"dummy_most_frequent": "Dummy (most frequent)", "logreg": "LogReg",
                "logreg_balanced": "LogReg (balanced weights)"}


def plot_confusion_matrices(test_metrics: dict, path) -> None:
    """One 2x2 confusion matrix per model, counts and row percentages."""
    names = list(test_metrics)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUES)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8))
    for ax, name in zip(np.atleast_1d(axes), names):
        cm = test_metrics[name]["confusion_matrix"]
        grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        row_share = grid / grid.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = "white" if row_share[i, j] > 0.6 else INK
                ax.text(j, i, f"{grid[i, j]:,}\n({_pct(row_share[i, j])} of row)",
                        ha="center", va="center", color=ink, fontsize=9)
        ax.set_xticks([0, 1], ["Pred: retained", "Pred: churned"])
        ax.set_yticks([0, 1], ["Actual: retained", "Actual: churned"])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(MODEL_LABELS.get(name, name), fontsize=11)
    fig.suptitle("Test-set confusion matrices (threshold 0.5)", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_roc_pr_curves(y_true, probas: dict, path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, p in probas.items():
        color = MODEL_COLORS.get(name, BLUE)
        style = "--" if name.startswith("dummy") else "-"
        fpr, tpr, _ = roc_curve(y_true, p)
        ax_roc.plot(fpr, tpr, style, color=color, linewidth=2,
                    label=f"{MODEL_LABELS.get(name, name)}  AUC {roc_auc_score(y_true, p):.3f}")
        label = f"{MODEL_LABELS.get(name, name)}  AP {average_precision_score(y_true, p):.3f}"
        if name.startswith("dummy"):
            # A constant score has one real operating point: precision = churn rate.
            # (sklearn's curve adds a (recall 0, precision 1) endpoint that would
            # draw a misleading vertical spike.)
            ax_pr.axhline(np.mean(y_true), linestyle="--", color=color, linewidth=2, label=label)
            continue
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax_pr.plot(rec, prec, style, color=color, linewidth=2, drawstyle="steps-post", label=label)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.set_title("ROC curve (test set)")
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set_title(f"Precision–recall curve (test set; churn rate {np.mean(y_true):.1%})")
    for ax in (ax_roc, ax_pr):
        ax.legend(frameon=False, fontsize=9, loc="lower right" if ax is ax_roc else "upper right")
    _save(fig, path)


# --- Phase 4: comparison, threshold, calibration, interpretation -------------

AQUA = "#1baf7a"
VIOLET = "#6250d6"
MODEL_COLORS.update({"decision_tree": AQUA, "random_forest": VIOLET})
MODEL_LABELS.update({"decision_tree": "Decision tree (depth 5)", "random_forest": "Random forest (300 trees)"})
VARIANT_COLORS = {"uncalibrated": BLUE, "sigmoid": ORANGE, "isotonic": AQUA}


def plot_model_comparison(folds_table: pd.DataFrame, path) -> None:
    """Fold-level ROC-AUC and AP per model: each dot is one CV fold, the bar is the mean."""
    models = list(dict.fromkeys(folds_table["model"]))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.7 * len(models) + 1.6), sharey=True)
    for ax, (metric, label) in zip(axes, [("roc_auc", "ROC-AUC"), ("average_precision", "Average precision (PR-AUC)")]):
        for i, model in enumerate(models):
            values = folds_table.loc[folds_table["model"] == model, metric]
            color = MODEL_COLORS.get(model, BLUE)
            ax.scatter(values, np.full(len(values), i), s=36, color=color, zorder=3, edgecolor=SURFACE, linewidth=1)
            ax.plot([values.mean()] * 2, [i - 0.28, i + 0.28], color=INK, linewidth=2)
            ax.text(values.max() + 0.006, i, f"{values.mean():.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlabel(f"{label} per fold (dot) and mean (bar)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models])
    axes[0].set_xlim(0.45, 0.9)
    axes[1].set_xlim(0.2, 0.75)
    fig.suptitle("Model comparison: 5-fold out-of-fold results on the training set",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_threshold_costs(tables: dict, sensitivity: pd.DataFrame, primary_table: pd.DataFrame,
                         primary_threshold: float, path) -> None:
    """Left: cost vs threshold per FN:FP ratio (relative to cost at 0.5). Right: precision/recall."""
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blues, low -> high ratio
    fig, (ax_cost, ax_pr) = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, (ratio, table) in zip(ramp, tables.items()):
        at_half = table.loc[np.isclose(table["threshold"], 0.5), "cost"].iloc[0]
        ax_cost.plot(table["threshold"], table["cost"] / at_half, color=color, linewidth=2,
                     label=f"FN cost = {ratio}x FP cost")
        best = sensitivity.loc[sensitivity["fn_to_fp_cost_ratio"] == ratio].iloc[0]
        ax_cost.scatter([best["selected_threshold"]], [best["cost_at_selected"] / at_half], color=color, s=50,
                        zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax_cost.axvline(0.5, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax_cost.axhline(1.0, color=GRID, linewidth=1)
    ax_cost.set(xlabel="Threshold (flag if churn probability >= threshold)",
                ylabel="Total cost relative to threshold 0.5", ylim=(0.4, 2.0), xlim=(0.05, 0.95))
    ax_cost.set_title("Assumed cost vs threshold (dot = lowest cost)")
    ax_cost.legend(frameon=False, fontsize=9, loc="lower right")

    ax_pr.plot(primary_table["threshold"], primary_table["precision"], color=BLUE, linewidth=2, label="Precision")
    ax_pr.plot(primary_table["threshold"], primary_table["recall"], color=ORANGE, linewidth=2, label="Recall")
    ax_pr.axvline(0.5, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax_pr.axvline(primary_threshold, color=INK, linewidth=1)
    ax_pr.text(primary_threshold, 1.02, f"selected {primary_threshold:.2f}", ha="center", fontsize=9,
               transform=ax_pr.get_xaxis_transform())
    ax_pr.text(0.5, 1.02, "0.5", ha="center", fontsize=9, color=INK_SECONDARY, transform=ax_pr.get_xaxis_transform())
    ax_pr.set(xlabel="Threshold", ylim=(0, 1), xlim=(0.05, 0.95))
    ax_pr.set_title("Precision and recall vs threshold", pad=18)
    ax_pr.legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Threshold analysis on training out-of-fold predictions (costs are illustrative assumptions)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_calibration(curves: pd.DataFrame, variants: dict, path) -> None:
    fig, (ax_lr, ax_models, ax_hist) = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax in (ax_lr, ax_models):
        ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linestyle="--", linewidth=1, label="Perfect calibration")
        ax.set(xlabel="Mean predicted churn probability (bin)", ylabel="Observed churn rate (bin)",
               xlim=(0, 1), ylim=(0, 1))
    for name in variants:
        c = curves[curves["variant"] == name]
        ax_lr.plot(c["mean_predicted"], c["observed_rate"], "-o", color=VARIANT_COLORS.get(name, BLUE),
                   linewidth=2, markersize=5, label=f"LogReg {name}")
    ax_lr.set_title("LogReg: uncalibrated vs calibrated")
    ax_lr.legend(frameon=False, fontsize=9, loc="upper left")
    c = curves[curves["variant"] == "uncalibrated"]
    ax_models.plot(c["mean_predicted"], c["observed_rate"], "-o", color=BLUE, linewidth=2, markersize=5, label="LogReg")
    for model in ("decision_tree", "random_forest"):
        c = curves[curves["variant"] == f"model:{model}"]
        ax_models.plot(c["mean_predicted"], c["observed_rate"], "-o", color=MODEL_COLORS[model], linewidth=2,
                       markersize=5, label=MODEL_LABELS[model])
    ax_models.set_title("Other candidates (uncalibrated)")
    ax_models.legend(frameon=False, fontsize=9, loc="upper left")
    ax_hist.hist(variants["uncalibrated"], bins=np.linspace(0, 1, 21), color=BLUE, histtype="step", linewidth=2,
                 label="LogReg uncalibrated")
    ax_hist.set(xlabel="Predicted churn probability", ylabel="Customers")
    ax_hist.set_title("How many customers get each probability")
    ax_hist.legend(frameon=False, fontsize=9)
    fig.suptitle("Calibration on training out-of-fold predictions (10 equal-count bins)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_coefficients(coefficients: pd.DataFrame, path, top: int = 20) -> None:
    """Top terms by |log-odds|. Orange = higher churn odds, blue = lower (two-hue diverging)."""
    t = coefficients.head(top).iloc[::-1]
    labels = [f"{f}: {term}" + (f" (vs {ref})" if ref not in ("-", "not missing") else "")
              for f, term, ref in zip(t["feature"], t["term"], t["reference"])]
    colors = [ORANGE if v > 0 else BLUE for v in t["log_odds"]]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(t) + 1.6))
    ax.barh(range(len(t)), t["log_odds"], color=colors, height=0.6)
    for i, (v, odds, ew) in enumerate(zip(t["log_odds"], t["odds_ratio"], t["early_warning"])):
        marker = "  [retrospective]" if ew.startswith("retrospective") else ""
        # Labels always sit to the right (of the bar end, or of zero for negative bars),
        # so they never collide with the category labels on the left.
        ax.text(max(v, 0) + 0.04, i, f"OR {odds:.2f}{marker}", va="center", ha="left", fontsize=8, color=INK)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_yticks(range(len(t)), labels, fontsize=8)
    lim = t["log_odds"].abs().max()
    ax.set_xlim(-lim * 1.1, max(t["log_odds"].max(), 0) + 1.2)
    ax.set_xlabel("Log-odds coefficient (numeric: per 1 SD; categorical: vs reference category)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"LogReg: top {len(t)} terms by |coefficient|\n"
                 "orange = higher churn odds, blue = lower")
    _save(fig, path)


def plot_permutation_importance(perm: dict, path, top: int = 15) -> None:
    fig, axes = plt.subplots(1, len(perm), figsize=(7 * len(perm), 0.38 * top + 1.6))
    for ax, (name, table) in zip(np.atleast_1d(axes), perm.items()):
        t = table.head(top).iloc[::-1]
        colors = [INK_SECONDARY if g else MODEL_COLORS.get(name, BLUE) for g in t["is_group"]]
        ax.barh(range(len(t)), t["auc_drop_mean"], xerr=t["auc_drop_std"], color=colors, height=0.6,
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
        ax.set_yticks(range(len(t)), [f + (" *" if f in ("tenure", "TotalCharges") else "") for f in t["feature"]],
                      fontsize=8)
        ax.axvline(0, color=INK_SECONDARY, linewidth=1)
        ax.set_xlabel("Drop in validation ROC-AUC when shuffled (mean, sd whiskers)")
        ax.grid(axis="y", visible=False)
        ax.set_title(MODEL_LABELS.get(name, name), fontsize=11)
    fig.suptitle("Permutation importance on training CV folds (grey = correlated group shuffled together; "
                 "* = retrospective)", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)
