"""Phase 4: choose a classification threshold from (assumed) business costs.

The model outputs a probability. The THRESHOLD turns it into an action:
"flag this customer for a retention offer if p >= t".
- Lower t  -> more customers flagged: more churners caught (fewer FN) but
              more offers wasted on customers who would have stayed (more FP).
- Higher t -> the reverse.
Which is better depends on what each mistake COSTS, and that is a business
fact, not a statistical one. So the threshold is a business decision that
the model's probabilities inform.

Cost model (illustrative, NOT measured):
    total_cost = FP * FP_COST + FN * FN_COST
FP_COST ~ cost of a retention offer given to someone who would have stayed.
FN_COST ~ value lost when a churner is not contacted. This assumes contacting a
churner always saves them, which is optimistic.

Theory check: if the probabilities are well calibrated, the expected cost of
flagging a customer with probability p is (1-p)*FP_COST, and of NOT flagging
them p*FN_COST. Flag when p*FN_COST > (1-p)*FP_COST, i.e.
    p > FP_COST / (FP_COST + FN_COST)
We compare the threshold found on the data with this theoretical one.
"""

import numpy as np
import pandas as pd

# Illustrative assumptions in abstract "cost units". Replace with real business values.
DEFAULT_FP_COST = 1.0
DEFAULT_FN_COST = 5.0
COST_RATIOS = (1, 2, 3, 5, 10)                     # FN_COST / FP_COST scenarios for sensitivity
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)   # 0.05, 0.06, ..., 0.95
NEAR_OPTIMAL_TOLERANCE = 0.01                      # thresholds within 1% of the minimum cost


def confusion_at(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_table(y_true, proba, fp_cost: float = DEFAULT_FP_COST, fn_cost: float = DEFAULT_FN_COST,
                    thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        c = confusion_at(y_true, proba, t)
        flagged = c["tp"] + c["fp"]
        precision = c["tp"] / flagged if flagged else np.nan     # undefined when nobody is flagged
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if flagged and precision + recall > 0 else 0.0)
        rows.append({"threshold": float(t), **c, "flagged": flagged, "precision": precision,
                     "recall": recall, "f1": f1, "cost": c["fp"] * fp_cost + c["fn"] * fn_cost})
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> dict:
    """Lowest total cost; ties go to the HIGHER threshold (fewer customers contacted).

    Also reports the band of thresholds within 1% of the minimum cost. A wide
    band means the exact choice matters little.
    """
    best_cost = table["cost"].min()
    best = table[table["cost"] == best_cost].iloc[-1]
    near = table[table["cost"] <= best_cost * (1 + NEAR_OPTIMAL_TOLERANCE)]["threshold"]
    return {"threshold": float(best["threshold"]), "cost": float(best_cost),
            "near_optimal_min": float(near.min()), "near_optimal_max": float(near.max()),
            **{k: (None if pd.isna(best[k]) else float(best[k])) for k in ("precision", "recall", "f1")},
            **{k: int(best[k]) for k in ("tp", "fp", "fn", "tn")}}


def theoretical_threshold(fp_cost: float, fn_cost: float) -> float:
    return fp_cost / (fp_cost + fn_cost)


def cost_sensitivity(y_true, proba, fp_cost: float = DEFAULT_FP_COST, ratios=COST_RATIOS) -> pd.DataFrame:
    """Best threshold for each FN:FP cost ratio, next to the 0.5 default."""
    rows = []
    for ratio in ratios:
        fn_cost = fp_cost * ratio
        table = threshold_table(y_true, proba, fp_cost, fn_cost)
        best = select_threshold(table)
        at_half = table[np.isclose(table["threshold"], 0.5)].iloc[0]
        rows.append({
            "fn_to_fp_cost_ratio": ratio, "fp_cost": fp_cost, "fn_cost": fn_cost,
            "theoretical_threshold": theoretical_threshold(fp_cost, fn_cost),
            "selected_threshold": best["threshold"],
            "near_optimal_min": best["near_optimal_min"], "near_optimal_max": best["near_optimal_max"],
            "cost_at_selected": best["cost"], "cost_at_0_5": float(at_half["cost"]),
            "saving_vs_0_5_pct": float(100 * (at_half["cost"] - best["cost"]) / at_half["cost"]),
            "precision": best["precision"], "recall": best["recall"],
            "flagged": best["tp"] + best["fp"], "tp": best["tp"], "fp": best["fp"], "fn": best["fn"],
        })
    return pd.DataFrame(rows)
