"""Phase 2: run the full EDA + SQL analysis and write every output.

Usage (from the project root):
    python -m churn_platform.eda

Steps:
  1. Load the raw CSV (as text) and record its SHA-256.
  2. Run the Phase 1 validator; stop if it reports any ERROR.
  3. Build the typed analysis frame (raw frame untouched).
  4. Compute summary tables  -> reports/tables/*.csv
  5. Draw charts             -> reports/figures/*.png
  6. Build SQLite DB, run SQL, compare with pandas
                             -> data/processed/telco_analysis.db
  7. Write reports/phase2_summary.json and reports/phase2_findings.md
  8. Re-hash the raw CSV to prove it was not modified.

Exit code 1 if validation fails, SQL disagrees with pandas, or the raw file changed.
"""

import json
import sys

from churn_platform import analysis as A
from churn_platform import plots
from churn_platform.config import ANALYSIS_DB_FILE, FIGURES_DIR, RAW_DATA_FILE, REPORTS_DIR, TABLES_DIR
from churn_platform.download import sha256_of
from churn_platform.findings import write_findings
from churn_platform.sql_analysis import build_and_query, compare_with_python
from churn_platform.validation import load_raw, validate


def main() -> int:
    sha_before = sha256_of(RAW_DATA_FILE)
    raw = load_raw(RAW_DATA_FILE)

    errors = [i for i in validate(raw) if i.severity == "ERROR"]
    if errors:
        print("Validation failed; fix the data first (python -m churn_platform.inspect_data).")
        return 1

    df = A.to_analysis_frame(raw)
    target = A.target_summary(df)
    overall = target["churn_rate"]

    # --- Tables ---------------------------------------------------------
    tables = {
        "contract": A.churn_rate_by(df, "Contract", ["Month-to-month", "One year", "Two year"]),
        "tenure_group": A.churn_rate_by(df, "tenure_group", A.TENURE_GROUPS),
        "payment_method": A.churn_rate_by(df, "PaymentMethod"),
        "internet_service": A.churn_rate_by(df, "InternetService", ["Fiber optic", "DSL", "No"]),
        "monthly_charges_band": A.churn_rate_by(df, "monthly_charges_band", A.CHARGE_BANDS),
        "senior_citizen": A.churn_rate_by(df, "SeniorCitizen", ["0", "1"]),
        "contract_internet": A.churn_rate_by(df, ["Contract", "InternetService"]),
    }
    numeric = A.numeric_summary(df)
    correlations = A.numeric_correlations(df)
    categorical = A.categorical_churn_rates(df)
    dup_table, dup_summary = A.duplicate_profiles(df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"churn_by_{name}.csv", index=False)
    numeric.to_csv(TABLES_DIR / "numeric_summary.csv", index=False)
    correlations.to_csv(TABLES_DIR / "numeric_correlations.csv")
    categorical.to_csv(TABLES_DIR / "categorical_churn_rates.csv", index=False)
    dup_table.to_csv(TABLES_DIR / "duplicate_profiles.csv", index=False)

    # --- Figures --------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots.plot_churn_distribution(target, FIGURES_DIR / "churn_distribution.png")
    plots.plot_numeric_distributions(df, FIGURES_DIR / "numeric_distributions.png")
    rate_charts = [
        ("contract", "Contract", "Churn rate by contract type"),
        ("tenure_group", "tenure_group", "Churn rate by tenure group (months)"),
        ("payment_method", "PaymentMethod", "Churn rate by payment method"),
        ("internet_service", "InternetService", "Churn rate by internet service"),
        ("monthly_charges_band", "monthly_charges_band", "Churn rate by monthly charges ($)"),
        ("senior_citizen", "SeniorCitizen", "Churn rate by senior-citizen flag (1 = senior)"),
    ]
    for key, column, title in rate_charts:
        plots.plot_rate_bars(tables[key], column, title, overall, FIGURES_DIR / f"churn_by_{key}.png")
    plots.plot_all_categorical_rates(categorical, overall, FIGURES_DIR / "churn_by_all_categorical_features.png")
    plots.plot_segment_heatmap(tables["contract_internet"], FIGURES_DIR / "churn_by_contract_and_internet.png")

    # --- SQL ------------------------------------------------------------
    sql_results = build_and_query(raw)
    sql_checks = compare_with_python(sql_results, df)

    # --- Reports --------------------------------------------------------
    results = {
        "file": RAW_DATA_FILE.name, "sha256": sha_before, "target": target,
        "quality": A.data_quality_summary(raw, df), "duplicates": dup_summary,
        "tables": tables, "numeric_summary": numeric, "correlations": correlations,
        "sql_checks": sql_checks,
    }
    summary = {k: results[k] for k in ["file", "sha256", "target", "quality", "duplicates", "sql_checks"]}
    (REPORTS_DIR / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_findings(results, REPORTS_DIR / "phase2_findings.md")

    # --- Console summary ------------------------------------------------
    raw_unchanged = sha256_of(RAW_DATA_FILE) == sha_before
    sql_ok = all(c["passed"] for c in sql_checks)
    print(f"Customers: {target['customers']:,}  churned: {target['churned']:,}  churn rate: {overall:.4f}")
    for key in ["contract", "tenure_group", "payment_method", "internet_service"]:
        t = tables[key]
        col = t.columns[0]
        print(f"  {key:17}", ", ".join(f"{v}={r:.1%}" for v, r in zip(t[col], t["churn_rate"])))
    print(f"Duplicate profiles: {dup_summary['groups']} groups, {dup_summary['rows']} rows, "
          f"{dup_summary['conflicting_groups']} with mixed outcomes")
    for c in sql_checks:
        print(f"  SQL {c['query']:32} {'match' if c['passed'] else 'MISMATCH ' + str(c['problems'])}")
    print(f"Tables:   {TABLES_DIR}")
    print(f"Figures:  {FIGURES_DIR} ({len(list(FIGURES_DIR.glob('*.png')))} PNG files)")
    print(f"Database: {ANALYSIS_DB_FILE}")
    print(f"Report:   {REPORTS_DIR / 'phase2_findings.md'}")
    print(f"Raw CSV unchanged: {raw_unchanged}")
    ok = sql_ok and raw_unchanged
    print("RESULT:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
