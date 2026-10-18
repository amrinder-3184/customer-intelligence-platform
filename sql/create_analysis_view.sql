-- Typed analysis view over the raw table.
--
-- customers_raw holds the CSV exactly as text (loaded by
-- churn_platform/sql_analysis.py). This view converts types and derives
-- segments IN SQL, independently of the pandas code, so that comparing the
-- two is a genuine cross-check rather than the same calculation twice.
--
-- Note: SQLite's CAST('abc' AS REAL) returns 0 instead of failing. That is
-- safe here only because the Phase 1 validator rejects non-numeric values
-- before this view is ever built.

DROP VIEW IF EXISTS customers;

CREATE VIEW customers AS
SELECT
    customerID                                  AS customer_id,  -- identifier only, never a predictor
    gender,
    CAST(SeniorCitizen AS INTEGER)              AS senior_citizen,
    Partner                                     AS partner,
    Dependents                                  AS dependents,
    CAST(tenure AS INTEGER)                     AS tenure,
    PhoneService                                AS phone_service,
    MultipleLines                               AS multiple_lines,
    InternetService                             AS internet_service,
    OnlineSecurity                              AS online_security,
    OnlineBackup                                AS online_backup,
    DeviceProtection                            AS device_protection,
    TechSupport                                 AS tech_support,
    StreamingTV                                 AS streaming_tv,
    StreamingMovies                             AS streaming_movies,
    Contract                                    AS contract,
    PaperlessBilling                            AS paperless_billing,
    PaymentMethod                               AS payment_method,
    CAST(MonthlyCharges AS REAL)                AS monthly_charges,
    -- Blank (" ") becomes NULL, not 0: missingness stays visible.
    CAST(NULLIF(TRIM(TotalCharges), '') AS REAL) AS total_charges,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END   AS churned,
    -- Same boundaries as TENURE_GROUPS in churn_platform/analysis.py.
    CASE
        WHEN CAST(tenure AS INTEGER) <= 12 THEN '0-12'
        WHEN CAST(tenure AS INTEGER) <= 24 THEN '13-24'
        WHEN CAST(tenure AS INTEGER) <= 48 THEN '25-48'
        ELSE '49+'
    END                                         AS tenure_group,
    -- Same boundaries as CHARGE_BANDS in churn_platform/analysis.py.
    CASE
        WHEN CAST(MonthlyCharges AS REAL) < 35 THEN '<35'
        WHEN CAST(MonthlyCharges AS REAL) < 70 THEN '35-70'
        WHEN CAST(MonthlyCharges AS REAL) < 90 THEN '70-90'
        ELSE '90+'
    END                                         AS monthly_charges_band
FROM customers_raw;
