-- Churn analytics queries over the `customers` view.
--
-- Each query starts with a "-- name: <query_name>" line; the Python runner
-- (churn_platform/sql_analysis.py) splits the file on those markers.
-- All queries are fixed, read-only and parameter-free. No user input is ever
-- inserted into SQL text.
--
-- Churn rate = churned customers / all customers in the group.
-- AVG(churned) gives the same number because churned is 0 or 1.


-- name: overall_churn
-- Q: How many customers churned, and what fraction is that?
SELECT
    COUNT(*)                       AS customers,
    SUM(churned)                   AS churned,
    COUNT(*) - SUM(churned)        AS retained,
    ROUND(AVG(churned), 4)         AS churn_rate
FROM customers;


-- name: churn_by_contract
-- Q: Does churn differ between month-to-month and longer contracts?
SELECT
    contract,
    COUNT(*)                       AS customers,
    SUM(churned)                   AS churned,
    ROUND(AVG(churned), 4)         AS churn_rate
FROM customers
GROUP BY contract
ORDER BY churn_rate DESC;


-- name: churn_by_tenure_group
-- Q: Are newer customers more likely to have churned than long-standing ones?
SELECT
    tenure_group,
    COUNT(*)                       AS customers,
    SUM(churned)                   AS churned,
    ROUND(AVG(churned), 4)         AS churn_rate
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);


-- name: churn_by_payment_method
-- Q: Which payment methods are associated with higher churn?
SELECT
    payment_method,
    COUNT(*)                       AS customers,
    SUM(churned)                   AS churned,
    ROUND(AVG(churned), 4)         AS churn_rate
FROM customers
GROUP BY payment_method
ORDER BY churn_rate DESC;


-- name: monthly_charges_by_churn_status
-- Q: Do customers who churned pay more per month than those who stayed?
SELECT
    CASE churned WHEN 1 THEN 'Churned' ELSE 'Retained' END AS churn_status,
    COUNT(*)                       AS customers,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(MIN(monthly_charges), 2) AS min_monthly_charges,
    ROUND(MAX(monthly_charges), 2) AS max_monthly_charges
FROM customers
GROUP BY churned
ORDER BY churned DESC;


-- name: churn_by_monthly_charges_band
-- Q: How does churn rate change across monthly-charge bands?
SELECT
    monthly_charges_band,
    COUNT(*)                       AS customers,
    SUM(churned)                   AS churned,
    ROUND(AVG(churned), 4)         AS churn_rate
FROM customers
GROUP BY monthly_charges_band
ORDER BY MIN(monthly_charges);


-- name: segment_contract_internet
-- Q: Which contract x internet-service segments are largest, and how do
--    their churn rates compare? (Shows whether one factor explains another.)
SELECT
    contract,
    internet_service,
    COUNT(*)                                              AS customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 2) AS pct_of_customers,
    SUM(churned)                                          AS churned,
    ROUND(AVG(churned), 4)                                AS churn_rate
FROM customers
GROUP BY contract, internet_service
ORDER BY churn_rate DESC;
