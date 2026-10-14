-- Run after the notebook creates data/churn.sqlite.
-- Churn is stored as 1 (Yes) or 0 (No). Rates are percentages.

-- 1. GROUP BY: size, churn rate and average bill for each contract.
SELECT s.Contract,
       COUNT(*) AS customers,
       SUM(c.Churn) AS churned,
       ROUND(100.0 * AVG(c.Churn), 2) AS churn_rate_pct,
       ROUND(AVG(s.MonthlyCharges), 2) AS avg_monthly_charges
FROM customers AS c
JOIN subscriptions AS s USING (customerID)
GROUP BY s.Contract
ORDER BY churn_rate_pct DESC;

-- 2. HAVING: sizeable internet-service segments with above-25% churn.
SELECT s.InternetService,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(c.Churn), 2) AS churn_rate_pct
FROM customers AS c
JOIN subscriptions AS s USING (customerID)
GROUP BY s.InternetService
HAVING COUNT(*) >= 500 AND AVG(c.Churn) > 0.25
ORDER BY churn_rate_pct DESC;

-- 3. JOIN: contract and demographic information live in separate tables.
SELECT s.Contract, c.gender,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(c.Churn), 2) AS churn_rate_pct
FROM customers AS c
JOIN subscriptions AS s ON c.customerID = s.customerID
GROUP BY s.Contract, c.gender
ORDER BY s.Contract, c.gender;

-- 4. CTE: label bill bands, then summarize their conditional churn rates.
WITH bill_bands AS (
    SELECT c.Churn,
           CASE WHEN s.MonthlyCharges < 50 THEN 'Under $50'
                WHEN s.MonthlyCharges < 80 THEN '$50 to <$80'
                ELSE '$80 or more' END AS bill_band
    FROM customers AS c
    JOIN subscriptions AS s USING (customerID)
)
SELECT bill_band, COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn), 2) AS churn_rate_pct
FROM bill_bands
GROUP BY bill_band
ORDER BY CASE bill_band WHEN 'Under $50' THEN 1
                        WHEN '$50 to <$80' THEN 2 ELSE 3 END;

-- 5. Window function: highest monthly bills within each contract.
-- customerID breaks ties so ROW_NUMBER is deterministic.
WITH ranked AS (
    SELECT s.Contract, s.customerID, s.MonthlyCharges, c.Churn,
           ROW_NUMBER() OVER (
               PARTITION BY s.Contract
               ORDER BY s.MonthlyCharges DESC, s.customerID
           ) AS bill_rank
    FROM subscriptions AS s
    JOIN customers AS c USING (customerID)
)
SELECT Contract, customerID, MonthlyCharges, Churn, bill_rank
FROM ranked
WHERE bill_rank <= 3
ORDER BY Contract, bill_rank;
