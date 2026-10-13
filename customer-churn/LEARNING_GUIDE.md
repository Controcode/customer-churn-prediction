# Learning guide: customer churn, from CSV to decision

## 1. Follow the complete workflow

1. **Define the target.** `Churn = Yes` means a customer left. Predict a future customer's likelihood of leaving from account information. This sample is a snapshot, so we cannot establish the exact timing of every feature relative to churn.
2. **Inspect the CSV.** Check shape, types, missing cells, duplicate rows and IDs. `TotalCharges` looks numeric but contains 11 blanks; convert it with `pd.to_numeric(..., errors="coerce")`. Keep the original file intact.
3. **Describe patterns.** Calculate the overall churn proportion and churn conditional on contract and bill band. Plot contract rates, bill distributions by outcome, and bill-band rates. These reveal associations, not causes.
4. **Split once.** Reserve 20% with `stratify=y` and `random_state=42`. Stratification keeps the minority churn class represented in both partitions.
5. **Preprocess and fit.** Exclude `customerID`. For four numeric features, fill missing values with the training median and standardize. For categorical features, fill missing values with the training mode and one-hot encode. Each model's pipeline fits these steps only on its training input.
6. **Evaluate on the holdout.** Report the same six metrics and confusion matrix for logistic regression and random forest. Export logistic-regression test predictions with customer IDs matched by the preserved row index; the ID is never a feature. Do not tune thresholds or features based on these test results; that would turn the test into a validation set.
7. **Analyze in SQL.** Split the same CSV rows into customer and subscription tables by the genuine `customerID`. Verify both row counts and the foreign key. Run the five queries in `analysis.sql`.

## 2. Statistics used in EDA

- **Mean:** sum divided by count. For bills `$20, $30, $100`, mean = `$50`; the high bill pulls it upward.
- **Median:** middle value after sorting. The same bills have median `$30`, often a better description when values are skewed.
- **Sample standard deviation:** `sqrt(sum((x - mean)^2) / (n - 1))`. It describes spread in the original units; a larger value means bills vary more. pandas `std()` uses `n - 1` by default.
- **Overall churn rate:** `number who churned / all customers`.
- **Conditional churn rate:** `P(churn | contract = month-to-month) = churned month-to-month customers / all month-to-month customers`. Group sizes matter: a 50% rate among 2 people is less stable than 50% among 2,000.
- **Association versus causation:** A high churn rate in an expensive bill band does not prove that the price caused departure. Contract, tenure, service mix, and unobserved factors may differ too. This is observational sample data.

## 3. Logistic regression and the sigmoid

For feature vector `x`, logistic regression makes a linear score `z = b + w1*x1 + ... + wp*xp`. It converts that score into a probability with the sigmoid:

`P(Churn = 1 | x) = σ(z) = 1 / (1 + e^(-z))`.

For example, `z = 0` gives `0.5`; `z ≈ 2.2` gives about `0.90`; `z ≈ -2.2` gives about `0.10`. At the notebook's fixed 0.5 threshold, probabilities at least 0.5 are predicted as churn. During training, coefficients are chosen to reduce log loss, with scikit-learn's default regularization. A coefficient changes log-odds when its feature changes by one encoded/scaled unit, holding other features fixed; it is not a causal effect. Correlated one-hot and billing variables complicate interpretation.

## 4. Random forests

A decision tree repeatedly divides rows using rules such as `tenure < 12`. A random forest trains many trees on bootstrap samples and considers a random subset of features at each split. For classification it combines the trees' votes/probabilities. This can capture nonlinear effects and interactions (for example, a bill may matter differently for short and long tenure). Its limits include less direct interpretability and possible overfitting on small or noisy datasets. We use 200 trees with a fixed seed, without hyperparameter tuning. The preprocessing is shared for a fair, simple comparison; trees themselves usually do not need scaling.

## 5. Encoding, scaling, missing values, leakage

- **One-hot encoding:** Turn `Contract = Month-to-month / One year / Two year` into indicator columns. Assigning 0/1/2 would falsely imply equal ordered steps. `handle_unknown='ignore'` lets a new category at test time produce all-zero indicators instead of an error.
- **Standard scaling:** `x_scaled = (x - training_mean) / training_standard_deviation`. Logistic regression's optimization and regularization behave better when numerical units are comparable. A `$100` charge and `1` senior-citizen indicator then do not differ merely due to units.
- **Imputation:** Use the training median for missing numeric `TotalCharges`; use the training mode for missing categories. Eleven blank totals are genuine missing values after conversion, not zeros. In production, the reason for missing totals deserves investigation.
- **Data leakage:** Fitting imputation, scaling, or encoding on the whole dataset lets test information influence training. So does choosing a threshold after checking test scores. A pipeline fitted on `X_train` computes each statistic there and only transforms `X_test` with those stored values.
- **Identifiers:** `customerID` labels records but does not describe customer behavior. Excluding it avoids learning accidental ID patterns.

## 6. Confusion matrix and metrics

For churn as the positive class, the matrix is:

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actually stayed | True negative (TN) | False positive (FP) |
| Actually churned | False negative (FN) | True positive (TP) |

Imagine `TN=70, FP=10, FN=8, TP=12` on 100 customers. Then:

- **Accuracy:** `(TP + TN) / all = 82%`. This can look good even if churners are missed.
- **Precision:** `TP / (TP + FP) = 12/22 ≈ 54.5%`. Among customers flagged, how many really churned? Useful when calls or offers are expensive.
- **Recall:** `TP / (TP + FN) = 12/20 = 60%`. Among actual churners, how many were caught? Useful when missing a churner is costly.
- **F1:** `2 * precision * recall / (precision + recall) ≈ 57.1%`. A balance of both, but it does not encode business costs.
- **ROC-AUC:** The area under the true-positive-rate versus false-positive-rate curve across thresholds. It measures ranking, not the precision of a particular outreach list. About 0.5 is chance ranking; 1 is perfect ranking.

Changing the threshold usually trades recall for precision. Select it using validation data and an explicit cost or capacity model in a future project, then evaluate once on a separate test set. This project deliberately fixes 0.5.

## 7. The five SQL queries

The notebook writes `customers(customerID PRIMARY KEY, gender, SeniorCitizen, Partner, Dependents, Churn)` and `subscriptions(customerID PRIMARY KEY REFERENCES customers, ...all other original service and billing columns)`. Thus the join is one-to-one and uses existing IDs. The cleaned `TotalCharges` becomes SQL `NULL` in 11 rows; no synthetic business records are added. `Churn` is 0/1, so `AVG(Churn)` is the churn proportion. Multiplying by 100 gives a percentage. SQLite ignores `NULL` in `AVG(TotalCharges)` if queried.

1. **GROUP BY and aggregation:** Count customers and churners and average the monthly bill within each contract. `SUM(Churn)` works because the target is 0/1. Compare both group sizes and percentages.
2. **HAVING:** After grouping by internet-service type, retain only groups with at least 500 customers and churn rate above 25%. `WHERE` filters rows *before* aggregation; `HAVING` filters whole groups *after* aggregation. The 25% and 500 cutoffs are descriptive examples, not model selection.
3. **JOIN:** Combine demographics from `customers` with contract from `subscriptions` on `customerID`, then compare churn by contract and gender. The two tables came from one original row; each ID is preserved and unique.
4. **CTE:** A `WITH bill_bands AS (...)` expression gives each customer a readable monthly-bill band. The outer query computes group counts and conditional churn rates. This separates row-level labeling from aggregation.
5. **ROW_NUMBER:** Within each contract, order customers by monthly bill, breaking ties with `customerID`, and number them. Return the top three per contract. This identifies high-bill records for inspection; it does not imply they will churn or should automatically receive an offer.

## 8. Fifteen realistic internship questions and short answers

1. **What is the business objective?** Rank customers by churn risk so a retention team can prioritize outreach; actual interventions need a cost and effectiveness study.
2. **What is the unit of observation?** One row per original `customerID`, representing one customer in this sample.
3. **How did you check data quality?** I checked types, blank/missing values, exact duplicates, duplicate IDs, and row counts after loading into SQLite.
4. **Why did `TotalCharges` need conversion?** Eleven blank strings caused the CSV column to be read as text; converting to numeric turns them into missing values.
5. **Why not fill blanks with zero?** Zero is a real billing amount and would assert a fact not present in the data; median imputation is a simple training-only baseline.
6. **Why drop `customerID`?** It identifies a row but is not a meaningful behavioral predictor; patterns in the ID could be spurious.
7. **Why stratify the split?** Churn is the minority class, so stratification gives similar class proportions in training and test data.
8. **What would leakage look like here?** Computing the `TotalCharges` median or scaler mean before the split, or using test results to choose a threshold.
9. **Why one-hot encode contract?** Its categories are labels rather than evenly spaced numbers; one-hot indicators avoid an invented numerical order.
10. **Why scale for logistic regression?** The model's regularization and optimization are sensitive to feature magnitudes; tenure and charges use different units.
11. **What does a logistic coefficient mean?** For one scaled/encoded unit, it changes log-odds while holding other variables fixed, not a causal effect.
12. **How does a forest differ from one tree?** Bootstrap sampling and random feature subsets create diverse trees; combining them generally reduces variance.
13. **Why is accuracy insufficient?** A majority-class predictor can look accurate while missing many actual churners; recall and the confusion matrix reveal that.
14. **Which model would you deploy?** That depends on outreach cost, capacity, and missed-churn cost. I would choose a threshold on validation data and confirm on a fresh holdout before deployment.
15. **What is the biggest limitation?** This is a static sample without a confirmed feature-observation time, intervention outcomes, or external validation. It supports a learning benchmark, not a causal retention policy.

## 9. Five concepts to rehearse aloud

Explain the **unit and target**, **missing-data decisions**, **train-only preprocessing**, **precision/recall tradeoff**, and **how SQL grouping and joins reproduce the same descriptive patterns**. Be ready to point to a specific cell or query for each.
