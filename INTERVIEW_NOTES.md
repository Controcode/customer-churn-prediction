# Interview notes

## 60-second explanation

I studied customer churn using IBM's 7,043-row Telco sample. I checked types, duplicates and missing values, then explored churn rates by contract and monthly bill. `TotalCharges` contained 11 blank strings, so I converted it to numeric and imputed missing values inside a training-only pipeline. I made a stratified 80/20 split and compared logistic regression with a random forest on the same holdout. Logistic regression did better here: 0.806 accuracy, 0.559 recall, and 0.842 ROC-AUC, versus 0.776, 0.465, and 0.819 for the forest. I also split the original rows into customer and subscription tables in SQLite and wrote five queries for segment analysis. The project demonstrates a reproducible baseline, not proof that any particular retention offer will work.

## 2–3-minute technical explanation

The unit is one customer, keyed by the genuine `customerID`; the target is the binary historical `Churn` label. I confirmed 7,043 rows, no duplicate IDs, and 11 blank `TotalCharges` values. I converted that column with `pd.to_numeric(errors="coerce")`, kept the original CSV intact, and excluded the ID and EDA-only bill band from modeling. The 80/20 split is stratified with seed 42, leaving 5,634 training and 1,409 test rows.

Each model is a scikit-learn pipeline. Numeric features get median imputation and standard scaling; categorical features get mode imputation and one-hot encoding with unknown categories ignored. Pipeline fitting happens only on training rows, which prevents test-set statistics from leaking into the imputer, scaler, or encoder. I trained logistic regression with `max_iter=1000` and a 200-tree random forest, both with fixed seeds. At the default 0.5 threshold, logistic regression had 209 true positives, 165 false negatives, 109 false positives, and 926 true negatives. Its precision was 0.657, recall 0.559, F1 0.604, and ROC-AUC 0.842. The forest had 174 true positives and 200 false negatives, with lower precision, recall, F1, and AUC on this split. Outreach costs and missed-churn costs would determine the actual operating threshold; I did not tune one on the test set.

For SQL, I preserved the original ID while separating customer attributes and subscription attributes into one-to-one tables. Both contain 7,043 rows and pass the foreign-key check. The five queries use grouping, `HAVING`, a join, a CTE, and `ROW_NUMBER` to describe contract, service, demographic, billing-band, and high-bill patterns. The notebook saves only ID, actual label, predicted label, and probability for each held-out customer.

## Five challenging follow-up questions

1. **How do you know preprocessing did not leak test information?** The imputer, scaler, and encoder are inside a pipeline fitted on `X_train`. A test checks that its learned numeric medians equal training medians.
2. **Why did logistic regression beat the forest?** On this split it ranked and classified better, but that does not establish a universal reason. Regularization may suit this dataset, while the untuned forest may overfit; cross-validation and later data would test that explanation.
3. **What if the retention team can contact only 100 customers?** Rank probabilities, choose a cutoff on validation data that respects capacity and expected value, then evaluate that policy on a fresh holdout. The 0.5 threshold is only a baseline.
4. **Could `TotalCharges` leak the outcome?** Possibly, if recorded after a customer has left. The dataset lacks clear feature timestamps, so I would validate measurement timing before production use or test a model without this feature.
5. **Why compare precision and recall as well as ROC-AUC?** AUC describes ranking across thresholds, while precision and recall describe the actual 0.5-threshold decisions and different business errors. Class imbalance makes accuracy insufficient by itself.

## Five limitations or next improvements

1. Confirm when each feature was observed relative to churn; remove any post-outcome feature.
2. Validate on later customer cohorts or an external dataset to assess generalization.
3. Use training-only cross-validation for hyperparameters and threshold selection, keeping a final untouched test set.
4. Estimate outreach cost, customer value, and treatment effect before recommending a retention policy.
5. Check probability calibration and performance across customer segments before operational use.
