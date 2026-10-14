# Customer Churn Prediction & SQL Analytics

A small, reproducible data science project using Python, scikit-learn, SQLite, and Jupyter. It focuses on explaining a practical workflow rather than building an app.

## Business problem

A telecom provider wants to identify customers likely to leave, so a retention team can prioritize research or outreach. Missing a churner may lose revenue; contacting a customer who would stay uses time or offer budget. The right model therefore depends on recall, precision, and outreach costs, not accuracy alone.

## Dataset and provenance

The file in [`data/Telco-Customer-Churn.csv`](data/Telco-Customer-Churn.csv) is the public [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). It was downloaded from IBM's [raw CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) without changing its contents. SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. You can verify the file with `Get-FileHash data/Telco-Customer-Churn.csv -Algorithm SHA256` in PowerShell or `sha256sum data/Telco-Customer-Churn.csv` on macOS/Linux.

To obtain it again, save that raw CSV URL as `data/Telco-Customer-Churn.csv`, or run this inside `customer-churn`:

```powershell
Invoke-WebRequest 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv' -OutFile 'data/Telco-Customer-Churn.csv'
```

The CSV has 7,043 customer rows. Its historical `Churn` label (`Yes`/`No`) is the target; contract, tenure, services, and charges are among the predictors. It is a sample dataset, not a current population of real customers.

## Technology stack

Python 3.11+, pandas, NumPy, scikit-learn, matplotlib, SQLite via Python's `sqlite3`, and Jupyter Notebook. pytest checks the notebook's actual code on a fresh temporary copy of the data.

## Install and run

Use Python 3.11 or newer. From the `customer-churn` directory, create and use a **project-local virtual environment**. All dependency installation commands below explicitly target `.venv`; no global package installation is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name customer-churn --display-name 'Python (customer-churn .venv)'
.\.venv\Scripts\jupyter-lab.exe
```

If your machine has multiple Python installations, replace `python` in the first command with the full path to a Python 3.11+ interpreter. On macOS/Linux use `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, `.venv/bin/python -m ipykernel install --prefix .venv --name customer-churn --display-name 'Python (customer-churn .venv)'`, and `.venv/bin/jupyter-lab`. Open `churn_analysis.ipynb`, choose **Python (customer-churn .venv)**, and **Run All**. The notebook creates `data/churn.sqlite` locally. This generated database is gitignored. Run the notebook from this directory so its relative data and SQL paths resolve.

Run every cell in the notebook to rebuild `data/churn.sqlite`, execute the five queries in [`analysis.sql`](analysis.sql), and write [`outputs/test_predictions.csv`](outputs/test_predictions.csv). The prediction CSV has the original customer ID, actual 0/1 churn label, logistic-regression prediction, and predicted churn probability for the 1,409 test rows. IDs are matched by the preserved test-row index and are never model features.

To run the existing tests, use `.\.venv\Scripts\python.exe -m pytest -q tests` on Windows or `.venv/bin/python -m pytest -q tests` on macOS/Linux. The tests execute notebook code in a temporary directory and do not depend on saved notebook outputs.

## Methodology

1. Inspect all 21 source columns, shape, missing data, duplicates, churn rate, and numeric summaries. Show three charts: contract churn rate, bill distribution by churn outcome, and churn rate by bill band.
2. Convert the 11 blank `TotalCharges` strings to missing numeric values. Keep the raw CSV intact. Exclude `customerID` from modeling.
3. Make an 80/20 stratified split (`random_state=42`). Inside each model pipeline, impute missing numeric values with the **training** median, scale numeric features, impute categorical features with the training mode, and one-hot encode them. This prevents test information from entering fitted preprocessing.
4. Fit only logistic regression and a 200-tree random forest, both with fixed seeds. Evaluate the same holdout with accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix. No threshold or model hyperparameters are tuned on the holdout.
5. Split each real CSV row into `customers` and `subscriptions` tables, preserving `customerID` as a one-to-one key. Run the five queries in [`analysis.sql`](analysis.sql).

## Actual model performance

Measured by executing the full notebook with Python 3.13.5, pandas 2.3.3, and scikit-learn 1.9.1 on the committed dataset. The stratified split has 5,634 training rows and 1,409 test rows. Churn prevalence is 26.54% overall. Scores below use the positive class `Churn = Yes` and a 0.5 decision threshold.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 | 926 | 109 | 165 | 209 |
| Random Forest | 0.776 | 0.600 | 0.465 | 0.524 | 0.819 | 919 | 116 | 200 | 174 |

On this holdout, logistic regression is stronger on all five reported scores. It caught 35 more churners and made 7 fewer false-positive flags than the forest. For a team that strongly values finding potential churners, compare **recall and false negatives**; for a team with expensive outreach, compare **precision and false positives**. Here logistic regression is preferable under either of those simple objectives, but a real decision still needs the costs of missed churn and outreach, and preferably validation on later customers. ROC-AUC reflects ranking over thresholds; it is not the precision of a specific contact list.

## SQL insights

The notebook created 7,043 rows in each table, with no foreign-key violations, and executed all five queries in [`analysis.sql`](analysis.sql). Measured examples:

- **Contract aggregation:** month-to-month customers had 42.71% churn (1,655 of 3,875), compared with 11.27% for one-year contracts and 2.83% for two-year contracts.
- **HAVING:** among internet-service groups of at least 500 customers, fiber optic was the only one above the query's 25% churn cutoff: 41.89% among 3,096 customers.
- **JOIN:** the month-to-month segment had 43.74% churn among women and 41.69% among men. These descriptive rates do not isolate a gender effect.
- **CTE bill bands:** churn was 15.74% below $50/month, 28.86% from $50 to below $80, and 33.99% at $80 or above. These are conditional associations, not a price experiment.
- **ROW_NUMBER:** the query returns the three highest monthly bills within each contract, with `customerID` breaking ties. High billing alone does not establish likely churn.

## Limitations

- This is an observational sample; contract and billing associations do not prove causes of churn.
- The file does not establish feature observation times or intervention outcomes. We cannot claim a retention offer would work.
- One stratified split gives an understandable baseline, but scores can change with new customers or a different time period.
- The 0.5 classification threshold is fixed for teaching. A real outreach program should select a threshold using validation data and explicit cost/capacity constraints, then test it on fresh data.
- The descriptive EDA uses all rows, including the eventual holdout. Modeling choices here were fixed before examining held-out model results; repeated iteration based on that holdout would invalidate it.
