"""Small integration checks against the notebook's real code and CSV."""

import os
import shutil
import sqlite3
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    # Fresh namespace and temporary files avoid relying on saved outputs.
    work = tmp_path_factory.mktemp("churn_notebook")
    (work / "data").mkdir()
    shutil.copy2(ROOT / "data" / "Telco-Customer-Churn.csv", work / "data")
    shutil.copy2(ROOT / "analysis.sql", work)
    notebook = nbformat.read(ROOT / "churn_analysis.ipynb", as_version=4)
    previous = Path.cwd()
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    original_show = plt.show
    plt.show = lambda: None  # Figures are created; no interactive window in pytest.
    namespace = {"__name__": "__main__"}
    try:
        os.chdir(work)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, "churn_analysis.ipynb", "exec"), namespace)
        yield namespace, work
    finally:
        plt.show = original_show
        os.chdir(previous)


def test_cleaning_and_conversion(run):
    values, _ = run
    raw, clean = values["raw"], values["df"]
    assert raw.shape == (7043, 21)
    assert raw["customerID"].is_unique
    assert raw["TotalCharges"].dtype == object
    assert pd.api.types.is_numeric_dtype(clean["TotalCharges"])
    assert clean["TotalCharges"].isna().sum() == 11
    assert clean["Churn"].isin([0, 1]).all()
    assert len(clean) == len(raw)


def test_preprocessing_uses_train_only(run):
    values, _ = run
    train, test = values["X_train"], values["X_test"]
    assert len(train) == 5634 and len(test) == 1409
    assert set(train.index).isdisjoint(test.index)
    assert "customerID" not in train and "charge_band" not in train
    for pipeline in values["models"].values():
        transform = pipeline.named_steps["preprocess"]
        numeric = transform.named_transformers_["numeric"]
        medians = numeric.named_steps["imputer"].statistics_
        np.testing.assert_allclose(medians, train[values["numeric_features"]].median().to_numpy())
        scaled_train = numeric.transform(train[values["numeric_features"]])
        np.testing.assert_allclose(scaled_train.mean(axis=0), 0, atol=1e-12)
        changed = test.iloc[[0]].copy()
        changed["Contract"] = "Unseen contract"
        assert transform.transform(changed).shape[0] == 1


def test_metrics_and_prediction_alignment(run):
    values, work = run
    exported = pd.read_csv(work / "outputs" / "test_predictions.csv")
    assert list(exported) == ["customerID", "actual_churn", "predicted_churn",
                              "predicted_churn_probability"]
    assert exported["customerID"].tolist() == values["raw"].loc[values["X_test"].index, "customerID"].tolist()
    assert exported["actual_churn"].tolist() == values["y_test"].tolist()
    assert exported["customerID"].is_unique
    assert exported["predicted_churn_probability"].between(0, 1).all()
    for name, row in values["comparison"].iterrows():
        tn, fp, fn, tp = (int(row[key]) for key in ("TN", "FP", "FN", "TP"))
        assert tn + fp + fn + tp == len(values["y_test"])
        assert row["accuracy"] == pytest.approx((tp + tn) / (tn + fp + fn + tp))
        assert row["precision"] == pytest.approx(tp / (tp + fp))
        assert row["recall"] == pytest.approx(tp / (tp + fn))
        assert row["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
        probability = values["models"][name].predict_proba(values["X_test"])[:, 1]
        assert row["roc_auc"] == pytest.approx(roc_auc_score(values["y_test"], probability))
    logistic = values["models"]["Logistic Regression"]
    np.testing.assert_array_equal(exported["predicted_churn"], logistic.predict(values["X_test"]))
    np.testing.assert_allclose(exported["predicted_churn_probability"],
                               logistic.predict_proba(values["X_test"])[:, 1])


def test_sqlite_integrity_and_reports(run):
    _, work = run
    with sqlite3.connect(work / "data" / "churn.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 7043
        assert connection.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 7043
        assert connection.execute("SELECT COUNT(*) FROM subscriptions WHERE TotalCharges IS NULL").fetchone()[0] == 11
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        statements = [text.strip() for text in (work / "analysis.sql").read_text().split(";") if text.strip()]
        assert len(statements) == 5
        counts = [len(connection.execute(statement).fetchall()) for statement in statements]
        assert counts == [3, 1, 6, 3, 9]
