import pytest
import numpy as np
import pandas as pd
from starter.ml.model import (
    train_model,
    compute_model_metrics,
    get_performance_on_slices,
    inference,
)

from sklearn.ensemble import RandomForestClassifier


def test_train_model_valid_input():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([1, 2, 3, 4])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_train_model_empty_input():
    X_train = np.array([])
    y_train = np.array([])
    with pytest.raises(ValueError):
        train_model(X_train, y_train)


def test_compute_model_metrics_valid_input():
    y_true = np.array([1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_compute_model_metrics_empty_input():
    y = np.array([])
    preds = np.array([])
    result = compute_model_metrics(y, preds)
    assert result == (1, 1, 1)


def test_inference_valid_input():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([1, 0, 1, 1])
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)


def test_get_performance_on_slices_valid_input():
    test = pd.DataFrame(
        {
            "workclass": ["Private", "Government", "Self-employed"],
            "education": ["Bachelors", "Masters", "High School"],
            "salary": [">50K", "<=50K", ">50K"],
        }
    )
    y_test = np.array([1, 0, 1])
    preds = np.array([1, 1, 1])
    metrics_on_slices = get_performance_on_slices(test, y_test, preds)
    assert isinstance(metrics_on_slices, list)
    assert all(isinstance(i, dict) for i in metrics_on_slices)
    assert all("category" in i for i in metrics_on_slices)
    assert all("group" in i for i in metrics_on_slices)
    assert all("precision" in i for i in metrics_on_slices)
    assert all("recall" in i for i in metrics_on_slices)
    assert all("f1" in i for i in metrics_on_slices)
