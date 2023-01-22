# Script to train machine learning model.
import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    get_performance_on_slices,
    inference,
    train_model,
)


def main(data_filepath, model_filepath, reports_filepath):

    data = pd.read_csv(data_filepath)

    # Add code to load in the data.

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    clf = train_model(X_train, y_train)

    Path(model_filepath).mkdir(exist_ok=True)
    dump(encoder, model_filepath + "/encoder.joblib")
    dump(lb, model_filepath + "/binarizer.joblib")
    dump(clf, model_filepath + "/model.joblib")

    preds = inference(clf, X_test)

    metrics = compute_model_metrics(y_test, preds)

    metrics_on_slices = get_performance_on_slices(test, y_test, preds)

    metrics_on_slices.insert(
        0,
        {
            "category": "all categories",
            "group": "all groups",
            "precision": metrics[0],
            "recall": metrics[1],
            "f1": metrics[2],
        },
    )

    metrics_df = pd.DataFrame.from_records(metrics_on_slices)

    Path(reports_filepath).mkdir(exist_ok=True)
    metrics_df.to_csv(reports_filepath + "/metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "-d",
        "--data-filepath",
        type=str,
        default="../data/census_clean.csv",
        help="Path to the training model data.",
    )

    parser.add_argument(
        "-a",
        "--model-filepath",
        type=str,
        default="../model",
        help="Path to the trained model and preprocess pipeline.",
    )

    parser.add_argument(
        "-r",
        "--reports-filepath",
        type=str,
        default="../reports",
        help="Path to the reports.",
    )

    args = parser.parse_args()
    main(**vars(args))
