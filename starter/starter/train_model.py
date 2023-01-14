# Script to train machine learning model.
import argparse
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    get_performance_on_slices,
)


def main(data_filepath, artifacts_filepath):

    data = pd.read_csv(data_filepath)

    # Add code to load in the data.

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    print(encoder)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    clf = train_model(X_train, y_train)

    dump(encoder, artifacts_filepath + "/preprocess.joblib")
    dump(clf, artifacts_filepath + "/model.joblib")

    preds = inference(clf, X_test)

    metrics = compute_model_metrics(y_test, preds)

    # metrics_on_slices = get_perfomance_on_slices(test, y_test, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "-d",
        "--data-filepath",
        type=str,
        help="Path to the training model data.",
    )

    parser.add_argument(
        "-a",
        "--artifacts-filepath",
        type=str,
        help="Path to the trained model and featurize pipeline.",
    )

    args = parser.parse_args()
    main(**vars(args))
