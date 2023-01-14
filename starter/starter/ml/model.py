from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model: RandomForestClassifier
        Trained machine learning model.
    """

    clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def get_performance_on_slices(test, y_test, preds):
    test = test.drop("salary", axis=1)
    categorical_columns = test.select_dtypes("object").columns

    metrics_on_slices = []
    for col in categorical_columns:
        categories = test[col].unique()
        for cat in categories:
            mask = test[col] == cat
            results = compute_model_metrics(y_test[mask], preds[mask])
            metrics_on_slices.append(
                {
                    "category": col,
                    "group": cat,
                    "precision": results[0],
                    "recall": results[1],
                    "f1": results[2],
                }
            )
    return metrics_on_slices
