import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


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
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model


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
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def calculate_metrics_on_slices(model, data, X, y, cat_features):
    """Calculate metrics for a model on slices by a given feature."""
    result = []

    for feature in cat_features:
        slice_values = data[feature].unique()
        for slice_value in slice_values:
            slice_filter = data[feature] == slice_value
            X_slice = X[slice_filter]
            y_slice = y[slice_filter]
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            result.append(
                {
                    "feature": feature,
                    "slice": slice_value,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )
    return pd.DataFrame(result)
