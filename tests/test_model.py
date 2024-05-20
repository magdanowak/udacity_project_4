import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    train_model, compute_model_metrics, inference
)


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


@pytest.fixture
def processed_data():
    """Return a tuple containing X and y processed data."""
    data = pd.read_csv("data/census.csv")
    X, y, _, _ = process_data(
        data, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    return (X, y)


def test_train_model(processed_data):
    """Positive test that a model is trained."""
    result = train_model(process_data[0], processed_data[1])
    assert isinstance(result, RandomForestClassifier)
    
@pytest.mark.parametrize(
    "y, preds, expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], (1, 1, 1)),
        ([1, 0, 1, 0], [0, 1, 0, 1], (0, 0, 0)),
        ([1, 0, 1], [0, 1, 1], (0.5, 0.5, 0.5)),
    ]
    )
def test_compute_model_metrics(y, preds, expected):
    """Positive test that metrics are calculated properly."""
    y = np.array(y)
    preds = np.array(preds)
    
    result = compute_model_metrics(y, preds)
    assert result == expected
    