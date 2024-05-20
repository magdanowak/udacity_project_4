# Script to train machine learning model.
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, calculate_metrics_on_slices

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

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

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)
with open("model/encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)
with open("model/lb.pkl", "wb") as file:
    pickle.dump(lb, file)

# Model metrics on slices of data
metrics = calculate_metrics_on_slices(model, test, X_test, y_test, cat_features)
metrics.to_csv("slice_output.txt", index=False)
