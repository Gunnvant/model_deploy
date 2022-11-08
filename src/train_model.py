# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
# Add the necessary imports for the starter code.
from ml import data, model
import logging
from constants import cat_features
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# Add code to load in the data.
logging.info("Reading data")
dat = pd.read_csv("./data/cleaned.csv")
# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
logging.info("Doing train-test data split")
train, test = train_test_split(dat, test_size=0.20)

# Process the test data with the process_data function.
logging.info("Processing data and creating encoders")
X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


def save_assets(clf, encoder, lb):
    asset_names = ['clf.pkl', 'encoder.pkl', 'lb.pkl']
    for asset, name in zip((clf, encoder, lb), asset_names):
        with open(f'./model/{name}', 'wb') as f:
            pickle.dump(asset, f)


def get_slice_performance(dat, clf, encoder, lb):
    slice_performance = []
    for cat in cat_features:
        for slice in dat[cat].unique():
            X = dat[dat[cat] == slice]
            X_slice, y_slice, _, _ = data.process_data(X,
                                                       cat_features,
                                                       label="salary",
                                                       training=False,
                                                       encoder=encoder,
                                                       lb=lb
                                                       )
            preds_slice = clf.predict(X_slice)
            precision, recall, fbeta = model.compute_model_metrics(
                y_slice, preds_slice)
            slice_performance.append([slice, precision, recall, fbeta])
    slice_performance = pd.DataFrame(
        slice_performance, columns=[
            'slice', 'precsion', 'recall', 'fbeta'])
    return slice_performance


# Train and save a model.
logging.info("Model training started...")
clf = model.train_model(X_train, y_train)
logging.info("Model training done. Saving assets")
save_assets(clf, encoder, lb)
logging.info("Generating Model Slices")
slice_performance = get_slice_performance(dat, clf, encoder, lb)
logging.info("Writing Slice Performance")
slice_performance.to_csv("./data/slice_performance.csv", index=False)
