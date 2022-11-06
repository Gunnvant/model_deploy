import os
import pandas as pd
from sklearn.model_selection import train_test_split
from train_model import cat_features, save_assets
from ml import data, model

dat = pd.read_csv('./data/cleaned.csv')
train, test = train_test_split(dat, test_size=0.20)


def test_cleaned_data_dim():
    assert dat.shape[0] == 32561
    assert dat.shape[1] == 12


def test_process_data():
    res = data.process_data(train,
                            categorical_features=cat_features,
                            label="salary",
                            training=True)
    assert len(res) == 4


def test_model_function():
    X_train, y_train, encoder, lb = data.process_data(train,
                                                      cat_features,
                                                      label="salary",
                                                      training=True)
    clf = model.train_model(X_train, y_train)
    save_assets(clf, encoder, lb)
    assert os.path.exists("./model/clf.pkl")
    assert os.path.exists("./model/lb.pkl")
    assert os.path.exists("./model/encoder.pkl")
