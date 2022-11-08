# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from ml import data, model
import pickle
import pandas as pd
import os
from constants import cat_features

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


def load_asset(path):
    '''
    Load pickled assets
    '''
    with open(path, 'rb') as f:
        asset = pickle.load(f)
    return asset


path_model = "./model/clf.pkl"
path_enc = "./model/encoder.pkl"
path_lb = "./model/lb.pkl"

clf = load_asset(path_model)
lb = load_asset(path_lb)
enc = load_asset(path_enc)


class PredictorData(BaseModel):
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       '?',
                       'Self-emp-inc',
                       'Without-pay',
                       'Never-worked']
    education: Literal['Bachelors',
                       'HS-grad',
                       '11th',
                       'Masters',
                       '9th',
                       'Some-college',
                       'Assoc-acdm',
                       'Assoc-voc',
                       '7th-8th',
                       'Doctorate',
                       'Prof-school',
                       '5th-6th',
                       '10th', '1st-4th',
                       'Preschool',
                       '12th']
    marital_status: Literal['Never-married',
                            'Married-civ-spouse',
                            'Divorced',
                            'Married-spouse-absent',
                            'Separated',
                            'Married-AF-spouse',
                            'Widowed']
    occupation: Literal['Adm-clerical',
                        'Exec-managerial',
                        'Handlers-cleaners',
                        'Prof-specialty',
                        'Other-service',
                        'Sales',
                        'Craft-repair',
                        'Transport-moving',
                        'Farming-fishing',
                        'Machine-op-inspct',
                        'Tech-support',
                        '?',
                        'Protective-serv',
                        'Armed-Forces',
                        'Priv-house-serv']
    relationship: Literal['Not-in-family',
                          'Husband',
                          'Wife',
                          'Own-child',
                          'Unmarried',
                          'Other-relative']
    race: Literal['White',
                  'Black',
                  'Asian-Pac-Islander',
                  'Amer-Indian-Eskimo',
                  'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int
    capital_loss: int
    hours_per_week: int


app = FastAPI()


@app.get("/")
async def health():
    '''
    Health check for API
    '''
    return {'status': 'healthy'}


@app.post("/")
async def predict(predict_data: PredictorData):
    '''
    Prediction Endpoint
    '''
    data_dict = {
        'age': [predict_data.age],
        'workclass': [predict_data.workclass],
        'education': [predict_data.education],
        'marital-status': [predict_data.marital_status],
        'occupation': [predict_data.occupation],
        'relationship': [predict_data.relationship],
        'race': [predict_data.race],
        'sex': [predict_data.sex],
        'captial-gain': [predict_data.capital_gain],
        'capital-loss': [predict_data.capital_loss],
        'hours-per-week': [predict_data.hours_per_week]
    }
    df = pd.DataFrame(data_dict)
    X, _, _, _ = data.process_data(df,
                                   cat_features,
                                   training=False,
                                   encoder=enc, lb=lb)
    pred = model.inference(clf, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}
