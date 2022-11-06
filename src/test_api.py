import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_get_malformed(client):
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post_below(client):
    r = client.post("/", json={
        "age": 0,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 0
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_above(client):
    r = client.post("/", json={
        "age": 52,
        "workclass": "Self-emp-inc",
        "education": "HS-grad",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_malformed(client):
    r = client.post("/", json={
        "age": 0,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "W",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 0
    })
    assert r.status_code == 422
