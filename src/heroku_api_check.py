import requests

data = {"age": 52,
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
        }

resp = requests.post("https://gun-model-deployment.herokuapp.com/", json=data)
assert resp.status_code == 200

print("Response code: %s" % resp.status_code)
print("Response body: %s" % resp.json())
