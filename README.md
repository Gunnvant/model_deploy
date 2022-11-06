## Model Deployment with CI/CD

This repository contains a sample project demonstrating how to build a CI pipeline. The project structure is explained below (refer to `src`):

- `main.py`: This contains the api endpoints
- `test_api.py`: This has unit tests for api endpoints
- `train_model.py`: This produces serialized model and encoders
- `test_model_train.py`: This contains unit tests for `train_model.py`

Data and model artifacts are version controlled using `dvc`. `s3` is used as artifact store.

Github workflow is used for Continuous Integration

Heroku is used for Continuous Deployment