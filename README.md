## Model Deployment with CI/CD

This repository contains a sample project demonstrating how to build a CI pipeline. The project structure is explained below (refer to `src`):

- `main.py`: This contains the api endpoints
- `test_api.py`: This has unit tests for api endpoints
- `train_model.py`: This produces serialized model and encoders
- `test_model_train.py`: This contains unit tests for `train_model.py`

Data and model artifacts are version controlled using `dvc`. `s3` is used as artifact store.

Github workflow is used for Continuous Integration

Heroku is used for Continuous Deployment

[Link to Api](https://gun-model-deployment.herokuapp.com/)

## Important docs:

1. [Slice_output.txt](./src/slice_output.txt)
2. [Model_card](./src/model_card.md)
3. [CI passing](./screenshots/continous_integration.png)
4. [Example docs](./screenshots/example.png)
5. [Continous deployment](./screenshots/continous_deployment.png)
6. [Live post](./screenshots/live_post.png)
7. [Live get](./screenshots/live_get.png)