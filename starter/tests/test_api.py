import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

import json


@pytest.fixture
def data():
    data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    return data


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert json.loads(response.text) == {"message": "Hello World!"}


def test_root_content_type():
    response = client.get("/")
    assert response.headers["Content-Type"] == "application/json"


def test_inference(data):
    response = client.post("/inference", json=data)
    assert response.status_code == 200
    assert "the model predicted as:" in json.loads(response.text)[0]


def test_inference_content_type(data):
    response = client.post("/inference", json=data)
    assert response.headers["Content-Type"] == "application/json"
