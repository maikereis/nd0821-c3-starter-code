import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


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


@pytest.fixture
def invalid_data():
    invalid_data = {
        "age": "not_a_number",
        "workclass": "",
        "fnlgt": "not_a_number",
        "education": "",
        "education_num": "not_a_number",
        "marital_status": "",
        "occupation": "",
        "relationship": "",
        "race": "",
        "sex": "",
        "capital_gain": "not_a_number",
        "capital_loss": "not_a_number",
        "hours_per_week": "not_a_number",
        "native_country": "",
    }
    return invalid_data


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


def test_inference_invalid_data(invalid_data):
    response = client.post("/inference", json=invalid_data)
    assert response.status_code == 422
    msg = json.loads(response.text)["detail"][0]["msg"]
    assert msg == "value is not a valid integer"
