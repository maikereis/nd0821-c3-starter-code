import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def data_below_50k():
    data_below_50k = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 117037,
        "education": "4th",
        "education_num": 4,
        "marital_status": "Divorced",
        "occupation": "Transport-moving",
        "relationship": "Husband",
        "race": "White",
        "sex": "Woman",
        "capital_gain": 0,
        "capital_loss": 5000,
        "hours_per_week": 30,
        "native_country": "United-States",
    }
    return data_below_50k


@pytest.fixture
def data_above_50k():
    data_above_50k = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 193524,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 999999,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States",
    }
    return data_above_50k


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
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"


def test_inference_with_data_below_50k(data_below_50k):
    with TestClient(app) as client:
        response = client.post("/inference", json=data_below_50k)
        assert response.status_code == 200
        assert "the model predicted as: <=50K" == json.loads(response.text)[0]


def test_inference_with_data_above_50k(data_above_50k):
    with TestClient(app) as client:
        response = client.post("/inference", json=data_above_50k)
        assert response.status_code == 200
        assert "the model predicted as: >50K" == json.loads(response.text)[0]


def test_inference_content_type(data_below_50k):
    with TestClient(app) as client:
        response = client.post("/inference", json=data_below_50k)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"


def test_inference_invalid_data(invalid_data):
    with TestClient(app) as client:
        response = client.post("/inference", json=invalid_data)
        assert response.status_code == 422
        msg = json.loads(response.text)["detail"][0]["msg"]
        assert msg == "value is not a valid integer"
