
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_predict_valid_data():
    response = client.post("/api/insurance-price", json={
        "age": 30,
        "sex": "male",
        "bmi": 25.5,
        "children": 2,
        "smoker": "no"
    })

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


    def test_predict_invalid_age():
        response = client.post("/api/insurance-price", json={
        "age": "thirty",
        "sex": "male",
        "bmi": 25.5,
        "children": 2,
        "smoker": "no"
    })
        assert response.status_code == 422
    
    def test_predict_missing_field():
        response = client.post("/api/insurance-price", json={
        "age": 30,
        "sex": "male",
        "bmi": 25.5,
        "children": 2,
        "smoker": "no"
    })
        assert response.status_code == 422

    
    def test_predict_negative_bmi():
        response = client.post("/api/insurance-price", json={
        "age": 30,
        "sex": "male",
        "bmi": -25.5,
        "children": 2,
        "smoker": "no"
    })
        assert response.status_code == 422
    
    def test_predict_woman_smoker():
        response = client.post("/api/insurance-price", json={
        "age": 30,
        "sex": "female",
        "bmi": 25.5,
        "children": 2,
        "smoker": "yes"
    })
        assert response.status_code == 200
        assert "prediction" in response.json()
    
