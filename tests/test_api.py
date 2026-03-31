# tests/test_api.py
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# test 1 - home endpoint
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("✅ home endpoint works")

# test 2 - health endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ health endpoint works")

# test 3 - predict endpoint
def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "flower_name" in response.json()
    assert "prediction" in response.json()
    assert "probability" in response.json()
    print("✅ predict endpoint works")

# test 4 - wrong input
def test_wrong_input():
    payload = {
        "sepal_length": "wrong",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    print("✅ wrong input handled correctly")