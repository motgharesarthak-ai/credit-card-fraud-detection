
import requests

API_URL = "http://127.0.0.1:8000"


def test_health_endpoint():
    response = requests.get(f"{API_URL}/health", timeout=10)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True


def test_fraud_transaction():
    transaction = {
        "Time": 74159.0,
        "V1": -1.5487880985,
        "V2": 1.8086979504,
        "V3": -0.9535090338,
        "V4": 2.2130853935,
        "V5": -2.0157277917,
        "V6": -0.9134568445,
        "V7": -2.3560129832,
        "V8": 1.1971689670,
        "V9": -1.6783740566,
        "V10": -3.5386502318,
        "V11": 3.1020899272,
        "V12": -3.9933730545,
        "V13": -1.9374106233,
        "V14": -3.8228941060,
        "V15": 0.8309701107,
        "V16": -2.4753588538,
        "V17": -5.2118751677,
        "V18": -0.4138716782,
        "V19": 0.9332621646,
        "V20": 0.3907859638,
        "V21": 0.8551382633,
        "V22": 0.7747448215,
        "V23": 0.0590371520,
        "V24": 0.3431998079,
        "V25": -0.4689379286,
        "V26": -0.2783379869,
        "V27": 0.6259222152,
        "V28": 0.3955733783,
        "Amount": 76.94,
    }

    response = requests.post(
        f"{API_URL}/predict", json=transaction, timeout=10
    )

    assert response.status_code == 200
    result = response.json()

    assert result["prediction"] == 1
    assert result["result"] == "Fraud"


def test_legitimate_transaction():
    transaction = {
        "Time": 61290.0,
        "V1": 1.2288211502,
        "V2": -0.0634077165,
        "V3": 0.2741451422,
        "V4": 0.6474650218,
        "V5": -0.0481345612,
        "V6": 0.3720730286,
        "V7": -0.2242305874,
        "V8": 0.0799390492,
        "V9": 0.6407588171,
        "V10": -0.2730537022,
        "V11": -1.2527279388,
        "V12": 0.4650787707,
        "V13": 0.4005021153,
        "V14": -0.2928418606,
        "V15": -0.1017740160,
        "V16": -0.3998358978,
        "V17": 0.0343356568,
        "V18": -0.7835502549,
        "V19": 0.1413449004,
        "V20": -0.0965659024,
        "V21": -0.1295544481,
        "V22": -0.0837793282,
        "V23": -0.1516614739,
        "V24": -0.7003715973,
        "V25": 0.5985501645,
        "V26": 0.4914090706,
        "V27": 0.0029892597,
        "V28": 0.0017822861,
        "Amount": 11.50,
    }

    response = requests.post(
        f"{API_URL}/predict", json=transaction, timeout=10
    )

    assert response.status_code == 200
    result = response.json()

    assert result["prediction"] == 0
    assert result["result"] == "Legitimate"


def test_negative_amount_is_rejected():
    response = requests.post(
        f"{API_URL}/predict",
        json={"Time": 10, "Amount": -5},
        timeout=10,
    )

    assert response.status_code == 422
