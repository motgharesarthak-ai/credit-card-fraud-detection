
import requests

API_URL = "http://127.0.0.1:8000"

# Test the health endpoint
health_response = requests.get(f"{API_URL}/health", timeout=10)

assert health_response.status_code == 200

health_data = health_response.json()
assert health_data["status"] == "healthy"
assert health_data["model_loaded"] is True
assert health_data["scaler_loaded"] is True

print("PASS: Health endpoint")


# Test a legitimate transaction
legitimate_transaction = {
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
    "Amount": 11.50
}

response = requests.post(
    f"{API_URL}/predict",
    json=legitimate_transaction,
    timeout=10
)

assert response.status_code == 200

result = response.json()

assert result["prediction"] == 0
assert result["result"] == "Legitimate"
assert 0 <= result["fraud_probability"] <= 1

print("PASS: Legitimate transaction test")
print("Response:", result)

print("\nAll API tests passed!")
