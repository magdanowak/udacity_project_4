import requests


url = "https://udacity-project-4.onrender.com/predict"

data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

if __name__ == "__main__":
    response = requests.post(url, json=data)
    print("status code: ", response.status_code)
    print("prediction: ", response.json)
