import requests

DEBUG = False


def main():
    if not DEBUG:
        uri = "https://udacity-maike-app.herokuapp.com/inference"
    else:
        uri = "http://127.0.0.1:8000/inference"

    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = requests.post(uri, json=data)
    print(f"API status code = {response.status_code}")
    print(f"API response = {response.json()[0]}")


if __name__ == "__main__":
    main()
