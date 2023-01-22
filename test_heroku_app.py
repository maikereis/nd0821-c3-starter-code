import requests


def main():
    host = "https://udacity-app.herokuapp.com"
    predict_uri = f"{host}/predict"

    item = {
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

    response = requests.post(predict_uri, json=item)
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
