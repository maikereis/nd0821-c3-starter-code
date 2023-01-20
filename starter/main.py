import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

CURRENT_DIR = Path(__file__).parent
ARTIFACTS_DIR = CURRENT_DIR / "model"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Data(BaseModel):
    age: int = 45
    workclass: str = "State-gov"
    fnlgt: int = 2334
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Prof-specialty"
    relationship: str = "Wife"
    race: str = "Black"
    sex: str = "Female"
    capital_gain: int = 2174
    capital_loss: int = 0
    hours_per_week: int = 60
    native_country: str = "Cuba"


mapping = {
    "education_num": "education-num",
    "marital_status": "marital-status",
    "capital_gain": "capital-gain",
    "capital_loss": "capital-loss",
    "hours_per_week": "hours-per-week",
    "native_country": "native-country",
}


def get_artifact(model_name):
    model_filepath = ARTIFACTS_DIR / model_name
    return load(model_filepath.resolve())


@app.get("/")
def root():
    """An endpoint just to check if the API is UP!

    Returns:
        Json: just a message.
    """
    return {"message": "Hello World!"}


@app.post("/inference")
def inference(data: Data):
    data_dict = data.dict()
    # replace '_' for '-' in keys names
    data_dict = {mapping.get(k, k): v for k, v in data_dict.items()}
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").T

    encoder = get_artifact("preprocess.joblib")
    model = get_artifact("model.joblib")

    X_cat = data_df[CAT_FEATURES]
    complementary_cols = set(data_df.columns) - set(CAT_FEATURES)

    X_num = data_df[list(complementary_cols)]
    X_cat = encoder.transform(X_cat.values)
    X = np.concatenate([X_num, X_cat], axis=1)

    pred = model.predict(X)[0]

    return {f"the model predicted as: {pred}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", reload=True, port=8000)
