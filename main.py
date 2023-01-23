import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

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


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Data(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


""" mapping = {
    "education_num": "education-num",
    "marital_status": "marital-status",
    "capital_gain": "capital-gain",
    "capital_loss": "capital-loss",
    "hours_per_week": "hours-per-week",
    "native_country": "native-country",
} """


@app.on_event("startup")
async def startup_event():
    global model, encoder, binarizer
    model = load(ARTIFACTS_DIR / "model.joblib")
    encoder = load(ARTIFACTS_DIR / "encoder.joblib")
    binarizer = load(ARTIFACTS_DIR / "binarizer.joblib")


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

    data_dict = data.dict(by_alias=True)

    data_df = pd.DataFrame.from_dict(data_dict, orient="index").T

    X_cat = data_df[CAT_FEATURES]
    complementary_cols = set(data_df.columns) - set(CAT_FEATURES)

    X_num = data_df[list(complementary_cols)]
    X_cat = encoder.transform(X_cat.values)
    X = np.concatenate([X_num, X_cat], axis=1)

    pred = model.predict(X)
    value = binarizer.inverse_transform(pred)[0]

    return {f"the model predicted as: {value.strip(' ')}"}


# if __name__ == "__main__":
#    import uvicorn

#    uvicorn.run("main:app", host="127.0.0.1", reload=True, port=8000)
