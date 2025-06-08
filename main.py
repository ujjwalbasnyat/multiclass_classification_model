from typing import Annotated
import pickle
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

# import machine learning model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# pydantic model to validate imcoming data

class UserInput(BaseModel):
    """
    Pydantic class for the validation of user input
    """

    ram: Annotated[int, Field(..., description="RAM in MB")]
    battery_power: Annotated[int, Field(..., description="battery_power in mAh")]
    px_width: Annotated[int, Field(..., description="in px")]
    px_height: Annotated[int, Field(..., description="in px")]
    int_memory: Annotated[int, Field(..., description="in GB")]


@app.post("/predict")
def predict(data: UserInput):
    """
    function for the user input and display the prediction
    """
    input_df = pd.DataFrame(
        [
            {
                "ram": data.ram,
                "battery_power": data.battery_power,
                "px_width": data.px_width,
                "px_height": data.px_height,
                "int_memory": data.int_memory,
            }
        ]
    )

    prediction = model.predict(input_df)

    return JSONResponse(status_code=200, content={"prediction": int(prediction[0])})
