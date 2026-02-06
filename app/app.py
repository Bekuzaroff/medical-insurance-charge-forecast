from fastapi import FastAPI
import joblib
import os

import pandas as pd
from pydantic import BaseModel

from src.column_transformer import Transformer

# Получаем абсолютный путь к родительской папке
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_path = os.path.join(parent_dir, 'best_model.joblib')


model = joblib.load(model_path)
transformer = Transformer()

class Customer(BaseModel):
     age: int
     sex : str
     bmi : float
     children : str
     smoker: str
     region: str

app = FastAPI()

@app.post("/predict")
def predict(data: Customer):
    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region
    }])
    prepared_input = transformer.fit_transform(input_df)

    prediction = model.predict(prepared_input)

    return {"data": prediction, "input": data.dict()}
    
    
     