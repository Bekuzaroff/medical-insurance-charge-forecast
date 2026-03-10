from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Загружаем все компоненты
model = joblib.load('best_model.joblib')
transformer = joblib.load('transformer.joblib')
selected_features = joblib.load('selected_features.joblib')  # ← загружаем список признаков

class Customer(BaseModel):
    age: int
    sex: str
    bmi: float
    children: float
    smoker: str
    region: str

app = FastAPI()

@app.post("/predict")
def predict(data: Customer):

    input_df = pd.DataFrame([data.dict()])
 
    prepared_input = transformer.transform(input_df)
    
    
    prepared_input = prepared_input[selected_features]
   
    prediction = model.predict(prepared_input)
    
    return {"prediction": float(prediction[0]), "input": data.dict()}