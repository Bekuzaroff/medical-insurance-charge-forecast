from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

model = joblib.load('model/best_model.joblib')
transformer = joblib.load('model/transformer.joblib')
model_columns = joblib.load('model/model_columns.joblib')

class Customer(BaseModel):
    age: int
    sex: str
    bmi: float
    children: float
    smoker: str

app = FastAPI(root_path='/api')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

@app.post("/insurance-price")
def predict(data: Customer):

    input_df = pd.DataFrame([data.dict()])
    
   
    prepared_input = transformer.transform(input_df)
    
   
    prepared_input_df = pd.DataFrame(
        prepared_input,
        columns=transformer.num_features + transformer.cat_features
    )
    
    
    final_input = transformer.feature_engineering(prepared_input_df)

    final_input = final_input[model_columns]
    
    
    prediction = model.predict(final_input)
    
    return {"prediction": float(prediction[0])}