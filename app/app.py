from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('best_model.joblib')
transformer = joblib.load('transformer.joblib')
selected_features = joblib.load('selected_features.joblib')

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
    
    
    prepared_input = prepared_input[selected_features]
   
    prediction = model.predict(prepared_input)
    
    return {"prediction": float(prediction[0])}