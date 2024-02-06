from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import uvicorn


app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


class BankNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float
    

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/{name}')
def get_name(name: str = "default_name"):
    return {'Welcome To FAST API': f'{name}'}


@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
         'prediction': prediction
    }