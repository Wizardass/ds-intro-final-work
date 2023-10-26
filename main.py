from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
with open('model/sber_pipe.pkl', 'rb') as file:
    model = dill.load(file)



class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    client_id: str
    Result: float


@app.get('/status')
def status():
    return "I'm okay"


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model.predict(df)
    return {
        'client_id': form.client_id,
        'Result': y[0]
    }




