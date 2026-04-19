from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import numpy as np

app = FastAPI()

class model_input(BaseModel):

    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.post('/predict')
def diabetes_pred(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    input_list = list(input_dict.values())
    input_list = np.asarray(input_list)
    input_list = input_list.reshape(1, -1)
    input_list = scaler.transform(input_list)

    prediction = diabetes_model.predict(input_list)

    if prediction[0] == 0:
        return 'The person is not Diabetic'
    else:
        return 'The person is Diabetic'
    
    