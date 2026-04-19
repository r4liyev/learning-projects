import json 
import requests

url = 'https://joya-serous-unartificially.ngrok-free.dev/predict'

input_data_for_model = {
    "Pregnancies": 0,
    "Glucose": 129,
    "BloodPressure": 80,
    "SkinThickness": 0,
    "Insulin": 0,
    "BMI": 31.2,
    "DiabetesPedigreeFunction": 0.703,
    "Age": 29
}

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)