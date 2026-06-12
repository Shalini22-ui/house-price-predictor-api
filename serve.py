from fastapi import FastAPI
import joblib
import numpy as np
app = FastAPI()
model = joblib.load("model.pkl")
@app.get('/')
def home():
    return {"message": "Welcome to the House Price Prediction API!"}
@app.post('/predict')
@app.post("/predict")
def predict(data: dict):

    size = data["size_sqft"]
    bedrooms = data["bedrooms"]

    features = np.array([[size, bedrooms]])

    prediction = model.predict(features)

    return {
        "predicted_price": prediction[0]
    }