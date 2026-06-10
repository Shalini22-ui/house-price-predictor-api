import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
data = {
    "size_sqft": [500, 1000, 1500, 2000, 2500],
    "bedrooms": [1, 2, 3, 3, 4],
    "price": [50, 100, 150, 200, 250]
}

df = pd.DataFrame(data)
X = df[["size_sqft", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")