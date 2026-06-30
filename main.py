import sys
import pickle
import pymysql
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import numpy as np
import pandas as pd
CURRENT_YEAR = 2025


class FullPipeline:

    def __init__(self):
        self.numeric_cols = []
        self.mean_std = {}
        self.fill_values = {}
        self.feature_columns = []
        self.luxury_brands = [
            'BMW', 'Audi', 'Mercedes-Benz',
            'Jaguar', 'Land', 'Volvo'
        ]

    def feature_engineering(self, df):
        df = df.copy()
        df['car_age'] = CURRENT_YEAR - df['year']
        df['km_per_year'] = df['km_driven'] / (df['car_age'] + 1)
        df['is_7_seater'] = (df['seats'] >= 7).astype(int)
        df['engine_power_ratio'] = df['engine'] / (df['max_power'] + 1)
        df['power_per_cc'] = df['max_power'] / (df['engine'] + 1)
        df['is_luxury'] = df['brand'].isin(self.luxury_brands).astype(int)
        return df

    def fit(self, df):
        df = self.feature_engineering(df)
        df = df.drop(columns=['selling_price'], errors='ignore')

        df = pd.get_dummies(df, drop_first=False) 

        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.fill_values = df.mean(numeric_only=True)
        df = df.fillna(self.fill_values)

        
        for col in self.numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                std = 1
            self.mean_std[col] = (mean, std)
            df[col] = (df[col] - mean) / std

        self.feature_columns = df.columns.tolist()
        return df

    def transform(self, df):
        df = self.feature_engineering(df)

        df = pd.get_dummies(df, drop_first=False)

        df = df.reindex(columns=self.feature_columns, fill_value=0)

        df = df.fillna(self.fill_values)

        for col in self.numeric_cols:
            if col in self.mean_std:
                mean, std = self.mean_std[col]
                df[col] = (df[col] - mean) / std

        return df

class SGDRegressor:

    def __init__(self, lr=0.01, epochs=450, l2=0.01, batch_size=32):

        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size

        self.coef_ = None
        self.intercept_ = 0


    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n, m = X.shape

        self.coef_ = np.zeros(m)

        for epoch in range(self.epochs):

            lr = self.lr / (1 + 0.01 * epoch)

            indices = np.random.permutation(n)

            X = X[indices]
            y = y[indices]

            for start in range(0, n, self.batch_size):

                end = start + self.batch_size

                X_batch = X[start:end]
                y_batch = y[start:end]

                y_pred = np.dot(X_batch, self.coef_) + self.intercept_

                error = y_pred - y_batch

                grad_w = (2 / len(X_batch)) * np.dot(X_batch.T, error) + 2 * self.l2 * self.coef_

                grad_b = 2 * np.mean(error)

                self.coef_ -= lr * grad_w
                self.intercept_ -= lr * grad_b


            if (epoch + 1) % 50 == 0:

                train_pred = np.dot(X, self.coef_) + self.intercept_

                loss = np.mean((y - train_pred) ** 2)

                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss:.4f}")


    def predict(self, X):

        X = np.asarray(X, dtype=np.float64)

        return np.dot(X, self.coef_) + self.intercept_


import __main__
__main__.FullPipeline = FullPipeline
__main__.SGDRegressor = SGDRegressor

app = FastAPI(title="Used Car Price Predictor App")

templates = Jinja2Templates(directory="templates")

with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name = "home.html"
    )


@app.get("/form", response_class=HTMLResponse)
async def serve_form(request: Request):
    """Renders the empty input form UI."""
    return templates.TemplateResponse(
        request=request,
        name = "form.html",
        context={"price": None}
    )


@app.post("/predict", response_class=HTMLResponse)
def handle_prediction(
    request: Request,
    year: int = Form(...),
    km_driven: int = Form(...),
    fuel: str = Form(...),
    seller_type: str = Form(...),
    transmission: str = Form(...),
    owner: str = Form(...),
    mileage: float = Form(...),
    engine: float = Form(...),
    max_power: float = Form(...),
    seats: float = Form(...),
    brand: str = Form(...),
    model_name: str = Form(..., alias="model") 
):


    user_input = {
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
        'brand': brand,
        'model': model_name,
        'brand_model': f"{brand}_{model_name}"
    }

    user_df = pd.DataFrame([user_input])
    
    processed = pipeline.transform(user_df)
    
    pred_log = model.predict(processed)
    
    final_price = int(np.expm1(pred_log[0]))
    formatted_price = f"{final_price:,}" # e.g., 1,250,000

    return templates.TemplateResponse(
        request=request,
        name = "form.html", 
        context={ "price": formatted_price}
    )