from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

xgb_model = joblib.load("qa_xgboost_model.pkl")
le = joblib.load("product_type_encoder.pkl")

@app.get("/")
def root():
    return {"message": "FROVITRAX AI Model API is running!"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    product_type: str = Form(...)
):
    """
    Predict QA score for a given product type based on uploaded CSV.
    CSV must have columns: 'field1' (Temperature) and 'field2' (Humidity)
    """
    try:

        df = pd.read_csv(file.file)

        df.rename(columns={"field1": "Temperature", "field2": "Humidity"}, inplace=True)

        df = df.dropna(subset=["Temperature", "Humidity"])
        if df.empty:
            return {"error": "No valid temperature or humidity data found in CSV!"}

        temps = df["Temperature"].astype(float).values
        humidity = df["Humidity"].astype(float).values

        temp_avg = np.mean(temps)
        temp_min = np.min(temps)
        temp_max = np.max(temps)
        humidity_avg = np.mean(humidity)
        humidity_min = np.min(humidity)
        humidity_max = np.max(humidity)
        duration_hours = len(temps)

        try:
            product_encoded = le.transform([product_type])[0]
        except Exception:
            return {"error": f"Product type '{product_type}' not found in encoder"}

        X_test = pd.DataFrame([[product_encoded, temp_avg, temp_min, temp_max,
                                humidity_avg, humidity_min, humidity_max, duration_hours]],
                              columns=['product_type_encoded','temp_avg','temp_min','temp_max',
                                       'humidity_avg','humidity_min','humidity_max','duration_hours'])

        qa_score = xgb_model.predict(X_test)[0]

        return {
            "product": product_type,
            "temperature": {"avg": round(float(temp_avg), 2), "min": round(float(temp_min), 2), "max": round(float(temp_max), 2)},
            "humidity": {"avg": round(float(humidity_avg), 2), "min": round(float(humidity_min), 2), "max": round(float(humidity_max), 2)},
            "duration_hours": int(duration_hours),
            "qa_score": round(float(qa_score), 2)
        }

    except Exception as e:
        return {"error": str(e)}