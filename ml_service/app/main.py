from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict
import asyncio
import os
from app.ml.data_collection.fetch_data import fetch_and_save_historical_data
from app.ml.models.train_model import train_and_save_model
from app.ml.models.predict import get_weather_prediction

app = FastAPI(title="VentiGlobe ML Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Inicjalizuje dane i model przy starcie serwisu, jeśli nie istnieją.
    """
    try:
        # Sprawdź czy istnieją dane
        if not os.path.exists("data/historical_weather.csv"):
            await fetch_and_save_historical_data()
            train_and_save_model()
    except Exception as e:
        print(f"Błąd podczas inicjalizacji: {str(e)}")

@app.get("/")
async def root():
    return {"message": "VentiGlobe ML Service is running"}

@app.get("/predict/{city}")
async def predict_weather(city: str, date: str = None) -> Dict:
    """
    Przewiduje pogodę dla danego miasta na określoną datę.
    Jeśli data nie jest podana, używa dzisiejszej daty.
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
        # Współrzędne miast
        city_coords = {
            "Warsaw": (52.22977, 21.01178),
            "Krakow": (50.06143, 19.93658),
            "Gdansk": (54.35227, 18.64912),
            "Wroclaw": (51.1, 17.03333)
        }
        
        if city not in city_coords:
            raise HTTPException(status_code=404, detail=f"Miasto {city} nie jest obsługiwane")
            
        lat, lon = city_coords[city]
        return get_weather_prediction(city, lat, lon, target_date)
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Nieprawidłowy format daty. Użyj formatu YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model() -> Dict:
    """
    Pobiera nowe dane i trenuje model od nowa.
    """
    try:
        # Pobierz nowe dane
        await fetch_and_save_historical_data()
        
        # Trenuj model
        metrics = train_and_save_model()
        
        return {
            "status": "success",
            "message": "Model został pomyślnie wytrenowany",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 