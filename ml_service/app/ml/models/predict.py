import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from .train_model import WeatherModel
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherPredictor:
    def __init__(self, model_dir: str = "models"):
        """
        Inicjalizuje predyktor pogodowy.
        """
        self.model = WeatherModel()
        self.model.load(model_dir)
    
    def prepare_input_features(self, 
                             latitude: float,
                             longitude: float,
                             date: datetime,
                             current_weather: Dict) -> np.ndarray:
        """
        Przygotowuje cechy wejściowe do predykcji.
        """
        features = np.array([[
            latitude,
            longitude,
            date.dayofyear,
            date.month,
            date.year,
            current_weather['max_temperature'],
            current_weather['min_temperature'],
            current_weather['precipitation_probability'],
            current_weather['max_windspeed'],
            current_weather['humidity'],
            current_weather['pressure']
        ]])
        
        # Skaluj cechy
        return self.model.scaler.transform(features)
    
    def predict_next_day(self,
                        latitude: float,
                        longitude: float,
                        current_weather: Dict) -> Dict:
        """
        Wykonuje predykcję pogody na następny dzień.
        """
        try:
            # Przygotuj datę na następny dzień
            next_day = datetime.now() + timedelta(days=1)
            
            # Przygotuj cechy
            features = self.prepare_input_features(
                latitude=latitude,
                longitude=longitude,
                date=next_day,
                current_weather=current_weather
            )
            
            # Wykonaj predykcje
            temp_pred = self.model.temperature_model.predict(features)[0]
            precip_pred = self.model.precipitation_model.predict(features)[0]
            
            # Przygotuj wynik
            prediction = {
                'date': next_day.strftime('%Y-%m-%d'),
                'predicted_max_temperature': round(temp_pred, 2),
                'predicted_precipitation_probability': round(precip_pred, 2),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                }
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania predykcji: {str(e)}")
            raise
    
    def predict_next_week(self,
                         latitude: float,
                         longitude: float,
                         current_weather: Dict) -> List[Dict]:
        """
        Wykonuje predykcję pogody na następny tydzień.
        """
        try:
            predictions = []
            current_date = datetime.now()
            
            for day in range(1, 8):
                next_day = current_date + timedelta(days=day)
                
                # Przygotuj cechy
                features = self.prepare_input_features(
                    latitude=latitude,
                    longitude=longitude,
                    date=next_day,
                    current_weather=current_weather
                )
                
                # Wykonaj predykcje
                temp_pred = self.model.temperature_model.predict(features)[0]
                precip_pred = self.model.precipitation_model.predict(features)[0]
                
                # Dodaj predykcję do listy
                predictions.append({
                    'date': next_day.strftime('%Y-%m-%d'),
                    'predicted_max_temperature': round(temp_pred, 2),
                    'predicted_precipitation_probability': round(precip_pred, 2),
                    'location': {
                        'latitude': latitude,
                        'longitude': longitude
                    }
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania predykcji tygodniowej: {str(e)}")
            raise

def get_weather_prediction(city: str, lat: float, lon: float, target_date: datetime) -> dict:
    """
    Przewiduje pogodę dla danego miasta na określoną datę.
    """
    try:
        # Sprawdź czy modele istnieją
        max_temp_model_path = os.path.join("models", "max_temp_model.joblib")
        min_temp_model_path = os.path.join("models", "min_temp_model.joblib")
        scaler_path = os.path.join("models", "scaler.joblib")
        
        if not os.path.exists(max_temp_model_path) or not os.path.exists(min_temp_model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Modele nie zostały jeszcze wytrenowane. Użyj endpointu /retrain aby wytrenować modele.")
            
        # Załaduj modele i scaler
        max_temp_model = joblib.load(max_temp_model_path)
        min_temp_model = joblib.load(min_temp_model_path)
        scaler = joblib.load(scaler_path)
        
        # Przygotuj dane wejściowe
        input_features = np.array([
            lat,                    # latitude
            lon,                    # longitude
            target_date.timetuple().tm_yday,  # day_of_year
            target_date.month,      # month
            target_date.year,       # year
            20.0,                   # max_temperature (przykładowa wartość)
            15.0,                   # min_temperature (przykładowa wartość)
            15.0,                   # max_windspeed (przykładowa wartość)
            65.0,                   # humidity (przykładowa wartość)
            1013.0                  # pressure (przykładowa wartość)
        ]).reshape(1, -1)
        
        # Skaluj dane
        scaled_features = scaler.transform(input_features)
        
        # Wykonaj predykcje
        max_temp_pred = max_temp_model.predict(scaled_features)[0]
        min_temp_pred = min_temp_model.predict(scaled_features)[0]
        
        return {
            "city": city,
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_max_temperature": float(max_temp_pred),
            "predicted_min_temperature": float(min_temp_pred)
        }
        
    except Exception as e:
        logger.error(f"Błąd podczas predykcji: {str(e)}")
        raise

if __name__ == "__main__":
    # Przykład użycia
    current_weather = {
        'max_temperature': 20.0,
        'min_temperature': 15.0,
        'precipitation_probability': 30.0,
        'max_windspeed': 15.0,
        'humidity': 65.0,
        'pressure': 1013.0
    }
    
    # Przykładowe współrzędne dla Warszawy
    result = get_weather_prediction(
        city="Warsaw",
        lat=52.2297,
        lon=21.0122,
        target_date=datetime.now()
    )
    
    print(result) 