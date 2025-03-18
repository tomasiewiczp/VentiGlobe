import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje cechy do szkolenia modelu
    """
    # Konwertuj datę na cechy numeryczne
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Usuń kolumny, których nie będziemy używać
    features = df.drop(['date', 'city', 'country'], axis=1)
    
    return features

def prepare_target(df: pd.DataFrame) -> pd.Series:
    """
    Przygotowuje zmienną docelową (max_temperature)
    """
    return df['max_temperature']

def train_model():
    """
    Szkoli model na zapisanych danych historycznych
    """
    try:
        # Wczytaj dane
        logger.info("Wczytywanie danych...")
        df = pd.read_csv("ml/data/historical_weather.csv")
        
        # Przygotuj dane do szkolenia
        logger.info("Przygotowywanie danych...")
        X = prepare_features(df)
        y = prepare_target(df)
        
        # Podziel dane na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Szkol model
        logger.info("Szkolenie modelu...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Oceń model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"R2: {r2:.2f}")
        
        # Zapisz model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"ml/models/weather_predictor_{timestamp}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model zapisany do {model_path}")
        
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas szkolenia modelu: {str(e)}")

if __name__ == "__main__":
    train_model() 