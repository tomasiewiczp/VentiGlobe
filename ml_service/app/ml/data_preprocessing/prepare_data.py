import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Wczytuje dane z pliku CSV i czyści je z wartości NaN.
    """
    try:
        logger.info(f"Wczytuję dane z pliku: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Wczytano {len(df)} wierszy")
        
        # Konwersja daty
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Zakres dat: od {df['date'].min()} do {df['date'].max()}")
        
        # Sprawdź brakujące wartości przed czyszczeniem
        missing_before = df.isna().sum()
        logger.info("\nBrakujące wartości przed czyszczeniem:")
        for col, count in missing_before.items():
            if count > 0:
                logger.info(f"{col}: {count}")
        
        # Usuń wiersze z brakującymi wartościami
        df = df.dropna()
        logger.info(f"Usunięto {len(df)} wierszy po usunięciu brakujących wartości")
        
        # Usuń duplikaty
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        logger.info(f"Usunięto {duplicates} duplikatów")
        
        # Usuń wartości odstające
        for col in ['max_temperature', 'min_temperature', 'max_windspeed', 
                   'humidity', 'pressure']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            if outliers > 0:
                logger.info(f"Usunięto {outliers} wartości odstających dla {col}")
        
        logger.info(f"\nStatystyki po czyszczeniu:")
        logger.info(f"Liczba wierszy: {len(df)}")
        logger.info(f"Zakres dat: od {df['date'].min()} do {df['date'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania danych: {str(e)}")
        raise

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Przygotowuje cechy do treningu modelu.
    """
    logger.info("\nPrzygotowywanie cech...")
    
    # Dodaj cechy czasowe
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Wybierz cechy do modelu
    features = [
        'latitude', 'longitude', 'day_of_year', 'month', 'year',
        'max_temperature', 'min_temperature', 'max_windspeed', 
        'humidity', 'pressure'
    ]
    
    X = df[features].values
    logger.info(f"Przygotowano {len(X)} próbek z {len(features)} cechami")
    
    # Przygotuj targety (temperatura na następny dzień)
    y_max = df['max_temperature'].shift(-1).values[:-1]
    y_min = df['min_temperature'].shift(-1).values[:-1]
    
    # Usuń ostatni wiersz z X, ponieważ nie mamy targetu dla następnego dnia
    X = X[:-1]
    
    # Usuń wiersze z NaN w targetach
    mask = ~np.isnan(y_max) & ~np.isnan(y_min)
    X = X[mask]
    y_max = y_max[mask]
    y_min = y_min[mask]
    
    logger.info(f"Przygotowano {len(X)} próbek do treningu")
    return X, y_max, y_min

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Skaluje cechy używając StandardScaler.
    """
    logger.info("\nSkalowanie cech...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Cechy zostały przeskalowane")
    return X_scaled, scaler

def split_data(X: np.ndarray, y_max: np.ndarray, y_min: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dzieli dane na zbiór treningowy i testowy.
    """
    logger.info("\nDzielenie danych na zbiory treningowy i testowy...")
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_max, y_test_max = y_max[:split_idx], y_max[split_idx:]
    y_train_min, y_test_min = y_min[:split_idx], y_min[split_idx:]
    logger.info(f"Zbiór treningowy: {len(X_train)} próbek")
    logger.info(f"Zbiór testowy: {len(X_test)} próbek")
    return X_train, X_test, y_train_max, y_test_max, y_train_min, y_test_min

def prepare_training_data(file_path: str) -> Dict:
    """
    Przygotowuje dane do treningu modelu.
    """
    try:
        # Wczytaj dane
        df = load_data(file_path)
        
        # Przygotuj cechy i targety
        X, y_max, y_min = prepare_features(df)
        
        # Skaluj cechy
        X_scaled, scaler = scale_features(X)
        
        # Podziel dane
        X_train, X_test, y_train_max, y_test_max, y_train_min, y_test_min = split_data(X_scaled, y_max, y_min)
        
        logger.info("\nPodsumowanie przygotowania danych:")
        logger.info(f"Liczba próbek treningowych: {len(X_train)}")
        logger.info(f"Liczba próbek testowych: {len(X_test)}")
        logger.info(f"Liczba cech: {X_train.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train_max': y_train_max,
            'y_test_max': y_test_max,
            'y_train_min': y_train_min,
            'y_test_min': y_test_min,
            'scaler': scaler
        }
        
    except Exception as e:
        logger.error(f"Błąd podczas przygotowywania danych: {str(e)}")
        raise

if __name__ == "__main__":
    data_path = "data/historical_weather.csv"
    if os.path.exists(data_path):
        prepared_data = prepare_training_data(data_path)
        logger.info("\nDane zostały pomyślnie przygotowane do treningu")
    else:
        logger.error(f"Nie znaleziono pliku z danymi: {data_path}") 