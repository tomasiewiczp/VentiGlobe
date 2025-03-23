import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import os
from app.ml.data_preprocessing.prepare_data import prepare_training_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherModel:
    def __init__(self):
        """
        Inicjalizuje model do przewidywania pogody.
        """
        self.max_temp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.min_temp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = None
        
    def train(self, data: dict) -> dict:
        """
        Trenuje model na przygotowanych danych.
        """
        try:
            logger.info("Rozpoczynam trenowanie modelu...")
            
            # Rozpakuj dane
            X_train = data['X_train']
            X_test = data['X_test']
            y_train_max = data['y_train_max']
            y_train_min = data['y_train_min']
            y_test_max = data['y_test_max']
            y_test_min = data['y_test_min']
            self.scaler = data['scaler']
            
            # Trenuj modele
            self.max_temp_model.fit(X_train, y_train_max)
            self.min_temp_model.fit(X_train, y_train_min)
            
            # Dokonaj predykcji na zbiorze testowym
            y_pred_max = self.max_temp_model.predict(X_test)
            y_pred_min = self.min_temp_model.predict(X_test)
            
            # Oblicz metryki dla max temp
            mse_max = mean_squared_error(y_test_max, y_pred_max)
            rmse_max = np.sqrt(mse_max)
            mae_max = mean_absolute_error(y_test_max, y_pred_max)
            r2_max = r2_score(y_test_max, y_pred_max)
            
            # Oblicz metryki dla min temp
            mse_min = mean_squared_error(y_test_min, y_pred_min)
            rmse_min = np.sqrt(mse_min)
            mae_min = mean_absolute_error(y_test_min, y_pred_min)
            r2_min = r2_score(y_test_min, y_pred_min)
            
            logger.info("\nWyniki trenowania:")
            logger.info("Temperatura maksymalna:")
            logger.info(f"RMSE: {rmse_max:.2f}°C")
            logger.info(f"MAE: {mae_max:.2f}°C")
            logger.info(f"R2 Score: {r2_max:.3f}")
            logger.info("\nTemperatura minimalna:")
            logger.info(f"RMSE: {rmse_min:.2f}°C")
            logger.info(f"MAE: {mae_min:.2f}°C")
            logger.info(f"R2 Score: {r2_min:.3f}")
            
            return {
                'max_temp': {'rmse': rmse_max, 'mae': mae_max, 'r2': r2_max},
                'min_temp': {'rmse': rmse_min, 'mae': mae_min, 'r2': r2_min}
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {str(e)}")
            raise
            
    def save(self, model_dir: str):
        """
        Zapisuje wytrenowane modele i skaler do plików.
        """
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            max_temp_path = os.path.join(model_dir, 'max_temp_model.joblib')
            min_temp_path = os.path.join(model_dir, 'min_temp_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
            joblib.dump(self.max_temp_model, max_temp_path)
            joblib.dump(self.min_temp_model, min_temp_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"\nModele zostały zapisane w:")
            logger.info(f"Max temp model: {max_temp_path}")
            logger.info(f"Min temp model: {min_temp_path}")
            logger.info(f"Skaler: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania modelu: {str(e)}")
            raise
            
    def load(self, model_dir: str):
        """
        Wczytuje wytrenowane modele i skaler z plików.
        """
        try:
            max_temp_path = os.path.join(model_dir, 'max_temp_model.joblib')
            min_temp_path = os.path.join(model_dir, 'min_temp_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
            self.max_temp_model = joblib.load(max_temp_path)
            self.min_temp_model = joblib.load(min_temp_path)
            self.scaler = joblib.load(scaler_path)
            
            logger.info(f"\nWczytano modele z:")
            logger.info(f"Max temp model: {max_temp_path}")
            logger.info(f"Min temp model: {min_temp_path}")
            logger.info(f"Skaler: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania modelu: {str(e)}")
            raise

def train_and_save_model() -> dict:
    """
    Trenuje i zapisuje model.
    """
    try:
        # Przygotuj dane
        data_path = "data/historical_weather.csv"
        data = prepare_training_data(data_path)
        
        # Trenuj model
        model = WeatherModel()
        metrics = model.train(data)
        
        # Zapisz model
        model.save("models")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Błąd podczas procesu trenowania: {str(e)}")
        raise

if __name__ == "__main__":
    metrics = train_and_save_model()
    logger.info("Model został pomyślnie wytrenowany i zapisany") 