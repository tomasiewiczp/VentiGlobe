import pandas as pd
from datetime import datetime
import logging
from fetch_historical_data import fetch_and_save_historical_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_weather_data():
    """
    Aktualizuje dane historyczne o nowe rekordy
    """
    try:
        # Wczytaj istniejące dane
        df = pd.read_csv("ml/data/historical_weather.csv")
        logger.info(f"Wczytano {len(df)} rekordów z pliku")
        
        # Pobierz listę unikalnych miast
        cities = df["city"].unique().tolist()
        logger.info(f"Znaleziono {len(cities)} unikalnych miast")
        
        # Pobierz nowe dane
        fetch_and_save_historical_data(cities, years=1)
        
        logger.info("Aktualizacja danych zakończona")
        
    except FileNotFoundError:
        logger.error("Nie znaleziono pliku z danymi historycznymi")
        logger.info("Tworzenie nowego pliku z danymi...")
        fetch_and_save_historical_data(cities, years=5)
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas aktualizacji danych: {str(e)}")

if __name__ == "__main__":
    update_weather_data() 