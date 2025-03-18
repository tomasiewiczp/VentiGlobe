import pandas as pd
from datetime import datetime, timedelta
import logging
from ...services.weather_service import WeatherService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_and_save_historical_data(cities: list, years: int = 5):
    """
    Pobiera dane historyczne dla listy miast i zapisuje je do pliku CSV
    
    Args:
        cities (list): Lista nazw miast
        years (int): Liczba lat wstecznych do pobrania
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    logger.info(f"Pobieranie danych od {start_date.date()} do {end_date.date()}")
    logger.info(f"Miasta: {', '.join(cities)}")
    
    all_data = []
    
    for city in cities:
        logger.info(f"Pobieranie danych dla {city}")
        coordinates = WeatherService.get_city_coordinates(city)
        if not coordinates:
            logger.error(f"Nie znaleziono współrzędnych dla {city}")
            continue
            
        data = WeatherService.get_historical_weather(
            coordinates["latitude"],
            coordinates["longitude"],
            start_date,
            end_date
        )
        
        if data:
            for record in data:
                record["city"] = city
                record["country"] = coordinates["country"]
            all_data.extend(data)
            logger.info(f"Pobrano {len(data)} rekordów dla {city}")
        else:
            logger.error(f"Nie udało się pobrać danych dla {city}")
    
    if all_data:
        # Zapisz do pliku CSV
        df = pd.DataFrame(all_data)
        output_file = "ml/data/historical_weather.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Zapisano {len(all_data)} rekordów do {output_file}")
    else:
        logger.error("Nie pobrano żadnych danych")

if __name__ == "__main__":
    # Przykładowa lista miast
    cities = [
        "Warsaw",
        "Krakow",
        "Wroclaw",
        "Poznan",
        "Gdansk",
        "Katowice",
        "Lodz",
        "Lublin",
        "Bialystok",
        "Szczecin"
    ]
    
    fetch_and_save_historical_data(cities) 