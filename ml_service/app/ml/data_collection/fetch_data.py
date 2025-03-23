import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import asyncio
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lista miast do pobrania danych
CITIES = [
    "Warsaw",
    "Krakow",
    "Gdansk",
    "Wroclaw"
]

async def fetch_and_save_historical_data(cities: list = CITIES, years: int = 10):
    """
    Pobiera dane historyczne dla podanych miast i zapisuje je do pliku CSV.
    """
    try:
        all_data = []
        
        for city in cities:
            logger.info(f"Rozpoczynam pobieranie danych dla miasta: {city}")
            
            # Get city coordinates
            geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            try:
                response = requests.get(geocoding_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Pobrano współrzędne dla {city}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Błąd podczas pobierania współrzędnych dla {city}: {str(e)}")
                continue
            
            if not data.get("results"):
                logger.error(f"Nie znaleziono miasta: {city}")
                continue
                
            city_data = data["results"][0]
            lat = city_data["latitude"]
            lon = city_data["longitude"]
            logger.info(f"Współrzędne {city}: lat={lat}, lon={lon}")
            
            # Calculate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            logger.info(f"Zakres dat: od {start_date.date()} do {end_date.date()}")
            
            # Get weather data
            weather_url = f"https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "windspeed_10m_max",
                    "relative_humidity_2m_mean",
                    "pressure_msl_mean"
                ],
                "timezone": "auto"
            }
            
            try:
                logger.info("Pobieram dane pogodowe...")
                response = requests.get(weather_url, params=params, timeout=30)
                response.raise_for_status()
                weather_data = response.json()
                logger.info("Pobrano dane pogodowe")
            except requests.exceptions.RequestException as e:
                logger.error(f"Błąd podczas pobierania danych pogodowych dla {city}: {str(e)}")
                continue
            
            if "daily" not in weather_data:
                logger.error(f"Nie udało się pobrać danych pogodowych dla {city}")
                continue
                
            # Process data
            for i in range(len(weather_data["daily"]["time"])):
                try:
                    daily_data = {
                        "city": city,
                        "latitude": lat,
                        "longitude": lon,
                        "date": weather_data["daily"]["time"][i],
                        "max_temperature": weather_data["daily"]["temperature_2m_max"][i],
                        "min_temperature": weather_data["daily"]["temperature_2m_min"][i],
                        "max_windspeed": weather_data["daily"]["windspeed_10m_max"][i],
                        "humidity": weather_data["daily"]["relative_humidity_2m_mean"][i],
                        "pressure": weather_data["daily"]["pressure_msl_mean"][i]
                    }
                    all_data.append(daily_data)
                except (KeyError, IndexError) as e:
                    logger.warning(f"Pominięto nieprawidłowy rekord dla {city}: {str(e)}")
                    continue
            
            # Dodaj opóźnienie między miastami
            await asyncio.sleep(2)
        
        if not all_data:
            raise Exception("Nie udało się pobrać żadnych danych")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save to CSV file
        df = pd.DataFrame(all_data)
        df.to_csv("data/historical_weather.csv", index=False)
        logger.info(f"Zapisano {len(df)} rekordów do pliku")
        
        # Basic data validation
        logger.info("\nStatystyki danych:")
        logger.info(f"Liczba rekordów: {len(df)}")
        logger.info(f"Zakres dat: od {df['date'].min()} do {df['date'].max()}")
        logger.info("\nStatystyki dla poszczególnych kolumn:")
        for col in ['max_temperature', 'min_temperature', 'max_windspeed', 'humidity', 'pressure']:
            logger.info(f"{col}:")
            logger.info(f"  Min: {df[col].min():.2f}")
            logger.info(f"  Max: {df[col].max():.2f}")
            logger.info(f"  Średnia: {df[col].mean():.2f}")
            logger.info(f"  Brakujące wartości: {df[col].isna().sum()}")
        
        return {
            "status": "success",
            "message": f"Pobrano dane dla {len(cities)} miast",
            "total_records": len(df)
        }
        
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas pobierania danych: {str(e)}")
        raise 