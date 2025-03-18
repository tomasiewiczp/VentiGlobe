import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WeatherService:
    BASE_URL = "https://api.open-meteo.com/v1"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1"
    
    @staticmethod
    def get_city_coordinates(city_name: str) -> Optional[Dict[str, float]]:
        """
        Get coordinates for a given city name using Open-Meteo's geocoding API
        """
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
        logger.debug(f"Fetching coordinates for city: {city_name}")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                logger.debug(f"Found coordinates: {result}")
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"],
                    "name": result["name"],
                    "country": result["country"]
                }
        logger.error(f"Failed to get coordinates for {city_name}")
        return None

    @staticmethod
    def get_weather_data(latitude: float, longitude: float, date: datetime) -> Optional[Dict]:
        """
        Get weather data for specific coordinates and date
        Uses forecast API for future dates and archive API for past dates
        """
        today = datetime.now().date()
        target_date = date.date()
        
        logger.debug(f"Getting weather data for date: {target_date}")
        logger.debug(f"Today's date: {today}")
        
        if target_date < today:
            # Use archive API for past dates
            url = f"{WeatherService.ARCHIVE_URL}/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": date.strftime("%Y-%m-%d"),
                "end_date": date.strftime("%Y-%m-%d"),
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "windspeed_10m_max"
                ],
                "timezone": "auto"
            }
            logger.debug("Using archive API")
        else:
            # Use forecast API for future dates
            url = f"{WeatherService.BASE_URL}/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "windspeed_10m_max"
                ],
                "timezone": "auto"
            }
            logger.debug("Using forecast API")
        
        logger.debug(f"Requesting URL: {url}")
        logger.debug(f"With params: {params}")
        
        response = requests.get(url, params=params)
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Response data: {data}")
            # Find the index of the requested date
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in data["daily"]["time"]]
            try:
                index = dates.index(date)
                result = {
                    "date": data["daily"]["time"][index],
                    "max_temperature": data["daily"]["temperature_2m_max"][index],
                    "min_temperature": data["daily"]["temperature_2m_min"][index],
                    "precipitation_probability": data["daily"].get("precipitation_probability_max", [None])[index],
                    "max_windspeed": data["daily"]["windspeed_10m_max"][index]
                }
                logger.debug(f"Found weather data: {result}")
                return result
            except ValueError:
                logger.error(f"Date {date} not found in response")
                return None
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing response: {e}")
                return None
        logger.error(f"Failed to get weather data. Status code: {response.status_code}")
        return None

    @staticmethod
    def get_historical_weather(latitude: float, longitude: float, start_date: datetime, end_date: datetime) -> Optional[List[Dict]]:
        """
        Get historical weather data for specific coordinates and date range
        """
        url = f"{WeatherService.ARCHIVE_URL}/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "windspeed_10m_max"
            ],
            "timezone": "auto"
        }
        
        logger.debug(f"Requesting historical weather data from: {url}")
        logger.debug(f"With params: {params}")
        
        response = requests.get(url, params=params)
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Response data: {data}")
            results = []
            for i in range(len(data["daily"]["time"])):
                results.append({
                    "date": data["daily"]["time"][i],
                    "max_temperature": data["daily"]["temperature_2m_max"][i],
                    "min_temperature": data["daily"]["temperature_2m_min"][i],
                    "precipitation_probability": data["daily"].get("precipitation_probability_max", [None])[i],
                    "max_windspeed": data["daily"]["windspeed_10m_max"][i]
                })
            return results
        logger.error(f"Failed to get historical weather data. Status code: {response.status_code}")
        return None 