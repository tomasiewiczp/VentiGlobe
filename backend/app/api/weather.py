from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List, Dict
import httpx

router = APIRouter()

ML_SERVICE_URL = "http://ml_service:8002"  # Wewnętrzny adres w sieci dockerowej

@router.get("/forecast/{city_name}")
async def get_weather_forecast(city_name: str, date: str) -> Dict:
    """
    Get weather data for a specific city and date
    """
    try:
        # Convert date string to datetime
        forecast_date = datetime.strptime(date, "%Y-%m-%d")
        
        # Get city coordinates
        coordinates = WeatherService.get_city_coordinates(city_name)
        if not coordinates:
            raise HTTPException(status_code=404, detail=f"City {city_name} not found")
        
        # Get weather data
        weather_data = WeatherService.get_weather_data(
            coordinates["latitude"],
            coordinates["longitude"],
            forecast_date
        )
        
        if not weather_data:
            raise HTTPException(status_code=404, detail="Weather data not available for this date")
        
        return {
            "city": coordinates["name"],
            "country": coordinates["country"],
            "weather": weather_data
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

@router.get("/historical/{city_name}")
async def get_historical_weather(
    city_name: str,
    start_date: str,
    end_date: str
) -> List[Dict]:
    """
    Get historical weather data for a specific city and date range
    """
    try:
        # Convert date strings to datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get city coordinates
        coordinates = WeatherService.get_city_coordinates(city_name)
        if not coordinates:
            raise HTTPException(status_code=404, detail=f"City {city_name} not found")
        
        # Get historical weather
        historical_data = WeatherService.get_historical_weather(
            coordinates["latitude"],
            coordinates["longitude"],
            start,
            end
        )
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="Historical weather data not available for this period")
        
        return historical_data
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

@router.get("/{city}")
async def get_weather_prediction(city: str, date: str = None) -> Dict:
    """
    Pobiera predykcję pogody dla danego miasta z serwisu ML.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ML_SERVICE_URL}/predict/{city}",
                params={"date": date} if date else None
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas komunikacji z serwisem ML: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 