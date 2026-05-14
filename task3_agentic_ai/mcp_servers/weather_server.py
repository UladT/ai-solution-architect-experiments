"""
Open-Meteo Weather MCP Server
Provides current weather and forecast tools with no API key required.

Run standalone:  python weather_server.py
MCP transport:   stdio (default)
"""

import asyncio
import json
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("open-meteo-weather")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather interpretation codes
WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _geocode(location: str) -> dict:
    """Resolve a city/location name to latitude and longitude."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "format": "json"}
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results")
    if not results:
        raise ValueError(f"Location '{location}' not found. Try a different city name.")

    r = results[0]
    return {
        "lat": r["latitude"],
        "lon": r["longitude"],
        "name": r["name"],
        "country": r.get("country", ""),
        "timezone": r.get("timezone", "auto"),
    }


@mcp.tool()
async def get_current_weather(location: str) -> str:
    """
    Get current weather conditions for any location worldwide using Open-Meteo.
    No API key required.

    Args:
        location: City name or location (e.g. 'London', 'New York', 'Tokyo')

    Returns:
        JSON string with temperature, humidity, wind speed, precipitation, and condition.
    """
    geo = await _geocode(location)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": geo["lat"],
                "longitude": geo["lon"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m,wind_direction_10m"
                ),
                "wind_speed_unit": "kmh",
                "timezone": geo["timezone"],
            }
        )
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    code = current.get("weather_code", 0)

    return json.dumps({
        "location": f"{geo['name']}, {geo['country']}",
        "temperature_celsius": current.get("temperature_2m"),
        "feels_like_celsius": current.get("apparent_temperature"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction_degrees": current.get("wind_direction_10m"),
        "condition": WEATHER_CODES.get(code, "Unknown"),
        "weather_code": code,
    }, indent=2)


@mcp.tool()
async def get_weather_forecast(location: str, days: int = 3) -> str:
    """
    Get a daily weather forecast for the next 1–7 days for any location.
    No API key required.

    Args:
        location: City name or location (e.g. 'Berlin', 'Sydney')
        days: Number of forecast days (1–7, default 3)

    Returns:
        JSON string with daily max/min temperature, precipitation, and condition.
    """
    days = max(1, min(7, int(days)))
    geo = await _geocode(location)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": geo["lat"],
                "longitude": geo["lon"],
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,weather_code,wind_speed_10m_max"
                ),
                "timezone": geo["timezone"],
                "forecast_days": days,
            }
        )
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    codes = daily.get("weather_code", [])
    wind = daily.get("wind_speed_10m_max", [])

    forecast = []
    for i, date in enumerate(dates):
        forecast.append({
            "date": date,
            "max_temp_celsius": max_temps[i] if i < len(max_temps) else None,
            "min_temp_celsius": min_temps[i] if i < len(min_temps) else None,
            "precipitation_mm": precip[i] if i < len(precip) else None,
            "max_wind_speed_kmh": wind[i] if i < len(wind) else None,
            "condition": WEATHER_CODES.get(
                codes[i] if i < len(codes) else 0, "Unknown"
            ),
        })

    return json.dumps({
        "location": f"{geo['name']}, {geo['country']}",
        "forecast_days": days,
        "forecast": forecast,
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
