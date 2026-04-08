from __future__ import annotations

import requests

from app.core.config import settings


class WeatherTool:
    current_url = "https://api.openweathermap.org/data/2.5/weather"
    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    geocode_url = "http://api.openweathermap.org/geo/1.0/direct"

    def resolve_location_candidates(self, location: str, limit: int = 5) -> list[dict]:
        params = {
            "q": location,
            "limit": limit,
            "appid": settings.openweather_api_key,
        }
        response = requests.get(self.geocode_url, params=params, timeout=20)
        if response.status_code == 401:
            raise ValueError("OpenWeather authentication failed (401). Check OPENWEATHER_API_KEY.")
        response.raise_for_status()
        return response.json() or []

    def resolve_or_nearby_location(self, location: str) -> tuple[dict | None, str | None]:
        direct_rows = self.resolve_location_candidates(location, limit=5)
        if direct_rows:
            return direct_rows[0], None

        # Fallback attempt: search by significant tokens and use first viable nearby match.
        tokens = [t for t in location.replace(",", " ").split() if len(t) >= 4]
        suggestions: list[dict] = []
        for token in tokens:
            rows = self.resolve_location_candidates(token, limit=5)
            for row in rows:
                key = (row.get("name"), row.get("state"), row.get("country"))
                if key not in {(x.get("name"), x.get("state"), x.get("country")) for x in suggestions}:
                    suggestions.append(row)
            if suggestions:
                break

        if suggestions:
            chosen = suggestions[0]
            chosen_name = ", ".join(
                [x for x in [chosen.get("name"), chosen.get("state"), chosen.get("country")] if x]
            )
            note = f"Exact match for '{location}' not found. Weather shown for nearby location: {chosen_name}."
            return chosen, note

        return None, None

    def run(self, location: str) -> dict:
        if not settings.openweather_api_key:
            raise ValueError("Weather API key missing. Set OPENWEATHER_API_KEY in .env.")

        resolved, nearby_note = self.resolve_or_nearby_location(location)
        if not resolved:
            raise ValueError(
                f"Weather location not found: {location}. Try a nearby city or add country (e.g., '{location}, IN')."
            )

        params = {
            "lat": resolved["lat"],
            "lon": resolved["lon"],
            "appid": settings.openweather_api_key,
            "units": "metric",
        }
        current = requests.get(self.current_url, params=params, timeout=20)
        if current.status_code == 401:
            raise ValueError("OpenWeather authentication failed (401). Check OPENWEATHER_API_KEY.")
        current.raise_for_status()
        forecast = requests.get(self.forecast_url, params=params, timeout=20)
        if forecast.status_code == 401:
            raise ValueError("OpenWeather authentication failed (401). Check OPENWEATHER_API_KEY.")
        forecast.raise_for_status()
        current_json = current.json()
        forecast_json = forecast.json()

        daily_preview = []
        for item in forecast_json.get("list", [])[:5]:
            daily_preview.append(
                {
                    "datetime": item.get("dt_txt"),
                    "temp_c": item.get("main", {}).get("temp"),
                    "description": (item.get("weather") or [{}])[0].get("description"),
                }
            )

        return {
            "location": current_json.get("name", location),
            "weather_location_note": nearby_note,
            "resolved_location": {
                "name": resolved.get("name"),
                "state": resolved.get("state"),
                "country": resolved.get("country"),
                "lat": resolved.get("lat"),
                "lon": resolved.get("lon"),
            },
            "current": {
                "temp_c": current_json.get("main", {}).get("temp"),
                "feels_like_c": current_json.get("main", {}).get("feels_like"),
                "humidity": current_json.get("main", {}).get("humidity"),
                "description": (current_json.get("weather") or [{}])[0].get("description"),
            },
            "forecast_preview": daily_preview,
        }
