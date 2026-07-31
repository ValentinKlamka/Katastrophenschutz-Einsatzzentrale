"""Tools: Umweltdaten (Wetter, Verkehr)."""
from __future__ import annotations

import requests
from langchain_core.tools import ToolException, tool


@tool
def wetter_daten_abrufen(standort: str) -> dict:
    """Ruft Echtzeit-Wetterdaten für den Einsatzort ab (Open-Meteo API)."""
    _WMO = {
        0: "Klarer Himmel", 1: "Überwiegend klar", 2: "Teilweise bewölkt", 3: "Bedeckt",
        45: "Nebel", 48: "Gefrierender Nebel",
        51: "Leichter Nieselregen", 53: "Mäßiger Nieselregen", 55: "Dichter Nieselregen",
        61: "Leichter Regen", 63: "Mäßiger Regen", 65: "Starkregen",
        71: "Leichter Schneefall", 73: "Mäßiger Schneefall", 75: "Starker Schneefall",
        77: "Schneegriesel",
        80: "Leichte Regenschauer", 81: "Mäßige Regenschauer", 82: "Starke Regenschauer",
        85: "Schneeschauer", 86: "Starke Schneeschauer",
        95: "Gewitter", 96: "Gewitter mit Hagel", 99: "Gewitter mit starkem Hagel",
    }
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": standort, "count": 1, "language": "de"},
            timeout=5,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            print(f"[WETTER] Standort '{standort}' nicht gefunden – kein Wetterdaten.")
            return {"verfuegbar": False, "hinweis": f"Standort '{standort}' nicht gefunden, Wetterdaten nicht abrufbar."}
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        ort_name = results[0].get("name", standort)

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,windspeed_10m,weathercode,precipitation",
                "wind_speed_unit": "kmh",
                "timezone": "Europe/Berlin",
            },
            timeout=5,
        )
        wx.raise_for_status()
        cur = wx.json().get("current", {})

        temp   = cur.get("temperature_2m", 0)
        wind   = cur.get("windspeed_10m", 0)
        code   = cur.get("weathercode", 0)
        precip = cur.get("precipitation", 0)

        niederschlag    = _WMO.get(code, f"Wettercode {code}")
        hubschrauber_ok = wind < 50 and code not in (45, 48, 71, 73, 75, 77, 85, 86, 95, 96, 99)

        warnung = None
        if code in (95, 96, 99):
            warnung = "Gewitterwarnung – Luftfahrt eingestellt"
        elif wind >= 60:
            warnung = "Sturmwarnung – Hubschraubereinsatz gesperrt"
        elif wind >= 40:
            warnung = "Windwarnung – Hubschrauber nur eingeschränkt einsetzbar"
        elif code in (65, 82):
            warnung = "Unwetterwarnung – Starkregen, eingeschränkte Sicht"

        result = {
            "standort": ort_name,
            "koordinaten": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "temperatur_celsius": round(temp, 1),
            "wind_kmh": round(wind),
            "niederschlag": niederschlag,
            "niederschlag_mm": round(precip, 1),
            "hubschrauber_einsatz_moeglich": hubschrauber_ok,
        }
        if warnung:
            result["warnung"] = warnung
        return result

    except Exception as exc:
        print(f"[WETTER] Nicht abrufbar für '{standort}': {exc}")
        return {"verfuegbar": False, "hinweis": f"Wetterdaten nicht abrufbar: {exc}"}


@tool
def verkehrslage_abrufen(von: str, nach: str) -> dict:
    """Ruft Echtzeit-Verkehrslage für eine Route ab (Simulation)."""
    try:
        return {
            "strecke": f"{von} → {nach}",
            "normalzeit_min": 11,
            "aktuelle_fahrzeit_min": 19,
            "stau": True,
            "ursache": "Unfall A81 Richtung Stadtmitte (Km 23)",
            "umleitung": "B27 → Ringstraße → Klinikstraße",
            "umleitung_min": 14,
        }
    except Exception as exc:
        raise ToolException(f"Verkehrslage nicht abrufbar: {exc}") from exc
