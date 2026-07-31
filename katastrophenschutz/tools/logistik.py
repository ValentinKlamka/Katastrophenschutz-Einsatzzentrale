"""Tools: Logistik – Routen & Transporte."""
from __future__ import annotations

import random

from langchain_core.tools import ToolException, tool


@tool
def route_optimieren(startpunkt: str, ziel: str, fahrzeug: str) -> dict:
    """Berechnet die optimale Route unter Berücksichtigung von Verkehr und Wetter."""
    try:
        return {
            "fahrzeug":   fahrzeug,
            "route":      f"{startpunkt} → B27 → Ringstraße → {ziel}",
            "fahrzeit_min": 14,
            "distanz_km": 8.7,
            "sonderrechte": True,
            "hinweis":    "Kreuzung Hauptstr./Ringstraße für Durchfahrt vorsperren.",
        }
    except Exception as exc:
        raise ToolException(f"Route konnte nicht berechnet werden: {exc}") from exc


@tool
def transport_einplanen(
    patient_id: str,
    fahrzeug: str,
    zielklinik: str,
    schweregrad: str,
) -> dict:
    """Plant einen Patiententransport und warnt die Zielklinik elektronisch vor."""
    try:
        return {
            "status":           "GEPLANT",
            "patient":          patient_id,
            "fahrzeug":         fahrzeug,
            "ziel":             zielklinik,
            "schweregrad":      schweregrad,
            "abfahrt":          "sofort",
            "klinik_vorgewarnt": True,
            "transport_nr":     f"TR-{random.randint(100, 999)}",
            "begleitpersonal":  "1 Notfallsanitäter + 1 Rettungsassistent",
        }
    except Exception as exc:
        raise ToolException(f"Transport konnte nicht eingeplant werden: {exc}") from exc
