"""Tools: Ressourcen & A2A-Krankenhaus-Verhandlung."""
from __future__ import annotations

import random
from typing import Literal

from langchain_core.tools import ToolException, tool

from ..config import KRANKENHAEUSER


@tool
def ressourcenstatus_abrufen() -> dict:
    """Gibt den aktuellen Verfügbarkeitsstatus aller Einsatzressourcen zurück."""
    try:
        return {
            "rtw":        {"verfuegbar": 3, "eingesetzt": 1},
            "nef":        {"verfuegbar": 1, "eingesetzt": 0},
            "hubschrauber": {"verfuegbar": 0, "grund": "Unwetter"},
            "sanitaeter": {"verfuegbar": 8},
        }
    except Exception as exc:
        raise ToolException(f"Ressourcenstatus nicht abrufbar: {exc}") from exc


@tool
def krankenhaus_anfragen(
    krankenhaus_id: str,
    bett_typ: Literal["normal", "intensiv"],
    benoetigt_spezialisierung: str,
) -> dict:
    """
    A2A: Sendet eine direkte Kapazitätsanfrage an einen Krankenhaus-Agenten.
    Simuliert das Agent-zu-Agent-Kommunikationsprotokoll inkl. Latenz.
    """
    kh = KRANKENHAEUSER.get(krankenhaus_id)
    if not kh:
        raise ToolException(f"Krankenhaus '{krankenhaus_id}' nicht im A2A-Netz registriert.")

    passt = benoetigt_spezialisierung.lower() in [s.lower() for s in kh["spezialisierungen"]]
    return {
        "krankenhaus":            krankenhaus_id,
        "bett_typ":               bett_typ,
        "freie_betten":           kh.get(bett_typ, 0),
        "verfuegbar":             kh.get(bett_typ, 0) > 0,
        "spezialisierung_passt":  passt,
        "spezialisierungen":      kh["spezialisierungen"],
        "aufnahmebereit_in_min":  random.randint(5, 15),
        "a2a_latenz_ms":          kh["latenz_ms"],
    }


@tool
def ressource_reservieren(
    typ: Literal["RTW", "NEF", "Sanitaeter"],
    anzahl: int,
    einsatz_id: str,
    prioritaet: Literal["KRITISCH", "HOCH", "NORMAL"],
) -> dict:
    """Reserviert Einsatzressourcen und gibt eine Bestätigungsnummer zurück."""
    try:
        return {
            "status":           "RESERVIERT",
            "ressource":        typ,
            "anzahl":           anzahl,
            "einsatz_id":       einsatz_id,
            "prioritaet":       prioritaet,
            "reservierungs_nr": f"RES-{random.randint(10000, 99999)}",
            "verfuegbar_in_min": random.randint(2, 7),
        }
    except Exception as exc:
        raise ToolException(f"Ressource konnte nicht reserviert werden: {exc}") from exc
