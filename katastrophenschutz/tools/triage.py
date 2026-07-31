"""Tools: Patienten-Triage (START-Schema) + Patienten-ID-Generierung."""
from __future__ import annotations

import random
import string
from typing import Literal

import psycopg

from langchain_core.tools import tool


@tool
def patient_id_generieren(einsatz_id: str) -> str:
    """
    Generiert eine eindeutige, zufällige Patienten-ID für den Einsatz.
    Prüft gegen die Datenbank und wiederholt bei Kollision.
    Format: PAT-XXXXXX (6 alphanumerische Zeichen).
    """
    from ..config import DATABASE_URL

    def _neue_id() -> str:
        chars = string.ascii_uppercase + string.digits
        return "PAT-" + "".join(random.choices(chars, k=6))

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            for _ in range(10):
                pid = _neue_id()
                exists = conn.execute(
                    "SELECT 1 FROM patienten WHERE einsatz_id = %s AND patient_id = %s",
                    (einsatz_id, pid),
                ).fetchone()
                if not exists:
                    return pid
    except Exception:
        # DB nicht erreichbar – ID ohne Kollisionsprüfung zurückgeben
        return _neue_id()
    return _neue_id()


@tool
def patient_triage_bewerten(
    patient_id: str,
    bewusstsein: Literal["klar", "getrübt", "bewusstlos"],
    atmung: Literal["normal", "eingeschränkt", "keine"],
    puls: Literal["kräftig", "schwach", "kein"],
    hauptverletzung: str,
) -> dict:
    """
    Bewertet einen Patienten nach dem START-Triage-Schema.
    Gibt Schweregrad GRUEN / GELB / ROT / SCHWARZ zurück.
    """
    if atmung == "keine" or puls == "kein":
        kat, prio = "SCHWARZ", 0
    elif bewusstsein == "bewusstlos" or atmung == "eingeschränkt" or puls == "schwach":
        kat, prio = "ROT", 1
    elif bewusstsein == "getrübt" or any(
        w in hauptverletzung for w in ["Fraktur", "Blutung", "Trauma", "Schock"]
    ):
        kat, prio = "GELB", 2
    else:
        kat, prio = "GRUEN", 3

    erstversorgung = {
        "SCHWARZ": "Keine aktiven Maßnahmen – würdevoller Umgang",
        "ROT":     "Sofortige lebensrettende Maßnahmen: Atemweg sichern, Blutung stillen",
        "GELB":    "Schmerztherapie, stabile Lagerung, Überwachung",
        "GRUEN":   "Erste Hilfe, Registrierung, Beruhigung",
    }[kat]

    return {
        "patient_id":    patient_id,
        "schweregrad":   kat,
        "prioritaet":    prio,
        "erstversorgung": erstversorgung,
        "transport":     "sofort" if prio <= 1 else "baldmöglichst" if prio == 2 else "ambulant",
    }
