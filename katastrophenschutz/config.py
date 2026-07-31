"""
Katastrophenschutz – Konfiguration
===================================
Zentrale Einstellungen, Konstanten und LLM-Factory.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def _secret(key: str, default: str = "") -> str:
    """Liest einen Wert aus Streamlit-Secrets (Cloud) oder os.environ (lokal)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# ── LLM-Modelle ───────────────────────────────────────────────
GEMINI_MODEL       = os.getenv("GEMINI_MODEL",       "gemini-2.0-flash")
GEMINI_LIGHT_MODEL = os.getenv("GEMINI_LIGHT_MODEL", "gemini-2.0-flash-lite")

# ── Datenbank ─────────────────────────────────────────────────
DATABASE_URL = _secret("DATABASE_URL", "postgresql://localhost/katastrophenschutz")

# ── Krankenhäuser (A2A-Netz) ──────────────────────────────────
KRANKENHAEUSER: dict[str, dict] = {
    "Klinikum_Stadtmitte": {
        "normal": 4, "intensiv": 6,
        "spezialisierungen": ["Traumatologie", "Neurochirurgie", "Verbrennungsmedizin", "Kardiologie"],
        "latenz_ms": 160,
        "beschreibung": "Überregionales Traumazentrum",
    },
    "St_Marien_Krankenhaus": {
        "normal": 14, "intensiv": 1,
        "spezialisierungen": ["Allgemeinchirurgie", "Innere Medizin", "Orthopädie"],
        "latenz_ms": 90,
        "beschreibung": "Allgemeinkrankenhaus mit großer Normalstation",
    },
    "Stadtspital_Nord": {
        "normal": 5, "intensiv": 2,
        "spezialisierungen": ["Kardiologie", "Neurologie"],
        "latenz_ms": 120,
        "beschreibung": "Städtisches Krankenhaus, Schwerpunkt Neurologie/Kardiologie",
    },
    "Kreiskrankenhaus_Sued": {
        "normal": 12, "intensiv": 0,
        "spezialisierungen": ["Allgemeinchirurgie", "Innere Medizin"],
        "latenz_ms": 85,
        "beschreibung": "Kreiskrankenhaus mit vielen Normalbetten, kein Intensivbereich",
    },
    "Uniklinik_Zentrum": {
        "normal": 2, "intensiv": 5,
        "spezialisierungen": ["Traumatologie", "Neurochirurgie", "Verbrennungsmedizin"],
        "latenz_ms": 190,
        "beschreibung": "Universitätsklinikum, höchste Spezialisierung + Intensivkapazität",
    },
}

# ── Demo-Einsatzmeldung ───────────────────────────────────────
EINSATZMELDUNG = """
NOTRUF 19:47 Uhr │ Industriepark Nord, Halle 3 (Chemiewerk Bauer GmbH)

Explosion + Brand in Produktionshalle. Feuerwehr im Anmarsch.
Gebäude teilweise eingestürzt. 4 Verletzte gemeldet.
Chemikalien vor Ort – Schutzausrüstung Stufe B erforderlich.

PATIENTEN:
• Männlich ~40 J.: bewusstlos, eingeschränkte Atmung, schwacher Puls,
  schwere Brandwunden Oberkörper + Arme (ca. 35% KOF)
• Weiblich ~35 J.: getrübtes Bewusstsein, offene Fraktur + starke Blutung
  rechter Oberschenkel, Verdacht auf Schock
• Männlich ~28 J.: klar, gehfähig, leichte Schnittwunden Hände,
  leichte Rauchgasvergiftung
• Männlich ~55 J.: klar, Verdacht Thoraxtrauma, eingeschränkte Atmung,
  kräftiger Puls

STANDORT: Industriestraße 47, Kreuzung B27 – ca. 8 km vom Stadtzentrum
"""


def get_llm(temperature: float = 0.1, model: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    """Erstellt eine ChatGoogleGenerativeAI-Instanz mit dem konfigurierten Modell."""
    api_key = _secret("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY nicht gesetzt. Bitte .env-Datei anlegen oder Variable setzen."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )
