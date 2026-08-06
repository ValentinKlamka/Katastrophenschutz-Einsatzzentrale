"""
Katastrophenschutz-Paket
=========================
Öffentliche API — alles was andere Module (app.py, CLI) brauchen.
"""
from .config import DATABASE_URL, EINSATZMELDUNG, GEMINI_MODEL, KRANKENHAEUSER, get_llm
from .state  import EinsatzState
from .utils  import _einsatz_id_generieren
from .db     import db_initialisieren, einsatz_anlegen, einsatz_speichern, patient_aktualisieren, patient_hinzufuegen
from .graph  import einsatz_koordinieren, graph_erstellen

__all__ = [
    # config
    "DATABASE_URL",
    "EINSATZMELDUNG",
    "GEMINI_MODEL",
    "KRANKENHAEUSER",
    "get_llm",
    # state
    "EinsatzState",
    # utils
    "_einsatz_id_generieren",
    # db
    "db_initialisieren",
    "einsatz_anlegen",
    "einsatz_speichern",
    "patient_aktualisieren",
    "patient_hinzufuegen",
    # graph
    "einsatz_koordinieren",
    "graph_erstellen",
]
