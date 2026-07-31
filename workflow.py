"""
workflow.py – Backward-Compatibility-Shim
==========================================
Alle Logik liegt jetzt im Paket ``katastrophenschutz/``.
Dieses Modul re-exportiert die öffentliche API, damit app.py
und bestehende Skripte unverändert funktionieren.

Neue Struktur:
  katastrophenschutz/
    config.py          ← Konstanten, get_llm, KRANKENHAEUSER, EINSATZMELDUNG
    state.py           ← EinsatzState TypedDict
    utils.py           ← _kontext, _invoke_mit_fallback, _trace_ausgeben,
                          _agent_log_schreiben, _einsatz_id_generieren
    db.py              ← einsatz_anlegen, patient_hinzufuegen, patient_aktualisieren
    graph.py           ← graph_erstellen, einsatz_koordinieren
    tools/             ← umwelt, triage, ressourcen, logistik, kommunikation
    agents/            ← triage, ressourcen, logistik, kommunikation
"""
from katastrophenschutz import (
    DATABASE_URL,
    EINSATZMELDUNG,
    KRANKENHAEUSER,
    EinsatzState,
    _einsatz_id_generieren,
    einsatz_anlegen,
    einsatz_koordinieren,
    patient_aktualisieren,
    patient_hinzufuegen,
)

__all__ = [
    "DATABASE_URL",
    "EINSATZMELDUNG",
    "KRANKENHAEUSER",
    "EinsatzState",
    "_einsatz_id_generieren",
    "einsatz_anlegen",
    "einsatz_koordinieren",
    "patient_aktualisieren",
    "patient_hinzufuegen",
]

if __name__ == "__main__":
    einsatz_koordinieren()
