"""
Katastrophenschutz – EinsatzState
===================================
Gemeinsamer LangGraph-State aller Agenten.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EinsatzState(TypedDict):
    messages:               Annotated[list[BaseMessage], add_messages]
    einsatz_id:             str
    standort:               str
    beschreibung:           str      # Freitext-Lagebeurteilung / Einsatzmeldung
    schweregrad:            str      # UNBEKANNT | GRUEN | GELB | ROT | SCHWARZ
    triage_abgeschlossen:   bool
    ressourcen_reserviert:  bool
    transport_koordiniert:  bool
    bericht_erstellt:       bool
    naechster_agent:        str      # Routing-Entscheidung (Legacy-Feld)
    runden:                 int      # Iterationszähler (Legacy-Feld)
    fallbacks:              Annotated[list[str], operator.add]  # aus parallelen Ästen zusammengeführt
    patienten:              list[dict]  # Triage-Ergebnisse aller bewerteten Patienten
