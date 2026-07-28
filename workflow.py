"""
Katastrophenschutz-Einsatzleitzentrale
Agent2Agent (A2A) Workflow  ·  LangChain + LangGraph + Gemini
=============================================================
Agenten:
  · Einsatzleitung      – Supervisor / Orchestrator
  · Triage-Agent        – Patientenbewertung (START-Schema)
  · Ressourcen-Agent    – Ressourcenplanung + A2A-Krankenhaus-Verhandlung
  · Logistik-Agent      – Transport- und Routenoptimierung
  · Kommunikations-Agent – Lageberichte und Broadcasts

A2A-Muster:
  · Handover      – Schweregrad-gesteuerte Agenten-Weitergabe
  · Negotiation   – Ressourcen-Agent verhandelt direkt mit Krankenhaus-Agenten
  · Orchestration – Einsatzleitung koordiniert den Gesamtablauf
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import ToolException, tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  KONFIGURATION
# ═══════════════════════════════════════════════════════════════

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_LIGHT_MODEL = os.getenv("GEMINI_LIGHT_MODEL", "gemini-2.0-flash-lite")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/katastrophenschutz")
MAX_RUNDEN = 10  # Schleifenschutz


def get_llm(temperature: float = 0.1, model: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY nicht gesetzt. Bitte .env-Datei anlegen oder Variable setzen."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )


# ═══════════════════════════════════════════════════════════════
#  EINSATZ-STATE  (gemeinsamer Zustand aller Agenten)
# ═══════════════════════════════════════════════════════════════

class EinsatzState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    einsatz_id: str
    standort: str
    schweregrad: str            # UNBEKANNT | GRUEN | GELB | ROT | SCHWARZ
    triage_abgeschlossen: bool
    ressourcen_reserviert: bool
    transport_koordiniert: bool
    bericht_erstellt: bool
    naechster_agent: str        # Routing-Entscheidung des Supervisors
    runden: int                 # Iterationszähler
    fallbacks: list[str]        # Agenten, bei denen ein Fallback ausgelöst wurde
    patienten: list[dict]       # Triage-Ergebnisse aller bewerteten Patienten


# ═══════════════════════════════════════════════════════════════
#  TOOLS  –  Externe Dienste (MCP-Simulation)
# ═══════════════════════════════════════════════════════════════

# ── Umwelt (MCP) ─────────────────────────────────────────────────────────────

@tool
def wetter_daten_abrufen(standort: str) -> dict:
    """Ruft Echtzeit-Wetterdaten für den Einsatzort ab (MCP-Simulation)."""
    try:
        return {
            "standort": standort,
            "temperatur_celsius": 7,
            "wind_kmh": 42,
            "niederschlag": "Starkregen",
            "sichtweite_meter": 500,
            "hubschrauber_einsatz_moeglich": False,
            "warnung": "Unwetterwarnung Stufe 2 – Seitenwind auf Brücken",
        }
    except Exception as exc:
        raise ToolException(f"Wetterdaten nicht abrufbar: {exc}") from exc


@tool
def verkehrslage_abrufen(von: str, nach: str) -> dict:
    """Ruft Echtzeit-Verkehrslage für eine Route ab (MCP-Simulation)."""
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


# ── Triage ────────────────────────────────────────────────────────────────────

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
        "ROT": "Sofortige lebensrettende Maßnahmen: Atemweg sichern, Blutung stillen",
        "GELB": "Schmerztherapie, stabile Lagerung, Überwachung",
        "GRUEN": "Erste Hilfe, Registrierung, Beruhigung",
    }[kat]

    return {
        "patient_id": patient_id,
        "schweregrad": kat,
        "prioritaet": prio,
        "erstversorgung": erstversorgung,
        "transport": "sofort" if prio <= 1 else "baldmöglichst" if prio == 2 else "ambulant",
    }


# ── Ressourcen + A2A-Krankenhaus-Verhandlung ─────────────────────────────────

@tool
def ressourcenstatus_abrufen() -> dict:
    """Gibt den aktuellen Verfügbarkeitsstatus aller Einsatzressourcen zurück."""
    try:
        return {
            "rtw": {"verfuegbar": 3, "eingesetzt": 1},
            "nef": {"verfuegbar": 1, "eingesetzt": 0},
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
    db: dict[str, dict] = {
        "Stadtspital_Nord": {
            "normal": 5, "intensiv": 2,
            "spezialisierungen": ["Kardiologie", "Neurologie"],
            "latenz_ms": 120,
        },
        "Kreiskrankenhaus_Sued": {
            "normal": 12, "intensiv": 0,
            "spezialisierungen": ["Allgemeinchirurgie", "Innere Medizin"],
            "latenz_ms": 85,
        },
        "Uniklinik_Zentrum": {
            "normal": 2, "intensiv": 5,
            "spezialisierungen": ["Traumatologie", "Neurochirurgie", "Verbrennungsmedizin"],
            "latenz_ms": 190,
        },
    }
    kh = db.get(krankenhaus_id)
    if not kh:
        raise ToolException(f"Krankenhaus '{krankenhaus_id}' nicht im A2A-Netz registriert.")

    passt = benoetigt_spezialisierung.lower() in [s.lower() for s in kh["spezialisierungen"]]
    return {
        "krankenhaus": krankenhaus_id,
        "bett_typ": bett_typ,
        "freie_betten": kh.get(bett_typ, 0),
        "verfuegbar": kh.get(bett_typ, 0) > 0,
        "spezialisierung_passt": passt,
        "spezialisierungen": kh["spezialisierungen"],
        "aufnahmebereit_in_min": random.randint(5, 15),
        "a2a_latenz_ms": kh["latenz_ms"],
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
            "status": "RESERVIERT",
            "ressource": typ,
            "anzahl": anzahl,
            "einsatz_id": einsatz_id,
            "prioritaet": prioritaet,
            "reservierungs_nr": f"RES-{random.randint(10000, 99999)}",
            "verfuegbar_in_min": random.randint(2, 7),
        }
    except Exception as exc:
        raise ToolException(f"Ressource konnte nicht reserviert werden: {exc}") from exc


# ── Logistik ──────────────────────────────────────────────────────────────────

@tool
def route_optimieren(startpunkt: str, ziel: str, fahrzeug: str) -> dict:
    """Berechnet die optimale Route unter Berücksichtigung von Verkehr und Wetter."""
    try:
        return {
            "fahrzeug": fahrzeug,
            "route": f"{startpunkt} → B27 → Ringstraße → {ziel}",
            "fahrzeit_min": 14,
            "distanz_km": 8.7,
            "sonderrechte": True,
            "hinweis": "Kreuzung Hauptstr./Ringstraße für Durchfahrt vorsperren.",
        }
    except Exception as exc:
        raise ToolException(f"Route konnte nicht berechnet werden: {exc}") from exc


@tool
def transport_einplanen(
    patient_id: str, fahrzeug: str, zielklinik: str, schweregrad: str
) -> dict:
    """Plant einen Patiententransport und warnt die Zielklinik elektronisch vor."""
    try:
        return {
            "status": "GEPLANT",
            "patient": patient_id,
            "fahrzeug": fahrzeug,
            "ziel": zielklinik,
            "schweregrad": schweregrad,
            "abfahrt": "sofort",
            "klinik_vorgewarnt": True,
            "transport_nr": f"TR-{random.randint(100, 999)}",
            "begleitpersonal": "1 Notfallsanitäter + 1 Rettungsassistent",
        }
    except Exception as exc:
        raise ToolException(f"Transport konnte nicht eingeplant werden: {exc}") from exc


# ── Kommunikation ─────────────────────────────────────────────────────────────

@tool
def broadcast_senden(
    nachricht: str,
    empfaenger: list[str],
    prioritaet: Literal["SOFORT", "HOCH", "NORMAL"],
) -> dict:
    """Sendet eine Broadcast-Nachricht an alle angegebenen Einheiten."""
    try:
        return {
            "gesendet": True,
            "empfaenger": empfaenger,
            "zugestellt": len(empfaenger),
            "prioritaet": prioritaet,
            "timestamp": "15.01.2024 19:52:30",
        }
    except Exception as exc:
        raise ToolException(f"Broadcast konnte nicht gesendet werden: {exc}") from exc


@tool
def lagebericht_erstellen(einsatz_id: str, phase: str, inhalt: str) -> str:
    """Erstellt einen offiziellen Lagebericht für die Einsatzleitung."""
    try:
        linie_doppelt = "═" * 55
        trennlinie = "─" * 55
        return (
            f"\n{linie_doppelt}\n"
            f"  LAGEBERICHT  │  Einsatz {einsatz_id}\n"
            f"  Phase: {phase}\n"
            f"{trennlinie}\n"
            f"{inhalt}\n"
            f"{linie_doppelt}\n"
        )
    except Exception as exc:
        raise ToolException(f"Lagebericht konnte nicht erstellt werden: {exc}") from exc


# ═══════════════════════════════════════════════════════════════
#  AGENTEN-DEFINITIONEN
# ═══════════════════════════════════════════════════════════════

def triage_agent_aufbauen():
    """Triage-Agent: Bewertet Patienten, erkennt Schweregrade, initiiert Handover."""
    return create_agent(
        model=get_llm(),
        tools=[patient_triage_bewerten, wetter_daten_abrufen],
        system_prompt=(
            "Du bist der Triage-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE:\n"
            "- Bewerte ALLE gemeldeten Patienten mit patient_triage_bewerten()\n"
            "- Rufe wetter_daten_abrufen() auf, um Umgebungsbedingungen zu berücksichtigen\n"
            "- Nenne am Ende klar den höchsten festgestellten Schweregrad\n\n"
            "HANDOVER-PROTOKOLL:\n"
            "- Bei ROT/SCHWARZ: 'HANDOVER → Ressourcen-Agent: Kritische Patienten erfordern "
            "sofortige Ressourcenallokation'\n"
            "- Bei GELB/GRUEN: 'Übergabe an Ressourcen-Agent zur Standardversorgung'\n\n"
            "Antworte auf Deutsch. Sei medizinisch präzise und strukturiert."
        ),
    )


def ressourcen_agent_aufbauen():
    """Ressourcen-Agent: Verwaltet Ressourcen + A2A-Verhandlung mit Krankenhäusern."""
    return create_agent(
        model=get_llm(),
        tools=[ressourcenstatus_abrufen, krankenhaus_anfragen, ressource_reservieren],
        system_prompt=(
            "Du bist der Ressourcen-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE:\n"
            "1. Rufe ressourcenstatus_abrufen() auf\n"
            "2. Führe A2A-VERHANDLUNG mit ALLEN drei Krankenhäusern durch:\n"
            "   - Anfrage an Stadtspital_Nord (krankenhaus_anfragen)\n"
            "   - Anfrage an Kreiskrankenhaus_Sued (krankenhaus_anfragen)\n"
            "   - Anfrage an Uniklinik_Zentrum (krankenhaus_anfragen)\n"
            "3. Wähle das optimale Krankenhaus:\n"
            "   - ROT → Klinik mit Traumatologie/Verbrennungsmedizin + freiem Intensivbett\n"
            "   - GELB → Klinik mit freiem Normalbett\n"
            "4. Reserviere Fahrzeuge mit ressource_reservieren()\n\n"
            "DOKUMENTATION:\n"
            "Dokumentiere die A2A-Verhandlung transparent:\n"
            "'A2A-Angebote: [KH-A: X Betten, Spez. Y] | [KH-B: ...] | [KH-C: ...]'\n"
            "'Entscheidung: [KH-X] für P-001, weil [Begründung]'\n\n"
            "Antworte auf Deutsch."
        ),
    )


def logistik_agent_aufbauen():
    """Logistik-Agent: Plant Transporte und optimiert Routen."""
    return create_agent(
        model=get_llm(),
        tools=[route_optimieren, transport_einplanen, verkehrslage_abrufen],
        system_prompt=(
            "Du bist der Logistik-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE:\n"
            "1. Rufe verkehrslage_abrufen() für die Hauptroute auf\n"
            "2. Berechne Routen mit route_optimieren() für jeden Transport\n"
            "3. Plane alle Transporte mit transport_einplanen() "
            "(ROT zuerst, dann GELB, dann GRUEN)\n\n"
            "PRIORITÄTEN:\n"
            "- ROT-Patienten: Maximalpriorität, sofortige Abfahrt\n"
            "- Zielkliniken immer über Ankunftszeiten informieren\n\n"
            "Antworte auf Deutsch. Sei logistisch präzise."
        ),
    )


def kommunikations_agent_aufbauen():
    """Kommunikations-Agent: Erstellt Lageberichte und koordiniert Broadcasts."""
    return create_agent(
        model=get_llm(),
        tools=[broadcast_senden, lagebericht_erstellen],
        system_prompt=(
            "Du bist der Kommunikations-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE:\n"
            "1. Erstelle einen vollständigen Abschluss-Lagebericht mit lagebericht_erstellen()\n"
            "   - Phase: 'Einsatz koordiniert / Übergabe an Kliniken'\n"
            "   - Fasse zusammen: Triage-Ergebnisse, reservierte Ressourcen, "
            "geplante Transporte\n"
            "2. Sende Broadcast an alle Einheiten mit broadcast_senden():\n"
            "   Empfänger: ['Einsatzleitung', 'Leitstelle', 'Feuerwehr', "
            "'Bereitschaftsdienst', 'Krankenhäuser']\n\n"
            "Antworte auf Deutsch. Kommuniziere klar und strukturiert."
        ),
    )


# ═══════════════════════════════════════════════════════════════
#  SUPERVISOR  –  EINSATZLEITUNG
# ═══════════════════════════════════════════════════════════════

def einsatzleitung_node(state: EinsatzState) -> dict:
    """Orchestriert den Einsatz: State-basiertes Routing mit Fortschritts-Tracking."""
    runden = state.get("runden", 0)

    # Schleifenschutz
    if runden >= MAX_RUNDEN:
        naechster = "FERTIG"
    elif not state.get("triage_abgeschlossen"):
        naechster = "triage_agent"
    elif not state.get("ressourcen_reserviert"):
        naechster = "ressourcen_agent"
    elif not state.get("transport_koordiniert"):
        naechster = "logistik_agent"
    elif not state.get("bericht_erstellt"):
        naechster = "kommunikations_agent"
    else:
        naechster = "FERTIG"

    sym = lambda key: "✓" if state.get(key) else "○"  # noqa: E731
    print(
        f"\n┌─ EINSATZLEITUNG [Runde {runden + 1}/{MAX_RUNDEN}] {'─'*18}\n"
        f"│  Triage {sym('triage_abgeschlossen')}  "
        f"Ressourcen {sym('ressourcen_reserviert')}  "
        f"Transport {sym('transport_koordiniert')}  "
        f"Bericht {sym('bericht_erstellt')}\n"
        f"│  → Aktiviere: {naechster.upper()}\n"
        f"└{'─'*45}"
    )

    return {"naechster_agent": naechster, "runden": runden + 1}


def routing_funktion(state: EinsatzState) -> str:
    """Liest die Routing-Entscheidung des Supervisors aus dem State."""
    naechster = state.get("naechster_agent", "FERTIG")
    return END if naechster == "FERTIG" else naechster


# ═══════════════════════════════════════════════════════════════
#  AGENT-NODES  (Wrapper für Subagenten-Aufrufe)
# ═══════════════════════════════════════════════════════════════

def _kontext(state: EinsatzState, aufgabe: str) -> list[BaseMessage]:
    """Erstellt eine kontextualisierte Eingabenachricht für einen Subagenten."""
    return [HumanMessage(content=(
        f"EINSATZ: {state['einsatz_id']}  |  Standort: {state['standort']}  |  "
        f"Schweregrad: {state.get('schweregrad', 'UNBEKANNT')}\n\n{aufgabe}"
    ))]


def _invoke_mit_fallback(agent, messages: list, agent_name: str) -> dict:
    """
    Ruft einen Agenten auf und fängt LLM- und API-Fehler ab.
    Im Fehlerfall wird eine AIMessage mit Fehlerbeschreibung zurückgegeben,
    damit der Workflow weiterlaufen kann statt abzustürzen.
    """
    try:
        return agent.invoke({"messages": messages})
    except Exception as exc:
        fehler_typ = type(exc).__name__
        print(f"\n[FALLBACK][{agent_name}] {fehler_typ}: {exc}")
        fallback_msg = AIMessage(
            content=(
                f"[FALLBACK] {agent_name} nicht erreichbar ({fehler_typ}). "
                "Manuelle Koordination erforderlich. "
                f"Ursache: {exc}"
            )
        )
        return {"messages": [fallback_msg], "_fallback_agent": agent_name}


def _trace_ausgeben(label: str, messages: list) -> None:
    """Gibt den vollständigen Message-Trace eines Agenten strukturiert aus."""
    import json
    from langchain_core.messages import AIMessage, ToolMessage

    print(f"\n{'─'*55}")
    print(f"  TRACE: {label}")
    print(f"{'─'*55}")
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.get("args", {}), ensure_ascii=False, indent=2)
                    print(f"  🔧 Tool-Call  → {tc['name']}")
                    for line in args_str.splitlines():
                        print(f"     {line}")
            if msg.content:
                print(f"  🤖 KI-Antwort:\n     {str(msg.content).replace(chr(10), chr(10) + '     ')}")
        elif isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError):
                result_str = str(msg.content)
            print(f"  📋 Tool-Ergebnis ({msg.name}):")
            for line in result_str.splitlines():
                print(f"     {line}")
    print(f"{'─'*55}")


def triage_node(state: EinsatzState) -> dict:
    print("\n[TRIAGE-AGENT] Starte Patientenbewertung nach START-Schema...")
    agent = triage_agent_aufbauen()
    einsatzmeldung = state["messages"][0].content if state["messages"] else "Keine Meldung"
    result = _invoke_mit_fallback(agent, _kontext(
        state,
        f"Eingehende Einsatzmeldung:\n{einsatzmeldung}",
    ), "TRIAGE-AGENT")

    # Höchsten Schweregrad aus Agentenantwort extrahieren
    letzte_antwort = str(result["messages"][-1].content).upper()
    schweregrad = next(
        (g for g in ["SCHWARZ", "ROT", "GELB", "GRUEN"] if g in letzte_antwort),
        state.get("schweregrad", "UNBEKANNT"),
    )
    # Patientendaten aus ToolMessages extrahieren
    import json
    from langchain_core.messages import ToolMessage
    patienten = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "patient_triage_bewerten":
            try:
                patienten.append(json.loads(msg.content) if isinstance(msg.content, str) else msg.content)
            except (json.JSONDecodeError, TypeError):
                pass

    _trace_ausgeben("TRIAGE-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print(f"[TRIAGE-AGENT] Abgeschlossen. Höchster Schweregrad: {schweregrad}, {len(patienten)} Patient(en) bewertet.")
    neue_fallbacks = state.get("fallbacks", []) + (["TRIAGE-AGENT"] if ist_fallback else [])
    return {
        "messages": result["messages"],
        "triage_abgeschlossen": not ist_fallback,
        "schweregrad": schweregrad,
        "fallbacks": neue_fallbacks,
        "patienten": patienten,
    }


def ressourcen_node(state: EinsatzState) -> dict:
    print("\n[RESSOURCEN-AGENT] Starte A2A-Verhandlung mit Krankenhaus-Agenten...")
    agent = ressourcen_agent_aufbauen()
    result = _invoke_mit_fallback(agent, _kontext(
        state,
        f"Schweregrad: {state.get('schweregrad', 'ROT')}\n"
        "Koordiniere Ressourcen. Frage ALLE drei Krankenhäuser per A2A an "
        "(Stadtspital_Nord, Kreiskrankenhaus_Sued, Uniklinik_Zentrum) "
        "und wähle das optimale für ROT- und GELB-Patienten.",
    ), "RESSOURCEN-AGENT")
    _trace_ausgeben("RESSOURCEN-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print("[RESSOURCEN-AGENT] A2A-Verhandlung abgeschlossen. Ressourcen reserviert.")
    neue_fallbacks = state.get("fallbacks", []) + (["RESSOURCEN-AGENT"] if ist_fallback else [])
    return {"messages": result["messages"], "ressourcen_reserviert": not ist_fallback, "fallbacks": neue_fallbacks}


def logistik_node(state: EinsatzState) -> dict:
    print("\n[LOGISTIK-AGENT] Starte Transportplanung...")
    agent = logistik_agent_aufbauen()
    result = _invoke_mit_fallback(agent, _kontext(
        state,
        f"Plane Patiententransporte von '{state['standort']}' zu den zugewiesenen Kliniken.\n"
        f"Schweregrad: {state.get('schweregrad', 'ROT')} – ROT-Patienten haben Priorität.",
    ), "LOGISTIK-AGENT")
    _trace_ausgeben("LOGISTIK-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print("[LOGISTIK-AGENT] Transportplanung abgeschlossen.")
    neue_fallbacks = state.get("fallbacks", []) + (["LOGISTIK-AGENT"] if ist_fallback else [])
    return {"messages": result["messages"], "transport_koordiniert": not ist_fallback, "fallbacks": neue_fallbacks}


def kommunikations_node(state: EinsatzState) -> dict:
    print("\n[KOMMUNIKATIONS-AGENT] Erstelle Abschluss-Lagebericht...")
    agent = kommunikations_agent_aufbauen()
    result = _invoke_mit_fallback(agent, _kontext(
        state,
        f"Erstelle den Abschluss-Lagebericht für Einsatz {state['einsatz_id']}.\n"
        f"Triage \u2713  |  Ressourcen reserviert \u2713  |  Transporte koordiniert \u2713\n"
        f"Höchster Schweregrad: {state.get('schweregrad', 'ROT')}\n"
        "Sende Broadcast an alle Einheiten.",
    ), "KOMMUNIKATIONS-AGENT")
    _trace_ausgeben("KOMMUNIKATIONS-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print("[KOMMUNIKATIONS-AGENT] Lagebericht erstellt und gesendet.")
    neue_fallbacks = state.get("fallbacks", []) + (["KOMMUNIKATIONS-AGENT"] if ist_fallback else [])
    return {"messages": result["messages"], "bericht_erstellt": not ist_fallback, "fallbacks": neue_fallbacks}


# ═══════════════════════════════════════════════════════════════
#  GRAPH-AUFBAU
# ═══════════════════════════════════════════════════════════════

AGENTEN_KNOTEN = [
    "triage_agent",
    "ressourcen_agent",
    "logistik_agent",
    "kommunikations_agent",
]


def graph_erstellen(checkpointer=None):
    """Baut den vollständigen A2A-Workflow-Graph auf und kompiliert ihn."""
    g = StateGraph(EinsatzState)

    # Nodes
    g.add_node("einsatzleitung", einsatzleitung_node)
    g.add_node("triage_agent", triage_node)
    g.add_node("ressourcen_agent", ressourcen_node)
    g.add_node("logistik_agent", logistik_node)
    g.add_node("kommunikations_agent", kommunikations_node)

    # Einstieg
    g.add_edge(START, "einsatzleitung")

    # Supervisor-Routing (bedingte Kanten)
    g.add_conditional_edges(
        "einsatzleitung",
        routing_funktion,
        {
            "triage_agent": "triage_agent",
            "ressourcen_agent": "ressourcen_agent",
            "logistik_agent": "logistik_agent",
            "kommunikations_agent": "kommunikations_agent",
            END: END,
        },
    )

    # Alle Agenten geben Kontrolle an Einsatzleitung zurück (A2A-Loop)
    for knoten in AGENTEN_KNOTEN:
        g.add_edge(knoten, "einsatzleitung")

    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════
#  DEMO-SZENARIO
# ═══════════════════════════════════════════════════════════════

EINSATZMELDUNG = """
NOTRUF 19:47 Uhr │ Industriepark Nord, Halle 3 (Chemiewerk Bauer GmbH)

Explosion + Brand in Produktionshalle. Feuerwehr im Anmarsch.
Gebäude teilweise eingestürzt. 4 Verletzte gemeldet.
Chemikalien vor Ort – Schutzausrüstung Stufe B erforderlich.

PATIENTEN:
• P-001 – Männlich ~40 J.: bewusstlos, eingeschränkte Atmung, schwacher Puls,
          schwere Brandwunden Oberkörper + Arme (ca. 35% KOF)
• P-002 – Weiblich ~35 J.: getrübtes Bewusstsein, offene Fraktur + starke Blutung
          rechter Oberschenkel, Verdacht auf Schock
• P-003 – Männlich ~28 J.: klar, gehfähig, leichte Schnittwunden Hände,
          leichte Rauchgasvergiftung
• P-004 – Männlich ~55 J.: klar, Verdacht Thoraxtrauma, eingeschränkte Atmung,
          kräftiger Puls

STANDORT: Industriestraße 47, Kreuzung B27 – ca. 8 km vom Stadtzentrum
"""


# ═══════════════════════════════════════════════════════════════
#  HAUPTPROGRAMM
# ═══════════════════════════════════════════════════════════════


def _einsatz_id_generieren() -> str:
    """Generiert eine eindeutige Einsatz-ID im Format E-YYYY-MMDD-NNN."""
    now = datetime.now()
    seq = random.randint(1, 999)
    return f"E-{now.year}-{now.month:02d}{now.day:02d}-{seq:03d}"




def einsatz_koordinieren(meldung: str = EINSATZMELDUNG) -> EinsatzState:
    """Startet und koordiniert den vollständigen Notfalleinsatz über A2A-Workflow."""
    print("\n" + "═" * 65)
    print("  KATASTROPHENSCHUTZ-EINSATZLEITZENTRALE")
    print("  Agent2Agent Workflow  │  LangChain + LangGraph + Gemini")
    print("═" * 65)
    print(meldung.strip())
    print("─" * 65)

    einsatz_id = _einsatz_id_generieren()
    print(f"  Einsatz-ID:  {einsatz_id}")
    print("─" * 65)

    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        app = graph_erstellen(checkpointer=checkpointer)

        initialer_state: EinsatzState = {
            "messages": [HumanMessage(content=meldung)],
            "einsatz_id": einsatz_id,
            "standort": "UNBEKANNT",
            "schweregrad": "UNBEKANNT",
            "triage_abgeschlossen": False,
            "ressourcen_reserviert": False,
            "transport_koordiniert": False,
            "bericht_erstellt": False,
            "naechster_agent": "",
            "runden": 0,
            "fallbacks": [],
        "patienten": [],
        }
        final_state: EinsatzState = app.invoke(
            initialer_state,
            config={
                "recursion_limit": 50,
                "configurable": {"thread_id": einsatz_id},
            },
        )

    fallbacks = final_state.get("fallbacks", [])
    alle_ok = (
        final_state.get("triage_abgeschlossen")
        and final_state.get("ressourcen_reserviert")
        and final_state.get("transport_koordiniert")
        and final_state.get("bericht_erstellt")
    )
    titel = "  EINSATZ VOLLSTÄNDIG KOORDINIERT" if alle_ok else "  EINSATZ TEILWEISE KOORDINIERT – MANUELLE NACHARBEIT ERFORDERLICH"

    def sym(key: str, label_ok: str = "Abgeschlossen", label_fail: str = "Fehlgeschlagen / Fallback") -> str:
        return f"✓ {label_ok}" if final_state.get(key) else f"✗ {label_fail}"

    print("\n" + "═" * 65)
    print(titel)
    print("═" * 65)
    print(f"  Einsatz-Nr:  {final_state['einsatz_id']}")
    print(f"  Schweregrad: {final_state['schweregrad']}")
    print(f"  Triage:      {sym('triage_abgeschlossen')}")
    print(f"  Ressourcen:  {sym('ressourcen_reserviert')}")
    print(f"  Transport:   {sym('transport_koordiniert')}")
    print(f"  Lagebericht: {sym('bericht_erstellt')}")
    print(f"  Runden:      {final_state['runden']}")
    if fallbacks:
        print(f"  Fallbacks:   {', '.join(fallbacks)}")
    print("═" * 65 + "\n")

    _einsatz_speichern(final_state)

    return final_state


def _einsatz_speichern(state: EinsatzState) -> None:
    """Schreibt eine Zusammenfassung des Einsatzes sowie Patientendaten in die Datenbank."""
    import psycopg

    ddl = """
    CREATE TABLE IF NOT EXISTS einsaetze (
        einsatz_id          TEXT PRIMARY KEY,
        standort            TEXT,
        schweregrad         TEXT,
        triage_ok           BOOLEAN,
        ressourcen_ok       BOOLEAN,
        transport_ok        BOOLEAN,
        bericht_ok          BOOLEAN,
        runden              INTEGER,
        fallbacks           TEXT[],
        zeitstempel         TIMESTAMPTZ DEFAULT now()
    );
    """
    upsert = """
    INSERT INTO einsaetze
        (einsatz_id, standort, schweregrad, triage_ok, ressourcen_ok,
         transport_ok, bericht_ok, runden, fallbacks)
    VALUES
        (%(einsatz_id)s, %(standort)s, %(schweregrad)s, %(triage_ok)s,
         %(ressourcen_ok)s, %(transport_ok)s, %(bericht_ok)s,
         %(runden)s, %(fallbacks)s)
    ON CONFLICT (einsatz_id) DO UPDATE SET
        schweregrad    = EXCLUDED.schweregrad,
        triage_ok      = EXCLUDED.triage_ok,
        ressourcen_ok  = EXCLUDED.ressourcen_ok,
        transport_ok   = EXCLUDED.transport_ok,
        bericht_ok     = EXCLUDED.bericht_ok,
        runden         = EXCLUDED.runden,
        fallbacks      = EXCLUDED.fallbacks,
        zeitstempel    = now();
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(ddl)
            conn.execute(upsert, {
                "einsatz_id":   state["einsatz_id"],
                "standort":     state["standort"],
                "schweregrad":  state["schweregrad"],
                "triage_ok":    state.get("triage_abgeschlossen", False),
                "ressourcen_ok": state.get("ressourcen_reserviert", False),
                "transport_ok": state.get("transport_koordiniert", False),
                "bericht_ok":   state.get("bericht_erstellt", False),
                "runden":       state.get("runden", 0),
                "fallbacks":    state.get("fallbacks", []),
            })
                # Patienten speichern
            ddl_pat = """
            CREATE TABLE IF NOT EXISTS patienten (
                id               SERIAL PRIMARY KEY,
                einsatz_id       TEXT REFERENCES einsaetze(einsatz_id) ON DELETE CASCADE,
                patient_id       TEXT,
                schweregrad      TEXT,
                prioritaet       INTEGER,
                erstversorgung   TEXT,
                transport        TEXT,
                zeitstempel      TIMESTAMPTZ DEFAULT now(),
                UNIQUE (einsatz_id, patient_id)
            );
            """
            conn.execute(ddl_pat)
            upsert_pat = """
            INSERT INTO patienten
                (einsatz_id, patient_id, schweregrad, prioritaet, erstversorgung, transport)
            VALUES
                (%(einsatz_id)s, %(patient_id)s, %(schweregrad)s, %(prioritaet)s,
                 %(erstversorgung)s, %(transport)s)
            ON CONFLICT (einsatz_id, patient_id) DO UPDATE SET
                schweregrad    = EXCLUDED.schweregrad,
                prioritaet     = EXCLUDED.prioritaet,
                erstversorgung = EXCLUDED.erstversorgung,
                transport      = EXCLUDED.transport,
                zeitstempel    = now();
            """
            for pat in state.get("patienten", []):
                conn.execute(upsert_pat, {
                    "einsatz_id":    state["einsatz_id"],
                    "patient_id":    pat.get("patient_id"),
                    "schweregrad":   pat.get("schweregrad"),
                    "prioritaet":    pat.get("prioritaet"),
                    "erstversorgung": pat.get("erstversorgung"),
                    "transport":     pat.get("transport"),
                })
            conn.commit()
        print(f"[DB] Einsatz {state['einsatz_id']} gespeichert ({len(state.get('patienten', []))} Patient(en)).")
    except Exception as exc:
        print(f"[DB] Warnung: Einsatz konnte nicht gespeichert werden – {exc}")


if __name__ == "__main__":
    einsatz_koordinieren()
