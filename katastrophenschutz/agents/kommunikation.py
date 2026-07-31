"""Agent: Kommunikation – Lageberichte & Broadcasts."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from ..config import get_llm
from ..state  import EinsatzState
from ..tools  import broadcast_senden, lagebericht_erstellen
from ..utils  import _agent_log_schreiben, _invoke_mit_fallback, _kontext, _trace_ausgeben


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
            "   - Fasse zusammen: Triage-Ergebnisse, reservierte Ressourcen, geplante Transporte\n"
            "2. Sende Broadcast an alle Einheiten mit broadcast_senden():\n"
            "   Empfänger: ['Einsatzleitung', 'Leitstelle', 'Feuerwehr', "
            "'Bereitschaftsdienst', 'Krankenhäuser']\n\n"
            "Antworte auf Deutsch. Kommuniziere klar und strukturiert."
        ),
    )


def kommunikations_node(state: EinsatzState) -> dict:
    print("\n[KOMMUNIKATIONS-AGENT] Erstelle Abschluss-Lagebericht...")
    agent  = kommunikations_agent_aufbauen()
    result = _invoke_mit_fallback(
        agent,
        _kontext(
            state,
            f"Erstelle den Abschluss-Lagebericht für Einsatz {state['einsatz_id']}.\n"
            f"Triage ✓  |  Ressourcen reserviert ✓  |  Transporte koordiniert ✓\n"
            f"Höchster Schweregrad: {state.get('schweregrad', 'ROT')}\n"
            "Sende Broadcast an alle Einheiten.",
        ),
        "KOMMUNIKATIONS-AGENT",
    )
    _trace_ausgeben("KOMMUNIKATIONS-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print("[KOMMUNIKATIONS-AGENT] Lagebericht erstellt und gesendet.")

    _agent_log_schreiben(
        einsatz_id=state["einsatz_id"],
        agent_name="KOMMUNIKATIONS-AGENT",
        status="FALLBACK" if ist_fallback else "OK",
        tool_calls=[
            {"tool": tc["name"], "args": tc.get("args", {})}
            for msg in result["messages"] if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        ],
        antwort=str(result["messages"][-1].content),
    )
    return {
        "messages":        result["messages"],
        "bericht_erstellt": not ist_fallback,
        "fallbacks":       ["KOMMUNIKATIONS-AGENT"] if ist_fallback else [],
    }
