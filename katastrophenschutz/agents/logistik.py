"""Agent: Logistik – Routen & Transportplanung."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from ..config import get_llm
from ..state  import EinsatzState
from ..tools  import route_optimieren, transport_einplanen, verkehrslage_abrufen
from ..utils  import _agent_log_schreiben, _invoke_mit_fallback, _kontext, _trace_ausgeben


def logistik_agent_aufbauen():
    """Logistik-Agent: Plant Transporte und optimiert Routen."""
    return create_agent(
        model=get_llm(),
        tools=[route_optimieren, transport_einplanen, verkehrslage_abrufen],
        system_prompt=(
            "Du bist der Logistik-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE (läuft parallel zum Ressourcen-Agenten):\n"
            "1. Rufe verkehrslage_abrufen() für die Hauptroute auf\n"
            "2. Berechne mit route_optimieren() Routen vom Einsatzort zu den zugewiesenen Krankenhäusern\n"
            "3. Plane Transporte mit transport_einplanen() nach Schweregrad-Standardzuweisung:\n"
            "   - ROT/SCHWARZ → Klinikum_Stadtmitte oder Uniklinik_Zentrum "
            "(Traumatologie/Verbrennungsmedizin)\n"
            "   - GELB/GRUEN → St_Marien_Krankenhaus, Kreiskrankenhaus_Sued oder Stadtspital_Nord\n\n"
            "PRIORITÄTEN:\n"
            "- ROT-Patienten: Maximalpriorität, sofortige Abfahrt\n"
            "- Zielkliniken immer über Ankunftszeiten informieren\n\n"
            "Antworte auf Deutsch. Sei logistisch präzise."
        ),
    )


def logistik_node(state: EinsatzState) -> dict:
    print("\n[LOGISTIK-AGENT] Starte Transportplanung...")
    agent  = logistik_agent_aufbauen()
    result = _invoke_mit_fallback(
        agent,
        _kontext(
            state,
            f"Plane Patiententransporte von '{state['standort']}' zu den zugewiesenen Kliniken.\n"
            f"Schweregrad: {state.get('schweregrad', 'ROT')} – ROT-Patienten haben Priorität.",
        ),
        "LOGISTIK-AGENT",
    )
    _trace_ausgeben("LOGISTIK-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print("[LOGISTIK-AGENT] Transportplanung abgeschlossen.")

    _agent_log_schreiben(
        einsatz_id=state["einsatz_id"],
        agent_name="LOGISTIK-AGENT",
        status="FALLBACK" if ist_fallback else "OK",
        tool_calls=[
            {"tool": tc["name"], "args": tc.get("args", {})}
            for msg in result["messages"] if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        ],
        antwort=str(result["messages"][-1].content),
    )
    return {
        "messages":              result["messages"],
        "transport_koordiniert": not ist_fallback,
        "fallbacks":             ["LOGISTIK-AGENT"] if ist_fallback else [],
    }
