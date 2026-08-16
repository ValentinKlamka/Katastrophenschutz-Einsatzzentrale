"""Agent: Ressourcen – A2A-Verhandlung mit Krankenhäusern."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from ..config import get_llm
from ..state  import EinsatzState
from ..tools  import krankenhaus_anfragen, ressource_reservieren, ressourcenstatus_abrufen
from ..utils  import _agent_log_schreiben, _invoke_mit_fallback, _kontext, _trace_ausgeben


def ressourcen_agent_aufbauen():
    """Ressourcen-Agent: Verwaltet Ressourcen + A2A-Verhandlung mit allen Krankenhäusern."""
    return create_agent(
        model=get_llm(),
        tools=[ressourcenstatus_abrufen, krankenhaus_anfragen, ressource_reservieren],
        system_prompt=(
            "Du bist der Ressourcen-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE:\n"
            "1. Rufe ressourcenstatus_abrufen() auf\n"
            "2. Führe A2A-VERHANDLUNG mit ALLEN fünf Krankenhäusern durch:\n"
            "   - Anfrage an Klinikum_Stadtmitte (krankenhaus_anfragen)\n"
            "   - Anfrage an St_Marien_Krankenhaus (krankenhaus_anfragen)\n"
            "   - Anfrage an Stadtspital_Nord (krankenhaus_anfragen)\n"
            "   - Anfrage an Kreiskrankenhaus_Sued (krankenhaus_anfragen)\n"
            "   - Anfrage an Uniklinik_Zentrum (krankenhaus_anfragen)\n"
            "3. Wähle das optimale Krankenhaus – TRIAGE-REGELN BEACHTEN:\n"
            "   - SCHWARZ: KEIN Transport, KEINE Ressourcenzuweisung (verstorben/aussichtslos)\n"
            "   - ROT → Klinik mit Traumatologie/Verbrennungsmedizin + freiem Intensivbett\n"
            "   - GELB/GRUEN → Klinik mit freiem Normalbett\n"
            "4. Reserviere Fahrzeuge mit ressource_reservieren() nur für ROT/GELB/GRUEN-Patienten\n\n"
            "DOKUMENTATION:\n"
            "Dokumentiere die A2A-Verhandlung transparent:\n"
            "'A2A-Angebote: [KH-A: X Betten, Spez. Y] | [KH-B: ...] | ...'\n"
            "'Entscheidung: [KH-X] für Patient, weil [Begründung]'\n\n"
            "Antworte auf Deutsch."
        ),
    )


def ressourcen_node(state: EinsatzState) -> dict:
    print("\n[RESSOURCEN-AGENT] Starte A2A-Verhandlung mit Krankenhaus-Agenten...")
    agent  = ressourcen_agent_aufbauen()
    result = _invoke_mit_fallback(
        agent,
        _kontext(
            state,
            f"Schweregrad: {state.get('schweregrad', 'ROT')}\n"
            "Koordiniere Ressourcen. Frage ALLE fünf Krankenhäuser per A2A an "
            "(Klinikum_Stadtmitte, St_Marien_Krankenhaus, Stadtspital_Nord, "
            "Kreiskrankenhaus_Sued, Uniklinik_Zentrum) "
            "und wähle das optimale für ROT/SCHWARZ- und GELB/GRUEN-Patienten.",
        ),
        "RESSOURCEN-AGENT",
    )
    _trace_ausgeben("RESSOURCEN-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    if not ist_fallback:
        print("[RESSOURCEN-AGENT] A2A-Verhandlung abgeschlossen. Ressourcen reserviert.")

    _agent_log_schreiben(
        einsatz_id=state["einsatz_id"],
        agent_name="RESSOURCEN-AGENT",
        status="FALLBACK" if ist_fallback else "OK",
        tool_calls=[
            {"tool": tc["name"], "args": tc.get("args", {})}
            for msg in result["messages"] if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        ],
        antwort=str(result["messages"][-1].content),
    )
    return {
        "messages":             result["messages"],
        "ressourcen_reserviert": not ist_fallback,
        "fallbacks":            ["RESSOURCEN-AGENT"] if ist_fallback else [],
    }
