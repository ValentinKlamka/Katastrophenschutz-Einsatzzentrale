"""
Katastrophenschutz – LangGraph-Workflow
=========================================
Graph-Definition und Einstiegspunkt einsatz_koordinieren().
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    kommunikations_node,
    logistik_node,
    ressourcen_node,
    triage_node,
)
from .config import DATABASE_URL, EINSATZMELDUNG
from .db     import einsatz_speichern
from .state  import EinsatzState
from .utils  import _einsatz_id_generieren


def graph_erstellen(checkpointer=None):
    """Baut den A2A-Workflow-Graph mit paralleler Ressourcen- und Logistikplanung auf."""
    g = StateGraph(EinsatzState)

    g.add_node("triage_agent",          triage_node)
    g.add_node("ressourcen_agent",       ressourcen_node)
    g.add_node("logistik_agent",         logistik_node)
    g.add_node("kommunikations_agent",   kommunikations_node)

    # Triage zuerst, dann Fan-out
    g.add_edge(START,            "triage_agent")
    g.add_edge("triage_agent",   "ressourcen_agent")   # ┐ parallel
    g.add_edge("triage_agent",   "logistik_agent")     # ┘

    # Fan-in: Kommunikation wartet automatisch auf beide
    g.add_edge("ressourcen_agent",     "kommunikations_agent")
    g.add_edge("logistik_agent",       "kommunikations_agent")
    g.add_edge("kommunikations_agent", END)

    return g.compile(checkpointer=checkpointer)


def einsatz_koordinieren(
    meldung: str = EINSATZMELDUNG,
    einsatz_id: str | None = None,
    standort: str = "UNBEKANNT",
) -> EinsatzState:
    """Startet und koordiniert den vollständigen Notfalleinsatz über A2A-Workflow."""
    print("\n" + "═" * 65)
    print("  KATASTROPHENSCHUTZ-EINSATZLEITZENTRALE")
    print("  Agent2Agent Workflow  │  LangChain + LangGraph + Gemini")
    print("═" * 65)
    print(meldung.strip())
    print("─" * 65)

    eid = einsatz_id or _einsatz_id_generieren()
    print(f"  Einsatz-ID:  {eid}")
    print("─" * 65)

    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        app = graph_erstellen(checkpointer=checkpointer)

        initialer_state: EinsatzState = {
            "messages":             [HumanMessage(content=meldung)],
            "einsatz_id":           eid,
            "standort":             standort,
            "beschreibung":         meldung,
            "schweregrad":          "UNBEKANNT",
            "triage_abgeschlossen": False,
            "ressourcen_reserviert": False,
            "transport_koordiniert": False,
            "bericht_erstellt":     False,
            "naechster_agent":      "",
            "runden":               0,
            "fallbacks":            [],
            "patienten":            [],
        }
        final_state: EinsatzState = app.invoke(
            initialer_state,
            config={
                "recursion_limit": 50,
                "configurable":    {"thread_id": eid},
            },
        )

    fallbacks = final_state.get("fallbacks", [])
    alle_ok = (
        final_state.get("triage_abgeschlossen")
        and final_state.get("ressourcen_reserviert")
        and final_state.get("transport_koordiniert")
        and final_state.get("bericht_erstellt")
    )
    titel = (
        "  EINSATZ VOLLSTÄNDIG KOORDINIERT"
        if alle_ok
        else "  EINSATZ TEILWEISE KOORDINIERT – MANUELLE NACHARBEIT ERFORDERLICH"
    )

    def sym(key: str) -> str:
        return "✓ Abgeschlossen" if final_state.get(key) else "✗ Fallback / Fehlgeschlagen"

    print("\n" + "═" * 65)
    print(titel)
    print("═" * 65)
    print(f"  Einsatz-Nr:  {final_state['einsatz_id']}")
    print(f"  Schweregrad: {final_state['schweregrad']}")
    print(f"  Triage:      {sym('triage_abgeschlossen')}")
    print(f"  Ressourcen:  {sym('ressourcen_reserviert')}")
    print(f"  Transport:   {sym('transport_koordiniert')}")
    print(f"  Lagebericht: {sym('bericht_erstellt')}")
    if fallbacks:
        print(f"  Fallbacks:   {', '.join(fallbacks)}")
    print("═" * 65 + "\n")

    einsatz_speichern(final_state)
    return final_state
