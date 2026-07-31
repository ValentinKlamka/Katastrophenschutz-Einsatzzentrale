"""Agent: Triage – Patientenbewertung nach START-Schema."""
from __future__ import annotations

import json
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from ..config import get_llm
from ..state  import EinsatzState
from ..tools  import patient_triage_bewerten, patient_id_generieren, wetter_daten_abrufen
from ..utils  import _agent_log_schreiben, _invoke_mit_fallback, _kontext, _trace_ausgeben


def triage_agent_aufbauen():
    """Triage-Agent: Bewertet Patienten, erkennt Schweregrade, initiiert Handover."""
    return create_agent(
        model=get_llm(),
        tools=[patient_id_generieren, patient_triage_bewerten, wetter_daten_abrufen],
        system_prompt=(
            "Du bist der Triage-Agent der Katastrophenschutz-Einsatzleitzentrale.\n\n"
            "AUFGABE:\n"
            "- Rufe wetter_daten_abrufen() auf, um Umgebungsbedingungen zu berücksichtigen\n"
            "  (falls keine Daten verfügbar sind, einfach ohne Wetterdaten weiterarbeiten)\n"
            "- Für JEDEN Patienten in der Meldung:\n"
            "  1. Rufe patient_id_generieren(einsatz_id=<aktuelle_einsatz_id>) auf\n"
            "  2. Rufe patient_triage_bewerten() mit der generierten ID auf\n"
            "- Schließe deine Antwort mit einem JSON-Block ab (letzter Inhalt der Nachricht):\n"
            "```json\n"
            "{\"schweregrad\": \"<SCHWARZ|ROT|GELB|GRUEN>\", "
            "\"standort\": \"<konkreter Ort: Gebäude/Halle/Anlage, nicht Straße>\"}\n"
            "```\n\n"
            "HANDOVER-PROTOKOLL:\n"
            "- Bei ROT/SCHWARZ: 'HANDOVER → Ressourcen-Agent: Kritische Patienten erfordern "
            "sofortige Ressourcenallokation'\n"
            "- Bei GELB/GRUEN: 'Übergabe an Ressourcen-Agent zur Standardversorgung'\n\n"
            "Antworte auf Deutsch. Sei medizinisch präzise und strukturiert."
        ),
    )


def triage_node(state: EinsatzState) -> dict:
    print("\n[TRIAGE-AGENT] Starte Patientenbewertung nach START-Schema...")
    agent = triage_agent_aufbauen()
    einsatzmeldung = state["messages"][0].content if state["messages"] else "Keine Meldung"
    result = _invoke_mit_fallback(
        agent,
        _kontext(state, f"Eingehende Einsatzmeldung:\n{einsatzmeldung}"),
        "TRIAGE-AGENT",
    )

    # Schweregrad + Standort aus JSON-Block der Agentenantwort lesen
    letzte_antwort_raw  = str(result["messages"][-1].content)
    standort_aus_agent  = None
    schweregrad         = state.get("schweregrad", "UNBEKANNT")
    m = re.search(r"```json\s*(.+?)\s*```", letzte_antwort_raw, re.DOTALL)
    if m:
        try:
            parsed              = json.loads(m.group(1))
            schweregrad         = parsed.get("schweregrad", schweregrad).upper()
            standort_aus_agent  = parsed.get("standort") or None
        except json.JSONDecodeError:
            pass
    if schweregrad not in {"SCHWARZ", "ROT", "GELB", "GRUEN"}:
        schweregrad = next(
            (g for g in ["SCHWARZ", "ROT", "GELB", "GRUEN"] if g in letzte_antwort_raw.upper()),
            "UNBEKANNT",
        )

    # Patientendaten aus ToolMessages extrahieren
    patienten = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "patient_triage_bewerten":
            try:
                patienten.append(
                    json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                )
            except (json.JSONDecodeError, TypeError):
                pass

    _trace_ausgeben("TRIAGE-AGENT", result["messages"])
    ist_fallback = "_fallback_agent" in result
    standort = (
        (standort_aus_agent or state.get("standort", "UNBEKANNT"))
        if not ist_fallback
        else state.get("standort", "UNBEKANNT")
    )
    if not ist_fallback:
        print(
            f"[TRIAGE-AGENT] Abgeschlossen. Standort: {standort} | "
            f"Höchster Schweregrad: {schweregrad}, {len(patienten)} Patient(en) bewertet."
        )

    _agent_log_schreiben(
        einsatz_id=state["einsatz_id"],
        agent_name="TRIAGE-AGENT",
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
        "triage_abgeschlossen": not ist_fallback,
        "schweregrad":          schweregrad,
        "standort":             standort,
        "fallbacks":            ["TRIAGE-AGENT"] if ist_fallback else [],
        "patienten":            patienten,
    }
