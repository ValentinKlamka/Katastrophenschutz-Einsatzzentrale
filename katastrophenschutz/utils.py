"""
Katastrophenschutz – Hilfsfunktionen
======================================
Agenten-Utilities: Kontext, Fallback-Aufruf, Tracing, Logging.
"""
from __future__ import annotations

import json
import random
from datetime import datetime

import psycopg
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .config import DATABASE_URL
from .state import EinsatzState


# ── ID-Generierung ────────────────────────────────────────────

def _einsatz_id_generieren() -> str:
    """Generiert eine eindeutige Einsatz-ID im Format E-YYYY-MMDD-NNN."""
    now = datetime.now()
    seq = random.randint(1, 999)
    return f"E-{now.year}-{now.month:02d}{now.day:02d}-{seq:03d}"


# ── Agenten-Helpers ───────────────────────────────────────────

def _kontext(state: EinsatzState, aufgabe: str) -> list[HumanMessage]:
    """Erstellt eine kontextualisierte Eingabenachricht für einen Subagenten."""
    return [HumanMessage(content=(
        f"EINSATZ: {state['einsatz_id']}  |  Standort: {state['standort']}  |  "
        f"Schweregrad: {state.get('schweregrad', 'UNBEKANNT')}\n\n{aufgabe}"
    ))]


def _invoke_mit_fallback(agent, messages: list, agent_name: str) -> dict:
    """
    Ruft einen Agenten auf und fängt alle Exceptions ab.
    Im Fehlerfall wird eine AIMessage mit Fehlerbeschreibung zurückgegeben,
    damit der Workflow weiterlaufen kann.
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
    """Gibt den vollständigen Message-Trace eines Agenten strukturiert auf der Konsole aus."""
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


# ── Persistenz ────────────────────────────────────────────────

def _agent_log_schreiben(
    einsatz_id: str,
    agent_name: str,
    status: str,
    tool_calls: list[dict],
    antwort: str,
) -> None:
    """Persistiert einen Agenten-Laufeintrag in der Tabelle agent_logs."""
    ddl = """
    CREATE TABLE IF NOT EXISTS agent_logs (
        id          SERIAL PRIMARY KEY,
        einsatz_id  TEXT,
        agent_name  TEXT,
        status      TEXT,
        tool_calls  JSONB,
        antwort     TEXT,
        zeitstempel TIMESTAMPTZ DEFAULT now()
    );
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(ddl)
            conn.execute(
                """
                INSERT INTO agent_logs
                    (einsatz_id, agent_name, status, tool_calls, antwort)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    einsatz_id,
                    agent_name,
                    status,
                    json.dumps(tool_calls, ensure_ascii=False),
                    antwort,
                ),
            )
            conn.commit()
    except Exception as exc:
        print(f"[LOG] Warnung: Log konnte nicht gespeichert werden – {exc}")
