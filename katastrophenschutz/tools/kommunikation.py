"""Tools: Kommunikation – Lageberichte & Broadcasts."""
from __future__ import annotations

from typing import Literal

from langchain_core.tools import ToolException, tool


@tool
def broadcast_senden(
    nachricht: str,
    empfaenger: list[str],
    prioritaet: Literal["SOFORT", "HOCH", "NORMAL"],
) -> dict:
    """Sendet eine Broadcast-Nachricht an alle angegebenen Einheiten."""
    try:
        return {
            "gesendet":    True,
            "empfaenger":  empfaenger,
            "zugestellt":  len(empfaenger),
            "prioritaet":  prioritaet,
            "timestamp":   "15.01.2024 19:52:30",
        }
    except Exception as exc:
        raise ToolException(f"Broadcast konnte nicht gesendet werden: {exc}") from exc


@tool
def lagebericht_erstellen(einsatz_id: str, phase: str, inhalt: str) -> str:
    """Erstellt einen offiziellen Lagebericht für die Einsatzleitung."""
    try:
        linie_doppelt = "═" * 55
        trennlinie    = "─" * 55
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
