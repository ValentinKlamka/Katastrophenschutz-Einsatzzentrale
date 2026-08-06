"""
Katastrophenschutz – Datenbankschicht
========================================
Alle PostgreSQL-Operationen: Einsätze anlegen, Patienten verwalten.
"""
from __future__ import annotations

import json

import psycopg

from .config import DATABASE_URL
from .state  import EinsatzState
from .utils  import _agent_log_schreiben, _einsatz_id_generieren, _invoke_mit_fallback

# ── DDL-Fragmente ─────────────────────────────────────────────

_DDL_EINSAETZE = """
CREATE TABLE IF NOT EXISTS einsaetze (
    einsatz_id   TEXT PRIMARY KEY,
    standort     TEXT,
    beschreibung TEXT,
    schweregrad  TEXT,
    triage_ok    BOOLEAN,
    ressourcen_ok BOOLEAN,
    transport_ok BOOLEAN,
    bericht_ok   BOOLEAN,
    runden       INTEGER,
    fallbacks    TEXT[],
    zeitstempel  TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE einsaetze ADD COLUMN IF NOT EXISTS beschreibung TEXT;
"""

_DDL_PATIENTEN = """
CREATE TABLE IF NOT EXISTS patienten (
    id                SERIAL PRIMARY KEY,
    einsatz_id        TEXT REFERENCES einsaetze(einsatz_id) ON DELETE CASCADE,
    patient_id        TEXT,
    schweregrad       TEXT,
    prioritaet        INTEGER,
    erstversorgung    TEXT,
    transport         TEXT,
    manuell_geaendert BOOLEAN DEFAULT FALSE,
    zeitstempel       TIMESTAMPTZ DEFAULT now(),
    UNIQUE (einsatz_id, patient_id)
);
ALTER TABLE patienten ADD COLUMN IF NOT EXISTS manuell_geaendert BOOLEAN DEFAULT FALSE;
"""


_DDL_AGENT_LOGS = """
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


# ── Öffentliche API ───────────────────────────────────────────

def db_initialisieren() -> None:
    """Erstellt alle Tabellen falls noch nicht vorhanden. Beim App-Start aufrufen."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            for ddl in [_DDL_EINSAETZE, _DDL_PATIENTEN, _DDL_AGENT_LOGS]:
                for stmt in ddl.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
            conn.commit()
        print("[DB] Tabellen initialisiert.")
    except Exception as exc:
        print(f"[DB] Warnung: Initialisierung fehlgeschlagen – {exc}")


def einsatz_anlegen(
    standort: str,
    beschreibung: str,
    einsatz_id: str | None = None,
) -> str:
    """
    Legt einen neuen Einsatz in der Datenbank an (ohne Workflow-Ausführung).
    Gibt die Einsatz-ID zurück.
    """
    eid = einsatz_id or _einsatz_id_generieren()
    with psycopg.connect(DATABASE_URL) as conn:
        for stmt in _DDL_EINSAETZE.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.execute(
            """
            INSERT INTO einsaetze
                (einsatz_id, standort, beschreibung, schweregrad,
                 triage_ok, ressourcen_ok, transport_ok, bericht_ok, runden, fallbacks)
            VALUES (%s, %s, %s, 'UNBEKANNT', FALSE, FALSE, FALSE, FALSE, 0, ARRAY[]::TEXT[])
            ON CONFLICT (einsatz_id) DO NOTHING;
            """,
            (eid, standort, beschreibung),
        )
        conn.commit()
    print(f"[DB] Einsatz {eid} angelegt.")
    return eid


def patient_hinzufuegen(einsatz_id: str, patient_beschreibung: str) -> dict:
    """
    Führt eine Triage für einen einzelnen Patienten durch und speichert das Ergebnis.
    Gibt das Triage-Ergebnis-Dict zurück.
    """
    # Import hier um zirkuläre Abhängigkeit zu vermeiden
    from .agents.triage import triage_agent_aufbauen
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    agent = triage_agent_aufbauen()
    nachrichten = [HumanMessage(content=(
        f"Einsatz {einsatz_id}: Bewerte folgenden Patienten nach START-Schema "
        f"und rufe patient_triage_bewerten() auf:\n\n{patient_beschreibung}"
    ))]
    result = _invoke_mit_fallback(agent, nachrichten, "TRIAGE-AGENT (Einzelpatient)")

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    agent = triage_agent_aufbauen()
    nachrichten = [HumanMessage(content=(
        f"Einsatz {einsatz_id}: Bewerte folgenden Patienten nach START-Schema "
        f"und rufe patient_triage_bewerten() auf:\n\n{patient_beschreibung}"
    ))]
    result = _invoke_mit_fallback(agent, nachrichten, "TRIAGE-AGENT (Einzelpatient)")

    # Triage-Ergebnis aus ToolMessage extrahieren
    triage_ergebnis: dict = {}
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "patient_triage_bewerten":
            try:
                triage_ergebnis = (
                    json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                )
            except (json.JSONDecodeError, TypeError):
                pass

    ist_fallback = "_fallback_agent" in result

    # Log immer schreiben – unabhängig davon ob Triage erfolgreich war
    _agent_log_schreiben(
        einsatz_id=einsatz_id,
        agent_name="TRIAGE-AGENT",
        status="FALLBACK" if ist_fallback else ("FEHLER" if not triage_ergebnis else "OK"),
        tool_calls=[
            {"tool": tc["name"], "args": tc.get("args", {})}
            for msg in result["messages"] if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        ],
        antwort=str(result["messages"][-1].content),
    )

    if not triage_ergebnis:
        raise RuntimeError(
            "Triage-Agent hat kein patient_triage_bewerten-Ergebnis zurückgegeben."
        )

    neuer_sg = triage_ergebnis.get("schweregrad", "UNBEKANNT")
    schweregrad_rang = {"UNBEKANNT": 0, "GRUEN": 1, "GELB": 2, "ROT": 3, "SCHWARZ": 4}

    with psycopg.connect(DATABASE_URL) as conn:
        for stmt in _DDL_PATIENTEN.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.execute(
            """
            INSERT INTO patienten
                (einsatz_id, patient_id, schweregrad, prioritaet, erstversorgung, transport)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (einsatz_id, patient_id) DO UPDATE SET
                schweregrad    = EXCLUDED.schweregrad,
                prioritaet     = EXCLUDED.prioritaet,
                erstversorgung = EXCLUDED.erstversorgung,
                transport      = EXCLUDED.transport,
                zeitstempel    = now();
            """,
            (
                einsatz_id,
                triage_ergebnis.get("patient_id"),
                neuer_sg,
                triage_ergebnis.get("prioritaet"),
                triage_ergebnis.get("erstversorgung"),
                triage_ergebnis.get("transport"),
            ),
        )
        conn.execute(
            """
            UPDATE einsaetze SET
                schweregrad = %s,
                triage_ok   = TRUE
            WHERE einsatz_id = %s
              AND %s > COALESCE(
                (SELECT rang FROM (VALUES
                    ('UNBEKANNT',0),('GRUEN',1),('GELB',2),('ROT',3),('SCHWARZ',4)
                ) AS t(sg, rang) WHERE sg = einsaetze.schweregrad), 0
              );
            """,
            (neuer_sg, einsatz_id, schweregrad_rang.get(neuer_sg, 0)),
        )
        conn.commit()

    print(
        f"[DB] Patient {triage_ergebnis.get('patient_id')} ({neuer_sg}) "
        f"zu Einsatz {einsatz_id} hinzugefügt."
    )
    return triage_ergebnis


def patient_aktualisieren(
    einsatz_id: str,
    patient_id: str,
    neuer_schweregrad: str,
    grund: str = "",
) -> None:
    """Aktualisiert den Schweregrad eines Patienten manuell. Setzt manuell_geaendert=TRUE."""
    erlaubt = {"GRUEN", "GELB", "ROT", "SCHWARZ"}
    if neuer_schweregrad not in erlaubt:
        raise ValueError(f"Ungültiger Schweregrad '{neuer_schweregrad}'. Erlaubt: {erlaubt}")

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            """
            UPDATE patienten SET
                schweregrad       = %s,
                manuell_geaendert = TRUE,
                zeitstempel       = now()
            WHERE einsatz_id = %s AND patient_id = %s;
            """,
            (neuer_schweregrad, einsatz_id, patient_id),
        )
        if conn.rowcount == 0:
            raise ValueError(
                f"Patient '{patient_id}' in Einsatz '{einsatz_id}' nicht gefunden."
            )
        conn.commit()

    hinweis = f" ({grund})" if grund else ""
    print(f"[DB] Patient {patient_id} → {neuer_schweregrad}{hinweis} (manuell).")


def einsatz_speichern(state: EinsatzState) -> None:
    """Schreibt Einsatz-Zusammenfassung und Patientendaten in die Datenbank (Upsert)."""
    upsert = """
    INSERT INTO einsaetze
        (einsatz_id, standort, beschreibung, schweregrad, triage_ok, ressourcen_ok,
         transport_ok, bericht_ok, runden, fallbacks)
    VALUES
        (%(einsatz_id)s, %(standort)s, %(beschreibung)s, %(schweregrad)s, %(triage_ok)s,
         %(ressourcen_ok)s, %(transport_ok)s, %(bericht_ok)s, %(runden)s, %(fallbacks)s)
    ON CONFLICT (einsatz_id) DO UPDATE SET
        schweregrad   = EXCLUDED.schweregrad,
        beschreibung  = EXCLUDED.beschreibung,
        triage_ok     = EXCLUDED.triage_ok,
        ressourcen_ok = EXCLUDED.ressourcen_ok,
        transport_ok  = EXCLUDED.transport_ok,
        bericht_ok    = EXCLUDED.bericht_ok,
        runden        = EXCLUDED.runden,
        fallbacks     = EXCLUDED.fallbacks,
        zeitstempel   = now();
    """
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
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            for stmt in _DDL_EINSAETZE.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            for stmt in _DDL_PATIENTEN.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.execute(upsert, {
                "einsatz_id":    state["einsatz_id"],
                "standort":      state["standort"],
                "beschreibung":  state.get("beschreibung", ""),
                "schweregrad":   state["schweregrad"],
                "triage_ok":     state.get("triage_abgeschlossen", False),
                "ressourcen_ok": state.get("ressourcen_reserviert", False),
                "transport_ok":  state.get("transport_koordiniert", False),
                "bericht_ok":    state.get("bericht_erstellt", False),
                "runden":        state.get("runden", 0),
                "fallbacks":     state.get("fallbacks", []),
            })
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
        print(
            f"[DB] Einsatz {state['einsatz_id']} gespeichert "
            f"({len(state.get('patienten', []))} Patient(en))."
        )
    except Exception as exc:
        print(f"[DB] Warnung: Einsatz konnte nicht gespeichert werden – {exc}")
