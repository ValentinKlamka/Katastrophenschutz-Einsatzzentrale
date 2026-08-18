"""
Katastrophenschutz – Datenbankschicht
========================================
Alle PostgreSQL-Operationen: Einsätze anlegen, Patienten verwalten.
"""
from __future__ import annotations

import json
import random
import re
import string

import psycopg
from langchain_core.messages import HumanMessage

from .config import DATABASE_URL, get_llm
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


def _neue_patient_id() -> str:
    chars = string.ascii_uppercase + string.digits
    return "PAT-" + "".join(random.choices(chars, k=6))


def patienten_batch_hinzufuegen(
    einsatz_id: str,
    patienten_texte: list[str],
) -> list[dict]:
    """
    Triagiert mehrere Patienten mit EINEM einzigen LLM-Aufruf.
    Das LLM klassifiziert direkt nach START-Schema.
    Gibt eine Liste von Triage-Ergebnis-Dicts zurück.
    """
    if not patienten_texte:
        return []

    llm = get_llm()
    n = len(patienten_texte)
    liste = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(patienten_texte))

    beispiel = ", ".join(
        f'{{"nr": {i+1}, "schweregrad": "ROT", "prioritaet": 1, "erstversorgung": "...", "transport": "sofort|baldmöglichst|ambulant|kein Transport"}}'
        for i in range(n)
    )
    prompt = (
        "Du bist Notarzt an einem Massenanfall von Verletzten (MANV) und führst die Triage "
        "nach dem START-Schema durch.\n\n"
        "## START-SCHEMA – Klassifikationsregeln (in dieser Reihenfolge prüfen)\n\n"
        "**SCHWARZ** – Verstorben oder hoffnungslos:\n"
        "- Keine Eigenatmung trotz Atemwegsöffnung\n"
        "- Keine Lebenszeichen, kein Puls\n"
        "- Verletzungen nicht mit dem Leben vereinbar\n\n"
        "**ROT** – Sofortbehandlung (lebensbedrohlich, aber behandelbar):\n"
        "- Atemfrequenz < 10 oder > 29/min, oder Atmung nur nach Atemwegsöffnung\n"
        "- Kapilläre Füllungszeit > 2 Sekunden oder kein radialer Puls\n"
        "- Bewusstlosigkeit oder stark reduziertes Bewusstsein (GCS < 14)\n"
        "- Starke, schwer stillbare Blutung\n"
        "- Schockzeichen (kalte/blasse Haut, schwacher/fadenförmiger Puls)\n\n"
        "**GELB** – Aufgeschobene Behandlung (ernst, aber stabil):\n"
        "- Atemwege frei, Atmung ausreichend, Kreislauf stabil\n"
        "- Aber: offene Frakturen, geschlossene Frakturen langer Röhrenknochen\n"
        "- Mittelschwere Verbrennungen (10–20% KOF)\n"
        "- Getrübtes Bewusstsein ohne akute Vitalbedrohung\n"
        "- Rückenverletzungen mit neurologischen Ausfällen\n\n"
        "**GRUEN** – Leichtverletzt / kann warten:\n"
        "- Gehfähig, orientiert, keine lebensbedrohlichen Verletzungen\n"
        "- Kleinere Wunden, leichte Prellungen, leichte Verbrennungen\n"
        "- Stabile Vitalzeichen\n\n"
        f"## Zu triagierender{'r Patient' if n == 1 else 'e Patienten'} ({n} gesamt)\n\n"
        f"{liste}\n\n"
        "## Ausgabeformat\n"
        f"Antworte mit einem JSON-Array mit GENAU {n} Objekten (gleiche Reihenfolge wie oben).\n"
        "Erlaubte Werte:\n"
        "  schweregrad: \"SCHWARZ\" | \"ROT\" | \"GELB\" | \"GRUEN\"\n"
        "  prioritaet:  0 (SCHWARZ) | 1 (ROT) | 2 (GELB) | 3 (GRUEN)\n"
        "  erstversorgung: konkrete medizinische Sofortmaßnahme\n"
        "  transport: \"sofort\" (ROT) | \"baldmöglichst\" (GELB) | \"ambulant\" (GRUEN) | \"kein Transport\" (SCHWARZ)\n\n"
        "Nur das JSON-Array ausgeben, kein Markdown, kein Text:\n"
        f"[{beispiel}]"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    raw = m.group(0) if m else raw

    def _parse(text: str):
        import ast
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(text)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
        raise RuntimeError(f"LLM-Antwort konnte nicht geparst werden: {text[:300]}")

    parsed = _parse(raw)

    if len(parsed) < n:
        print(
            f"[TRIAGE-BATCH] Warnung: LLM lieferte {len(parsed)} von {n} Einträgen – "
            f"fehlende mit GRUEN aufgefüllt."
        )
        for _ in range(n - len(parsed)):
            parsed.append({
                "schweregrad": "GRUEN", "prioritaet": 3,
                "erstversorgung": "Erste Hilfe, Registrierung", "transport": "ambulant",
            })

    _SCHWEREGRAD_RANG = {"UNBEKANNT": 0, "GRUEN": 1, "GELB": 2, "ROT": 3, "SCHWARZ": 4}
    ergebnisse: list[dict] = []

    with psycopg.connect(DATABASE_URL) as conn:
        for stmt in _DDL_PATIENTEN.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

        for item in parsed[:n]:
            kat            = item.get("schweregrad",   "GRUEN").upper()
            prio           = item.get("prioritaet",    3)
            erstversorgung = item.get("erstversorgung", "")
            transport      = item.get("transport",     "ambulant")
            if kat not in _SCHWEREGRAD_RANG:
                kat = "GRUEN"
            # Enforce correct values for SCHWARZ regardless of LLM output
            if kat == "SCHWARZ":
                prio           = 0
                erstversorgung = "Keine aktiven Maßnahmen – würdevoller Umgang"
                transport      = "kein Transport"
            pid = _neue_patient_id()

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
                (einsatz_id, pid, kat, prio, erstversorgung, transport),
            )
            ergebnisse.append({
                "patient_id":     pid,
                "schweregrad":    kat,
                "prioritaet":     prio,
                "erstversorgung": erstversorgung,
                "transport":      transport,
            })

        if ergebnisse:
            max_sg = max(
                ergebnisse,
                key=lambda e: _SCHWEREGRAD_RANG.get(e["schweregrad"], 0),
            )["schweregrad"]
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
                (max_sg, einsatz_id, _SCHWEREGRAD_RANG.get(max_sg, 0)),
            )
        conn.commit()

    _agent_log_schreiben(
        einsatz_id=einsatz_id,
        agent_name="TRIAGE-BATCH",
        status="OK",
        tool_calls=[],
        antwort=(
            f"{len(ergebnisse)} Patient(en) triagiert: "
            + str([e["schweregrad"] for e in ergebnisse])
        ),
    )
    print(f"[DB] {len(ergebnisse)} Patient(en) zu Einsatz {einsatz_id} hinzugefügt.")
    return ergebnisse


def patient_hinzufuegen(einsatz_id: str, patient_beschreibung: str) -> dict:
    """
    Triagiert einen einzelnen Patienten (dünner Wrapper um patienten_batch_hinzufuegen).
    Gibt das Triage-Ergebnis-Dict zurück.
    """
    ergebnisse = patienten_batch_hinzufuegen(einsatz_id, [patient_beschreibung])
    if not ergebnisse:
        raise RuntimeError("Triage hat kein Ergebnis zurückgegeben.")
    return ergebnisse[0]


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
