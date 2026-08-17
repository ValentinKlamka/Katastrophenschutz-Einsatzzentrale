"""
Katastrophenschutz-Einsatzleitzentrale · Streamlit Dashboard
"""
from __future__ import annotations

import psycopg
import pandas as pd
import streamlit as st
import plotly.express as px

from workflow import (
    DATABASE_URL,
    EINSATZMELDUNG,
    KRANKENHAEUSER,
    _einsatz_id_generieren,
    db_initialisieren,
    einsatz_anlegen,
    einsatz_koordinieren,
    patient_aktualisieren,
    patient_hinzufuegen,
    patienten_batch_hinzufuegen,
)

# Tabellen beim ersten Start anlegen
db_initialisieren()

EINSATZ_TEMPLATES = [
    {
        "label": "🔥 Explosion / Brand",
        "standort": "Industriepark Nord, Halle 3",
        "beschreibung": (
            "Explosion in der Produktionshalle 3 des Industrieparks Nord. "
            "Anschließender Großbrand mit starker Rauchentwicklung. "
            "Feuerwehr im Anmarsch, Halle teilweise eingestürzt. "
            "Mehrere Personen eingeschlossen oder verletzt im Freien."
        ),
        "patienten": (
            "Männlich ~40 J.: bewusstlos, eingeschränkte Atmung, schwacher Puls, Brandwunden Oberkörper\n"
            "Weiblich ~35 J.: getrübtes Bewusstsein, offene Fraktur rechter Oberschenkel, starke Blutung\n"
            "Männlich ~22 J.: klar, Schnittwunden Arme, leichte Rauchgasvergiftung"
        ),
    },
    {
        "label": "🚗 Massenunfall Autobahn",
        "standort": "A81 km 23, Fahrtrichtung Nord",
        "beschreibung": (
            "Massenkarambolage auf der A81 bei km 23 mit ca. 12 Fahrzeugen. "
            "Auslöser: Aquaplaning bei Starkregen, Kettenreaktion. "
            "Vollsperrung in Fahrtrichtung Nord. "
            "Ersthelfer vor Ort, RTW und Feuerwehr alarmiert."
        ),
        "patienten": (
            "Weiblich ~55 J.: klar, Thoraxtrauma, Schock, schwacher Puls\n"
            "Männlich ~30 J.: bewusstlos, keine Eigenatmung feststellbar, kein Puls\n"
            "Kind ~8 J.: getrübtes Bewusstsein, Kopfplatzwunde, Prellungen\n"
            "Männlich ~45 J.: klar, Fraktur Halswirbel V. (Verdacht), immobilisiert"
        ),
    },
    {
        "label": "☣ Chemieunfall / HAZMAT",
        "standort": "Chemiewerk Südring, Tanklager B",
        "beschreibung": (
            "Leck an einem Chlorgas-Behälter im Tanklager B des Chemiewerks Südring. "
            "Gasausbreitung mit Wind Richtung Wohngebiet. "
            "Werkschutz hat Betrieb evakuiert, Schutzzone 500 m eingerichtet. "
            "Mehrere Personen mit Reizgassymptomen, ein Mitarbeiter direkt exponiert."
        ),
        "patienten": (
            "Männlich ~50 J.: Direktexposition, Bewusstlosigkeit, keine Eigenatmung, kein Puls\n"
            "Weiblich ~28 J.: getrübt, starke Hustenreize, tränende Augen, Atemnot\n"
            "Männlich ~60 J.: klar, Schleimhautreizung, leichte Atemnot, geht selbständig"
        ),
    },
]

st.set_page_config(
    page_title="Einsatzleitzentrale",
    page_icon="🚨",
    layout="wide",
)

# ── Hilfsfunktionen ───────────────────────────────────────────

SCHWEREGRAD_FARBE = {
    "SCHWARZ": ("#ffffff", "#2d2d2d"),   # text, bg
    "ROT":     ("#ffffff", "#c0392b"),
    "GELB":    ("#333333", "#f39c12"),
    "GRUEN":   ("#ffffff", "#27ae60"),
    "UNBEKANNT": ("#333333", "#bdc3c7"),
}

SCHWEREGRAD_ROW_FARBE = {
    "SCHWARZ": "#e0e0e0",
    "ROT":     "#fde8e8",
    "GELB":    "#fef9e7",
    "GRUEN":   "#eafaf1",
    "UNBEKANNT": "#f5f5f5",
}


def sg_badge_html(sg: str) -> str:
    text_c, bg_c = SCHWEREGRAD_FARBE.get(sg, ("#333", "#bdc3c7"))
    return (
        f'<span style="background:{bg_c};color:{text_c};'
        f'padding:3px 10px;border-radius:4px;font-weight:bold;font-size:0.85em">'
        f'{sg}</span>'
    )





# ── Datenbank-Abfragen ────────────────────────────────────────

@st.cache_data(ttl=5)
def lade_einsaetze() -> pd.DataFrame:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute("""
                SELECT einsatz_id, standort, schweregrad,
                       triage_ok, ressourcen_ok, transport_ok, bericht_ok,
                       runden, zeitstempel
                FROM einsaetze
                ORDER BY zeitstempel DESC
            """).fetchall()
        cols = ["Einsatz-ID", "Standort", "Schweregrad",
                "Triage", "Ressourcen", "Transport", "Bericht",
                "Runden", "Zeitstempel"]
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"DB-Verbindung fehlgeschlagen: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def lade_patienten(einsatz_id: str) -> pd.DataFrame:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute("""
                SELECT patient_id, schweregrad, prioritaet,
                       erstversorgung, transport, manuell_geaendert, zeitstempel
                FROM patienten
                WHERE einsatz_id = %s
                ORDER BY prioritaet, patient_id
            """, (einsatz_id,)).fetchall()
        cols = ["Patient-ID", "Schweregrad", "Priorität",
                "Erstversorgung", "Transport", "Manuell", "Zeitstempel"]
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"Patientendaten nicht abrufbar: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def lade_einsatz_ids() -> list[str]:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute(
                "SELECT einsatz_id FROM einsaetze ORDER BY zeitstempel DESC"
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=5)
def lade_einsatz_detail(einsatz_id: str) -> tuple | None:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            return conn.execute("""
                SELECT standort, beschreibung, schweregrad, triage_ok,
                       ressourcen_ok, transport_ok, bericht_ok, runden,
                       fallbacks, zeitstempel
                FROM einsaetze WHERE einsatz_id = %s
            """, (einsatz_id,)).fetchone()
    except Exception as exc:
        st.error(str(exc))
        return None


@st.cache_data(ttl=5)
def lade_triage_verteilung(einsatz_id: str) -> pd.DataFrame:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute("""
                SELECT schweregrad, COUNT(*) AS anzahl
                FROM patienten
                WHERE einsatz_id = %s
                GROUP BY schweregrad
                ORDER BY
                    CASE schweregrad
                        WHEN 'ROT'      THEN 1
                        WHEN 'SCHWARZ'  THEN 2
                        WHEN 'GELB'     THEN 3
                        WHEN 'GRUEN'    THEN 4
                        ELSE 5
                    END
            """, (einsatz_id,)).fetchall()
        return pd.DataFrame(rows, columns=["Schweregrad", "Anzahl"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def lade_krankenhaus_anfragen(einsatz_id: str) -> pd.DataFrame:
    """Liest aus agent_logs, welche Krankenhäuser der Ressourcen-Agent angefragt hat."""
    try:
        import json as _json
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute("""
                SELECT tool_calls
                FROM agent_logs
                WHERE einsatz_id = %s AND agent_name = 'RESSOURCEN-AGENT'
                ORDER BY zeitstempel DESC
                LIMIT 1
            """, (einsatz_id,)).fetchall()
        anfragen = []
        for (tc_raw,) in rows:
            calls = tc_raw if isinstance(tc_raw, list) else _json.loads(tc_raw or "[]")
            for tc in calls:
                if tc.get("tool") == "krankenhaus_anfragen":
                    args = tc.get("args", {})
                    kh_id = args.get("krankenhaus_id", "?")
                    kh_info = KRANKENHAEUSER.get(kh_id, {})
                    anfragen.append({
                        "Krankenhaus": kh_id,
                        "Bettentyp": args.get("bett_typ", "?"),
                        "Spezialisierung angefragt": args.get("benoetigt_spezialisierung", "?"),
                        "Freie Betten (Sim.)": kh_info.get(args.get("bett_typ", ""), "?"),
                        "Spezialisierungen": ", ".join(kh_info.get("spezialisierungen", [])),
                    })
        return pd.DataFrame(anfragen)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def lade_agent_logs(einsatz_id: str) -> pd.DataFrame:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute("""
                SELECT agent_name, status, tool_calls, antwort, zeitstempel
                FROM agent_logs
                WHERE einsatz_id = %s
                ORDER BY zeitstempel
            """, (einsatz_id,)).fetchall()
        cols = ["Agent", "Status", "Tool-Calls", "Antwort", "Zeitstempel"]
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"Logs nicht abrufbar: {exc}")
        return pd.DataFrame()


# ── Navigation ────────────────────────────────────────────────

st.sidebar.title("🚨 Einsatzleitzentrale")
st.sidebar.markdown("---")
seite = st.sidebar.radio(
    "Navigation",
    ["📋 Übersicht", "➕ Neuer Einsatz", "🔍 Einsatz-Detail"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Katastrophenschutz A2A\nLangGraph + Gemini")


# ══════════════════════════════════════════════════════════════
#  SEITE: ÜBERSICHT
# ══════════════════════════════════════════════════════════════

if seite == "📋 Übersicht":
    col_h, col_btn = st.columns([5, 1])
    col_h.title("📋 Einsatz-Übersicht")
    if col_btn.button("🔄", help="Aktualisieren"):
        st.cache_data.clear()

    df = lade_einsaetze()

    if df.empty:
        st.info("Noch keine Einsätze in der Datenbank.")
    else:
        # Kennzahlen
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Einsätze gesamt", len(df))
        c2.metric("Kritisch (ROT/SCHWARZ)",
                  len(df[df["Schweregrad"].isin(["ROT", "SCHWARZ"])]))
        c3.metric("Abgeschlossen",
                  len(df[df["Bericht"] == True]))  # noqa: E712
        c4.metric("In Bearbeitung",
                  len(df[df["Bericht"] != True]))   # noqa: E712

        st.markdown("---")

        # Tabelle mit Farbgebung
        def _zeilen_farbe(row: pd.Series) -> list[str]:
            bg = SCHWEREGRAD_ROW_FARBE.get(row["Schweregrad"], "")
            return [f"background-color: {bg}"] * len(row)

        df_anzeige = df.copy()
        for col in ["Triage", "Ressourcen", "Transport", "Bericht"]:
            df_anzeige[col] = df_anzeige[col].map(
                {True: "✓", False: "○", None: "○"}
            )
        df_anzeige["Zeitstempel"] = pd.to_datetime(
            df_anzeige["Zeitstempel"]
        ).dt.strftime("%d.%m.%Y %H:%M")

        st.dataframe(
            df_anzeige.style.apply(_zeilen_farbe, axis=1),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════
#  SEITE: NEUER EINSATZ
# ══════════════════════════════════════════════════════════════

elif seite == "➕ Neuer Einsatz":
    st.title("➕ Neuer Einsatz")
    tab_manuell, tab_workflow = st.tabs(
        ["📝 Manuell anlegen", "▶ Vollständiger Workflow"]
    )

    # ── Tab: Manuell ─────────────────────────────────────────
    with tab_manuell:
        # Felder vorbelegen, falls noch nicht in session_state
        for _k, _v in [
            ("manuell_standort", ""),
            ("manuell_beschreibung", ""),
            ("manuell_patienten", ""),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # Template-Buttons
        st.caption("Vorlage laden:")
        _cols = st.columns(len(EINSATZ_TEMPLATES))
        for _col, _tpl in zip(_cols, EINSATZ_TEMPLATES):
            if _col.button(_tpl["label"], use_container_width=True):
                st.session_state["manuell_standort"]     = _tpl["standort"]
                st.session_state["manuell_beschreibung"] = _tpl["beschreibung"]
                st.session_state["manuell_patienten"]    = _tpl["patienten"]

        with st.form("einsatz_form", border=True):
            st.subheader("Stammdaten")
            id_modus = st.radio(
                "Einsatz-ID",
                ["Automatisch generieren", "Manuell eingeben"],
                horizontal=True,
            )
            einsatz_id_input = ""
            if id_modus == "Manuell eingeben":
                einsatz_id_input = st.text_input(
                    "Einsatz-ID", placeholder="E-2026-0728-001"
                )

            standort = st.text_input(
                "Standort / Einsatzort *",
                key="manuell_standort",
            )
            beschreibung = st.text_area(
                "Lagebeschreibung *",
                height=120,
                key="manuell_beschreibung",
            )

            st.subheader("Patienten (optional)")
            st.caption(
                "Je Patient eine Zeile. Patienten können auch später über "
                "**Einsatz-Detail** hinzugefügt werden."
            )
            patienten_text = st.text_area(
                "Patientenbeschreibungen",
                height=100,
                key="manuell_patienten",
                label_visibility="collapsed",
            )

            submitted = st.form_submit_button("▶ Einsatz anlegen & Workflow starten", type="primary")

        if submitted:
            if not standort.strip() or not beschreibung.strip():
                st.error("Standort und Beschreibung sind Pflichtfelder.")
            else:
                # Meldung aus Formularfeldern zusammenbauen
                zeilen = [
                    z.strip()
                    for z in patienten_text.strip().splitlines()
                    if z.strip()
                ]
                meldung_teile = [
                    f"Standort: {standort.strip()}",
                    f"Lagebeschreibung: {beschreibung.strip()}",
                ]
                if zeilen:
                    meldung_teile.append("Patienten:\n" + "\n".join(f"- {z}" for z in zeilen))
                meldung_komplett = "\n\n".join(meldung_teile)

                with st.spinner("A2A-Workflow läuft … (Triage → Ressourcen → Logistik → Kommunikation)"):
                    try:
                        final = einsatz_koordinieren(
                            meldung=meldung_komplett,
                            einsatz_id=einsatz_id_input.strip() or None,
                            standort=standort.strip(),
                        )
                        eid = final["einsatz_id"]
                        st.success(f"Einsatz **{eid}** angelegt und vollständig koordiniert.")
                        st.cache_data.clear()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Schweregrad", final["schweregrad"])
                        c2.metric("Triage",      "✓" if final.get("triage_abgeschlossen") else "✗")
                        c3.metric("Runden",      final.get("runden", 0))
                        c4.metric("Patienten",   len(final.get("patienten", [])))
                        if final.get("fallbacks"):
                            st.warning(f"⚠ Fallbacks: {', '.join(final['fallbacks'])}")
                    except Exception as exc:
                        st.error(f"Fehler beim Workflow: {exc}")

    # ── Tab: Vollständiger Workflow ───────────────────────────
    with tab_workflow:
        st.caption(
            "Führt den kompletten A2A-Workflow durch: "
            "Triage → Ressourcen → Logistik → Kommunikation."
        )
        with st.form("workflow_form", border=True):
            meldung = st.text_area(
                "Einsatzmeldung",
                value=EINSATZMELDUNG.strip(),
                height=280,
            )
            starten = st.form_submit_button("▶ Workflow starten", type="primary")

        if starten:
            with st.spinner("A2A-Workflow läuft … (1–2 Minuten)"):
                try:
                    final = einsatz_koordinieren(meldung)
                    st.success(
                        f"Workflow abgeschlossen – Einsatz **{final['einsatz_id']}**"
                    )
                    st.cache_data.clear()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Schweregrad", final["schweregrad"])
                    c2.metric("Triage", "✓" if final.get("triage_abgeschlossen") else "✗")
                    c3.metric("Runden", final.get("runden", 0))
                    c4.metric("Patienten", len(final.get("patienten", [])))
                except Exception as exc:
                    st.error(f"Workflow-Fehler: {exc}")


# ══════════════════════════════════════════════════════════════
#  SEITE: EINSATZ-DETAIL
# ══════════════════════════════════════════════════════════════

elif seite == "🔍 Einsatz-Detail":
    st.title("🔍 Einsatz-Detail")

    ids = lade_einsatz_ids()
    if not ids:
        st.info("Noch keine Einsätze vorhanden. Bitte zuerst einen Einsatz anlegen.")
        st.stop()

    col_sel, col_ref = st.columns([5, 1])
    selected_id = col_sel.selectbox("Einsatz auswählen", ids, label_visibility="collapsed")
    if col_ref.button("🔄", help="Aktualisieren"):
        st.cache_data.clear()

    row = lade_einsatz_detail(selected_id)
    if not row:
        st.stop()

    (standort, beschreibung, schweregrad, triage_ok,
     res_ok, trans_ok, bericht_ok, runden, fallbacks, zeitstempel) = row

    # Header
    col_info, col_sg = st.columns([3, 1])
    with col_info:
        st.markdown(f"## {selected_id}")
        st.markdown(f"**Standort:** {standort or '–'}")
        ts = zeitstempel.strftime("%d.%m.%Y %H:%M") if zeitstempel else "–"
        st.markdown(f"**Erfasst:** {ts}")
        if beschreibung:
            with st.expander("Einsatzmeldung anzeigen"):
                st.text(beschreibung)

    with col_sg:
        text_c, bg_c = SCHWEREGRAD_FARBE.get(schweregrad, ("#333", "#bdc3c7"))
        st.markdown(
            f"""<div style="background:{bg_c};color:{text_c};padding:20px;
            border-radius:8px;text-align:center;margin-top:8px">
            <div style="font-size:0.75em;letter-spacing:1px">SCHWEREGRAD</div>
            <div style="font-size:2.2em;font-weight:bold">{schweregrad}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")
        c1, c2 = st.columns(2)
        c1.metric("Triage",     "✓" if triage_ok else "○")
        c2.metric("Ressourcen", "✓" if res_ok    else "○")
        c1.metric("Transport",  "✓" if trans_ok  else "○")
        c2.metric("Bericht",    "✓" if bericht_ok else "○")

    if fallbacks:
        st.warning(f"⚠ Fallbacks ausgelöst: {', '.join(fallbacks)}")

    st.markdown("---")

    # ── Patientenliste ────────────────────────────────────────
    # ── Triage-Kreisdiagramm ─────────────────────────────────
    st.subheader("Patienten")
    df_vert = lade_triage_verteilung(selected_id)
    df_pat = lade_patienten(selected_id)

    if not df_vert.empty:
        FARBEN = {
            "ROT":      "#c0392b",
            "SCHWARZ":  "#2d2d2d",
            "GELB":     "#f39c12",
            "GRUEN":    "#27ae60",
            "UNBEKANNT": "#bdc3c7",
        }
        farbliste = [FARBEN.get(sg, "#aaa") for sg in df_vert["Schweregrad"]]
        fig = px.pie(
            df_vert,
            names="Schweregrad",
            values="Anzahl",
            color="Schweregrad",
            color_discrete_map=FARBEN,
            hole=0.35,
        )
        fig.update_traces(
            textinfo="label+value",
            hovertemplate="%{label}: %{value} Patient(en)<extra></extra>",
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        col_pie, col_metrics = st.columns([2, 1])
        col_pie.plotly_chart(fig, use_container_width=True)
        with col_metrics:
            st.markdown("<br>", unsafe_allow_html=True)
            for _, r in df_vert.iterrows():
                text_c, bg_c = SCHWEREGRAD_FARBE.get(r["Schweregrad"], ("#333", "#bdc3c7"))
                st.markdown(
                    f'<div style="background:{bg_c};color:{text_c};padding:8px 14px;'
                    f'border-radius:6px;margin-bottom:6px;font-weight:bold">'
                    f'{r["Schweregrad"]}: {r["Anzahl"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Noch keine Patienten erfasst.")

    if df_pat.empty:
        st.info("Noch keine Patienten erfasst.")
    else:
        def _pat_farbe(row: pd.Series) -> list[str]:
            bg = SCHWEREGRAD_ROW_FARBE.get(row["Schweregrad"], "")
            return [f"background-color: {bg}"] * len(row)

        df_anzeige = df_pat.copy()
        df_anzeige["Manuell"] = df_anzeige["Manuell"].map(
            {True: "✎", False: "", None: ""}
        )
        df_anzeige["Zeitstempel"] = pd.to_datetime(
            df_anzeige["Zeitstempel"]
        ).dt.strftime("%d.%m.%Y %H:%M")

        st.dataframe(
            df_anzeige.style.apply(_pat_farbe, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # ── Patient aktualisieren ─────────────────────────────
        st.subheader("Patient aktualisieren")
        with st.form("patient_update", border=True):
            col_pid, col_sg2, col_grund = st.columns([2, 1, 3])
            pid_sel = col_pid.selectbox("Patient", df_pat["Patient-ID"].tolist())
            neuer_sg = col_sg2.selectbox(
                "Neuer Schweregrad", ["GRUEN", "GELB", "ROT", "SCHWARZ"]
            )
            grund = col_grund.text_input(
                "Grund (optional)", placeholder="Verstorben 20:14 Uhr"
            )
            upd_btn = st.form_submit_button("Aktualisieren")

        if upd_btn:
            try:
                patient_aktualisieren(selected_id, pid_sel, neuer_sg, grund)
                st.success(f"Patient **{pid_sel}** → **{neuer_sg}**")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.markdown("---")

    # ── Patient hinzufügen ────────────────────────────────────
    st.subheader("Patient hinzufügen")
    with st.form("patient_hinzu", border=True):
        pat_text = st.text_area(
            "Patientenbeschreibungen (eine pro Zeile)",
            height=100,
            placeholder=(
                "Weiblich ~28 J.: klar, Schnittwunden Hände, leichte Rauchgasvergiftung\n"
                "Männlich ~40 J.: bewusstlos, eingeschränkte Atmung, schwere Brandwunden"
            ),
            label_visibility="collapsed",
        )
        hinzu_btn = st.form_submit_button(
            "Triage starten & hinzufügen", type="primary"
        )

    if hinzu_btn:
        zeilen = [z.strip() for z in pat_text.strip().splitlines() if z.strip()]
        if not zeilen:
            st.error("Bitte mindestens eine Patientenbeschreibung eingeben.")
        else:
            with st.spinner(f"Triage läuft für {len(zeilen)} Patient(en)…"):
                try:
                    ergebnisse = patienten_batch_hinzufuegen(selected_id, zeilen)
                    for ergebnis in ergebnisse:
                        sg_neu = ergebnis.get("schweregrad", "?")
                        pid_neu = ergebnis.get("patient_id", "?")
                        st.markdown(
                            f"Patient **{pid_neu}** triagiert: "
                            + sg_badge_html(sg_neu),
                            unsafe_allow_html=True,
                        )
                    if len(ergebnisse) > 1:
                        with st.expander("Details"):
                            st.json(ergebnisse)
                    elif ergebnisse:
                        with st.expander("Details"):
                            st.json(ergebnisse[0])
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Fehler: {exc}")

    # ── Vollständigen Workflow fortsetzen ─────────────────────
    if not (triage_ok and res_ok and trans_ok and bericht_ok):
        st.markdown("---")
        st.subheader("Vollständigen Workflow fortsetzen")
        st.caption(
            "Startet Triage → Ressourcen → Logistik → Kommunikation "
            "auf Basis der gespeicherten Einsatzmeldung."
        )
        if st.button("▶ Workflow starten", type="primary"):
            with st.spinner("Workflow läuft..."):
                try:
                    final = einsatz_koordinieren(
                        meldung=beschreibung or selected_id,
                        einsatz_id=selected_id,
                        standort=standort or "UNBEKANNT",
                    )
                    st.success("Workflow abgeschlossen.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    # ── Krankenhausübersicht ──────────────────────────────────
    st.markdown("---")
    st.subheader("🏥 Krankenhäuser")
    st.caption(
        "Die Krankenhauswahl basiert auf **Schweregrad und Spezialisierung**, "
        "nicht auf dem Einsatzstandort. Der Ressourcen-Agent verhandelt A2A mit "
        "allen verfügbaren Häusern und wählt das passendste aus."
    )

    tab_anfragen, tab_alle = st.tabs(["📋 Angefragte (dieser Einsatz)", "🏥 Alle verfügbaren"])

    with tab_anfragen:
        df_kh = lade_krankenhaus_anfragen(selected_id)
        if df_kh.empty:
            st.info("Keine Krankenhausanfragen gefunden (Ressourcen-Agent noch nicht gelaufen).")
        else:
            st.dataframe(df_kh, use_container_width=True, hide_index=True)

    with tab_alle:
        kh_rows = []
        for name, info in KRANKENHAEUSER.items():
            kh_rows.append({
                "Krankenhaus": name,
                "Beschreibung": info.get("beschreibung", ""),
                "Normalbetten": info["normal"],
                "Intensivbetten": info["intensiv"],
                "Spezialisierungen": ", ".join(info["spezialisierungen"]),
                "A2A-Latenz (ms)": info["latenz_ms"],
            })
        df_alle = pd.DataFrame(kh_rows)

        def _kh_farbe(row: pd.Series) -> list[str]:
            intensiv = row["Intensivbetten"]
            if intensiv >= 5:
                bg = "#fde8e8"  # rot-getönt = hohe Intensivkapazität
            elif intensiv >= 2:
                bg = "#fef9e7"
            else:
                bg = "#eafaf1"
            return [f"background-color: {bg}"] * len(row)

        st.dataframe(
            df_alle.style.apply(_kh_farbe, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Agent-Logs ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📜 Agent-Logs")

    STATUS_FARBE = {"OK": "🟢", "FALLBACK": "🔴"}
    df_logs = lade_agent_logs(selected_id)

    if df_logs.empty:
        st.info("Noch keine Logs vorhanden (Workflow wurde noch nicht ausgeführt).")
    else:
        import json
        for _, log in df_logs.iterrows():
            icon = STATUS_FARBE.get(log["Status"], "⚪")
            ts = log["Zeitstempel"].strftime("%H:%M:%S") if log["Zeitstempel"] else ""
            with st.expander(f"{icon} {log['Agent']}  —  {ts}  [{log['Status']}]"):
                # Tool-Calls
                try:
                    tool_calls = (
                        json.loads(log["Tool-Calls"])
                        if isinstance(log["Tool-Calls"], str)
                        else log["Tool-Calls"]
                    ) or []
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []

                if tool_calls:
                    st.markdown("**Tool-Calls:**")
                    for tc in tool_calls:
                        with st.container(border=True):
                            st.markdown(f"`{tc.get('tool', '?')}`")
                            args = tc.get("args", {})
                            if args:
                                st.json(args)
                else:
                    st.caption("Keine Tool-Calls.")

                # KI-Antwort
                if log["Antwort"]:
                    st.markdown("**KI-Antwort:**")
                    st.markdown(
                        f"<div style='background:#f8f9fa;padding:10px;border-radius:6px;"
                        f"font-size:0.9em;white-space:pre-wrap'>{log['Antwort']}</div>",
                        unsafe_allow_html=True,
                    )
