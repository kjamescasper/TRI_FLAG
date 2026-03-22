"""
streamlit_app.py — TRI_FLAG Week 9

Interactive triage UI with:
  - Single-molecule tab: SMILES input, decision badge, score table, descriptor
    table, PAINS alert, rationale
  - Batch upload tab: CSV → run all → download results (with reward column)
  - Generation Analytics tab: per-generation stats and oracle dashboard plots
  - AI reviewer: copy-to-clipboard prompt → paste into Claude.ai Pro
  - Sidebar: session history table loaded from JSONL run records

Run from triage_agent/ directory:
    streamlit run streamlit_app.py

Environment:
    TRIFLAG_RUNS_FILE  — optional override for JSONL path (default: runs/triage_runs.jsonl)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
try:
    from agent.agent_state import AgentState
    from agent.decision import DecisionType
    from agent.triage_agent import TriageAgent
    from policies.policy_engine import PolicyEngine
    from policies.thresholds import (
        DEFAULT_SA_THRESHOLDS,
        LEAD_OPTIMIZATION_THRESHOLDS,
        NATURAL_PRODUCT_THRESHOLDS,
        FRAGMENT_SCREENING_THRESHOLDS,
        SAScoreThresholds,
    )
    from reporting.rationale_builder import RationaleBuilder, format_text, format_dict
    from reporting.run_record import (
        AIReviewRecord,
        RunRecord,
        RunRecordBuilder,
        save,
        load_as_dicts,
    )
    from tools.validity_tool import ValidityTool
    from tools.sa_score_tool import SAScoreTool
    from tools.similarity_tool import SimilarityTool

    try:
        from tools.descriptor_tool import DescriptorTool
        _DESCRIPTOR_TOOL_OK = True
    except ImportError:
        _DESCRIPTOR_TOOL_OK = False

    try:
        from tools.pains_tool import PAINSTool
        _PAINS_TOOL_OK = True
    except ImportError:
        _PAINS_TOOL_OK = False

    _PIPELINE_OK = True
except ImportError as _e:
    _PIPELINE_OK = False
    _IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_RUNS_FILE      = os.environ.get("TRIFLAG_RUNS_FILE", os.path.join(_HERE, "runs", "triage_runs.jsonl"))
_BATCH_RUNS_FILE = os.path.join(_HERE, "runs", "batch_streamlit.jsonl")
_DB_PATH        = os.path.join(_HERE, "runs", "triflag.db")

_THRESHOLD_PRESETS = {
    "Default (pass<6, flag<7)":          "default",
    "Lead Optimization (pass<5, flag<6)": "lead_opt",
    "Natural Products (pass<7, flag<9)":  "nat_prod",
    "Fragment Screening (pass<3.5, flag<5)": "fragment",
}

_AI_REVIEWER_PROMPT = """\
You are a medicinal chemistry triage reviewer. A rule-based system has
evaluated a molecule and produced the following assessment. Your job is to
independently reason about whether the decision is appropriate.

--- MOLECULE ---
SMILES: {smiles}
Molecule ID: {molecule_id}

--- RULE-BASED SCORES ---
Validity: {is_valid} ({num_atoms} atoms, {num_bonds} bonds)
SA Score: {sa_score} — {synthesizability_category} ({sa_decision})
Nearest Neighbor Tanimoto: {nn_tanimoto} vs {nn_id} ({nn_source})
Similarity Decision: {similarity_decision}

--- RULE-BASED DECISION ---
Final Decision: {decision_type}
Rationale: {rationale}

--- YOUR TASK ---
1. Would you have made the same decision (PASS/FLAG/DISCARD)? State clearly.
2. Explain your reasoning in 2-4 sentences referencing specific scores.
3. Note any nuance the rule-based system may have missed.
4. If you disagree, state what you would recommend and why.

Respond in this exact JSON format (no markdown, no backticks, only the JSON object):
{{
  "ai_decision": "PASS",
  "agrees_with_rules": true,
  "reasoning": "...",
  "nuance": "...",
  "confidence": "high"
}}
"""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TRI_FLAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0f1117; color: #d0d5dd; }

h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    color: #f0f2f5 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0 !important;
}
h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #c5cad3 !important;
}
h4 {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #4b5563 !important;
    margin-bottom: 8px !important;
    margin-top: 20px !important;
}

[data-testid="stSidebar"] {
    background-color: #0a0d14 !important;
    border-right: 1px solid #1a2035;
}

[data-testid="stMetric"] {
    background: #141824;
    border: 1px solid #1e2330;
    border-radius: 4px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 0.68rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #f0f2f5 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #141824 !important;
    border: 1px solid #1e2330 !important;
    color: #d0d5dd !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem;
    border-radius: 4px;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.1) !important;
}

.stButton > button {
    background-color: #141824 !important;
    border: 1px solid #1e2330 !important;
    color: #c5cad3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem;
    font-weight: 500;
    border-radius: 4px;
}
.stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #f0f2f5 !important;
}
.stButton > button[kind="primary"] {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
}

.stSelectbox > div > div {
    background-color: #141824 !important;
    border: 1px solid #1e2330 !important;
    color: #d0d5dd !important;
    font-size: 0.84rem;
    border-radius: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #0f1117;
    border-bottom: 1px solid #1e2330;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 10px 22px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #f0f2f5 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

.streamlit-expanderHeader {
    background-color: #141824 !important;
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid #1e2330;
    border-radius: 4px;
}

[data-testid="stDataFrame"] { border: 1px solid #1e2330; border-radius: 4px; overflow: hidden; }

[data-testid="stFileUploader"] {
    background-color: #141824;
    border: 1px dashed #1e2330;
    border-radius: 4px;
}

.stSpinner > div { border-top-color: #3b82f6 !important; }

/* Decision badge */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 0.8rem;
    padding: 4px 14px;
    border-radius: 3px;
    letter-spacing: 0.1em;
}
.badge-pass    { background:#052e16; border:1px solid #166534; color:#4ade80; }
.badge-flag    { background:#2d1a00; border:1px solid #92400e; color:#fbbf24; }
.badge-discard { background:#1f0707; border:1px solid #7f1d1d; color:#f87171; }
.badge-error   { background:#141824; border:1px solid #374151; color:#6b7280; }

/* Status blocks */
.status-ok    { background:#052e16; border:1px solid #166534; border-radius:4px; padding:7px 12px; font-size:0.8rem; color:#4ade80; }
.status-warn  { background:#2d1a00; border:1px solid #92400e; border-radius:4px; padding:7px 12px; font-size:0.8rem; color:#fbbf24; }
.status-alert { background:#1f0707; border:1px solid #7f1d1d; border-radius:4px; padding:10px 14px; font-size:0.8rem; color:#f87171; }

/* Mono block */
.mono-block {
    background:#141824;
    border:1px solid #1e2330;
    border-left:3px solid #1e2330;
    border-radius:4px;
    padding:14px 16px;
    font-family:'JetBrains Mono', monospace;
    font-size:0.73rem;
    color:#6b7280;
    white-space:pre-wrap;
    line-height:1.65;
    overflow-x:auto;
}

/* Sidebar history */
.hist-item {
    background:#0f1117;
    border:1px solid #1e2330;
    border-radius:3px;
    padding:7px 10px;
    margin-bottom:5px;
    font-family:'JetBrains Mono', monospace;
    font-size:0.67rem;
    line-height:1.6;
}
.hist-pass    { border-left:3px solid #166534; }
.hist-flag    { border-left:3px solid #92400e; }
.hist-discard { border-left:3px solid #7f1d1d; }

.section-rule { border:none; border-top:1px solid #1e2330; margin:16px 0 14px 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _badge(decision: str) -> str:
    cls = {"PASS": "badge-pass", "FLAG": "badge-flag", "DISCARD": "badge-discard"}.get(decision, "badge-error")
    return f'<span class="badge {cls}">{decision}</span>'


def _decision_color(decision: str) -> str:
    return {"PASS": "#4ade80", "FLAG": "#fbbf24", "DISCARD": "#f87171"}.get(decision, "#6b7280")


def _get_threshold_objects(preset_key: str):
    m = {
        "default":  (DEFAULT_SA_THRESHOLDS,         0.90),
        "lead_opt": (LEAD_OPTIMIZATION_THRESHOLDS,  0.90),
        "nat_prod": (NATURAL_PRODUCT_THRESHOLDS,    0.90),
        "fragment": (FRAGMENT_SCREENING_THRESHOLDS, 0.90),
    }
    return m.get(preset_key, m["default"])


def _build_agent(sa_thresh, sim_flag_threshold: float, skip_similarity: bool):
    tools = [ValidityTool()]
    if _DESCRIPTOR_TOOL_OK:
        tools.append(DescriptorTool())
    tools.append(SAScoreTool(thresholds=sa_thresh))
    if not skip_similarity:
        tools.append(SimilarityTool(flag_threshold=sim_flag_threshold))
    if _PAINS_TOOL_OK:
        tools.append(PAINSTool())
    return TriageAgent(
        tools=tools,
        policy_engine=PolicyEngine(sa_thresholds=sa_thresh),
        logger=logging.getLogger("agent.triage_agent"),
    )


def _run_triage(smiles, molecule_id, sa_thresh, sim_flag_threshold, skip_similarity):
    agent = _build_agent(sa_thresh, sim_flag_threshold, skip_similarity)
    state = agent.run(molecule_id=molecule_id, raw_input=smiles)
    explanation = RationaleBuilder().build(state)
    record = RunRecordBuilder().build(state, explanation)
    os.makedirs(os.path.dirname(_RUNS_FILE), exist_ok=True)
    save(record, _RUNS_FILE)
    return state, explanation, record


def _build_ai_prompt(record: RunRecord) -> str:
    return _AI_REVIEWER_PROMPT.format(
        smiles=record.smiles_canonical or record.smiles_input or "N/A",
        molecule_id=record.molecule_id,
        is_valid=record.is_valid,
        num_atoms=record.num_atoms or "N/A",
        num_bonds=record.num_bonds or "N/A",
        sa_score=f"{record.sa_score:.2f}" if record.sa_score is not None else "N/A",
        synthesizability_category=record.synthesizability_category or "N/A",
        sa_decision=record.sa_decision or "N/A",
        nn_tanimoto=f"{record.nn_tanimoto:.3f}" if record.nn_tanimoto is not None else "N/A",
        nn_id=record.nn_id or "none",
        nn_source=record.nn_source or "N/A",
        similarity_decision=record.similarity_decision or "N/A",
        decision_type=record.final_decision,
        rationale=record.rule_rationale,
    )


def _load_history() -> list[dict]:
    try:
        return load_as_dicts(_RUNS_FILE)
    except Exception:
        return []


def _render_sidebar_history():
    st.sidebar.markdown(
        "<div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.07em;color:#4b5563;margin:12px 0 8px 0'>Run History</div>",
        unsafe_allow_html=True,
    )
    records = _load_history()
    if not records:
        st.sidebar.markdown(
            "<p style='color:#374151;font-size:0.74rem;font-family:JetBrains Mono,monospace'>"
            "No runs recorded.</p>",
            unsafe_allow_html=True,
        )
        return

    for r in reversed(records[-30:]):
        decision = r.get("final_decision", "?")
        mol_id   = r.get("molecule_id", "?")
        smiles   = r.get("smiles_canonical") or r.get("smiles_input") or "?"
        ts       = r.get("run_timestamp", "")[:16].replace("T", " ")
        css      = {"PASS": "hist-pass", "FLAG": "hist-flag", "DISCARD": "hist-discard"}.get(decision, "")
        short    = smiles[:24] + "…" if len(smiles) > 27 else smiles
        st.sidebar.markdown(
            f'<div class="hist-item {css}">'
            f'<span style="color:{_decision_color(decision)}">{decision}</span>'
            f'<br><span style="color:#c5cad3">{mol_id[:22]}</span>'
            f'<br><span style="color:#4b5563">{short}</span>'
            f'<br><span style="color:#374151">{ts}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    total    = len(records)
    passes   = sum(1 for r in records if r.get("final_decision") == "PASS")
    flags    = sum(1 for r in records if r.get("final_decision") == "FLAG")
    discards = sum(1 for r in records if r.get("final_decision") == "DISCARD")
    st.sidebar.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#374151;"
        f"border-top:1px solid #1e2330;padding-top:8px;margin-top:4px'>"
        f"n={total} &nbsp;"
        f"<span style='color:#4ade80'>P:{passes}</span> &nbsp;"
        f"<span style='color:#fbbf24'>F:{flags}</span> &nbsp;"
        f"<span style='color:#f87171'>D:{discards}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_score_table(record: RunRecord):
    import pandas as pd

    rows = [
        {
            "Check":  "Validity",
            "Result": "Valid" if record.is_valid else "Invalid",
            "Detail": f"{record.num_atoms or '—'} atoms, {record.num_bonds or '—'} bonds",
        },
        {
            "Check":  "SA Score",
            "Result": f"{record.sa_score:.2f}" if record.sa_score is not None else "—",
            "Detail": f"{record.synthesizability_category or '—'}  ·  {record.sa_decision or '—'}",
        },
        {
            "Check":  "Similarity",
            "Result": f"{record.nn_tanimoto:.3f}" if record.nn_tanimoto is not None else "—",
            "Detail": f"Nearest: {record.nn_id or '—'} ({record.nn_source or '—'})  ·  {record.similarity_decision or '—'}",
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_descriptor_table(state):
    if state is None:
        return
    desc = state.tool_results.get("DescriptorTool")
    if desc is None:
        return
    if hasattr(desc, "data"):
        desc = desc.data or {}
    if desc.get("error_message"):
        st.warning(f"Descriptor calculation failed: {desc['error_message']}")
        return

    import pandas as pd

    mw      = desc.get("mol_weight")
    logp    = desc.get("logp")
    tpsa    = desc.get("tpsa")
    hbd     = desc.get("hbd")
    hba     = desc.get("hba")
    rot     = desc.get("rotatable_bonds")
    scaffold = desc.get("scaffold_smiles")

    violations = []
    if mw   is not None and mw   > 500: violations.append("MW > 500 Da")
    if logp is not None and logp > 5:   violations.append("logP > 5")
    if hbd  is not None and hbd  > 5:   violations.append("HBD > 5")
    if hba  is not None and hba  > 10:  violations.append("HBA > 10")

    def _v(val, dp=3, sfx=""):
        return f"{val:.{dp}f}{sfx}" if val is not None else "—"

    rows = [
        {"Property": "Molecular weight", "Value": _v(mw, 3, " Da"), "Ro5 limit": "≤ 500 Da",  "Status": "Violation" if (mw   and mw   > 500) else "OK"},
        {"Property": "logP",             "Value": _v(logp, 3),      "Ro5 limit": "≤ 5",        "Status": "Violation" if (logp and logp > 5)   else "OK"},
        {"Property": "TPSA",             "Value": _v(tpsa, 3, " Å²"), "Ro5 limit": "—",         "Status": "—"},
        {"Property": "HBD",              "Value": str(hbd) if hbd is not None else "—", "Ro5 limit": "≤ 5",  "Status": "Violation" if (hbd and hbd > 5)  else "OK"},
        {"Property": "HBA",              "Value": str(hba) if hba is not None else "—", "Ro5 limit": "≤ 10", "Status": "Violation" if (hba and hba > 10) else "OK"},
        {"Property": "Rotatable bonds",  "Value": str(rot) if rot is not None else "—", "Ro5 limit": "—",    "Status": "—"},
    ]

    st.markdown("#### Physicochemical Properties")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if scaffold:
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
            f"color:#6b7280;margin-top:6px'>"
            f"Bemis-Murcko scaffold: <span style='color:#c5cad3'>{scaffold}</span></div>",
            unsafe_allow_html=True,
        )

    if violations:
        st.markdown(
            "<div class='status-warn' style='margin-top:8px'>"
            "Lipinski Ro5 violation — " + " &nbsp;|&nbsp; ".join(violations) +
            "</div>",
            unsafe_allow_html=True,
        )


def _render_pains_section(state):
    if state is None:
        return
    pains = state.tool_results.get("PAINSTool")
    if pains is None:
        return
    if hasattr(pains, "data"):
        pains = pains.data or {}
    if pains.get("error_message"):
        st.warning(f"PAINS screen failed: {pains['error_message']}")
        return

    pains_alert   = pains.get("pains_alert", False)
    pains_matches = pains.get("pains_matches", [])

    st.markdown("#### PAINS Screen")

    if not pains_alert:
        st.markdown("<div class='status-ok'>No structural alerts detected.</div>", unsafe_allow_html=True)
    else:
        items = "".join(f"<li style='margin:2px 0'>{p}</li>" for p in pains_matches)
        st.markdown(
            f"<div class='status-alert'>"
            f"<strong>{len(pains_matches)} PAINS pattern(s) matched</strong>"
            f"<ul style='margin:7px 0 5px 0;padding-left:16px;font-size:0.76rem'>{items}</ul>"
            f"<div style='font-size:0.74rem;color:#9ca3af;margin-top:5px'>"
            f"Pan-Assay Interference Compounds produce false positives due to reactivity, "
            f"fluorescence, or aggregation. Advisory — confirm with an orthogonal assay "
            f"before advancing this scaffold.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1>TRI_FLAG"
    "<span style='font-weight:300;color:#4b5563;font-size:0.82rem;margin-left:12px'>"
    "Molecular Triage System</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#374151;font-size:0.75rem;margin-top:2px;margin-bottom:18px'>"
    "Chemical validity &nbsp;·&nbsp; Synthetic accessibility &nbsp;·&nbsp; "
    "Physicochemical properties &nbsp;·&nbsp; IP similarity &nbsp;·&nbsp; "
    "PAINS screening &nbsp;·&nbsp; Reward scoring</p>",
    unsafe_allow_html=True,
)

if not _PIPELINE_OK:
    st.error(f"Pipeline import failed: {_IMPORT_ERROR}. Run from the triage_agent/ directory.")
    st.stop()

_render_sidebar_history()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch, tab_analytics = st.tabs([
    "Single Molecule",
    "Batch Upload",
    "Generation Analytics",
])


# ============================================================
# TAB 1 — Single Molecule
# ============================================================
with tab_single:

    col_s1, col_s2 = st.columns([2, 2])
    with col_s1:
        preset_label = st.selectbox("Threshold preset", list(_THRESHOLD_PRESETS.keys()), key="preset_single")
    with col_s2:
        skip_sim = st.checkbox("Skip similarity check (offline mode)", value=False, key="skip_sim_single")

    preset_key = _THRESHOLD_PRESETS[preset_label]
    sa_thresh, sim_flag_threshold = _get_threshold_objects(preset_key)

    col_smiles, col_id = st.columns([3, 1])
    with col_smiles:
        smiles_input = st.text_input("SMILES string", placeholder="CC(=O)Oc1ccccc1C(=O)O", key="smiles_single")
    with col_id:
        mol_id_input = st.text_input("Molecule ID", value="mol_001", key="id_single")

    run_btn = st.button("Run Triage", type="primary", key="run_single")

    if "single_result" not in st.session_state:
        st.session_state.single_result = None

    if run_btn:
        if not smiles_input.strip():
            st.warning("Enter a SMILES string to continue.")
        else:
            with st.spinner("Running pipeline…"):
                try:
                    state, explanation, record = _run_triage(
                        smiles=smiles_input.strip(),
                        molecule_id=mol_id_input.strip() or "mol_001",
                        sa_thresh=sa_thresh,
                        sim_flag_threshold=sim_flag_threshold,
                        skip_similarity=skip_sim,
                    )
                    st.session_state.single_result = (state, explanation, record)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.session_state.single_result = None

    if st.session_state.single_result is not None:
        state, explanation, record = st.session_state.single_result

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        col_dec, col_sum = st.columns([1, 4])
        with col_dec:
            st.markdown(_badge(record.final_decision), unsafe_allow_html=True)
            if record.early_termination:
                st.markdown(
                    "<div style='font-size:0.71rem;color:#4b5563;margin-top:5px'>Early termination</div>",
                    unsafe_allow_html=True,
                )
        with col_sum:
            st.markdown(
                f"<p style='color:#c5cad3;font-size:0.87rem;line-height:1.55;margin-top:2px'>"
                f"{explanation.summary}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Pipeline Scores")
        _render_score_table(record)

        _render_descriptor_table(state)
        _render_pains_section(state)

        if record.flags_raised:
            st.markdown("#### Flags")
            for flag in record.flags_raised:
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"color:#fbbf24;padding:3px 0;line-height:1.5'>"
                    f"[{flag.get('source', '?')}]&nbsp; {flag.get('reason', '?')}</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("Full Rationale Report", expanded=False):
            try:
                rationale_text = format_text(explanation)
            except Exception:
                rationale_text = record.rule_rationale
            st.markdown(f'<div class="mono-block">{rationale_text}</div>', unsafe_allow_html=True)

        with st.expander("Raw Run Record (JSON)", expanded=False):
            st.json({k: v for k, v in record.__dict__.items() if not k.startswith("_") and k != "ai_review"})

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("#### AI Reviewer")
        st.markdown(
            "<p style='color:#6b7280;font-size:0.79rem;margin-bottom:8px'>"
            "Copy this prompt and paste it into "
            "<a href='https://claude.ai' target='_blank' style='color:#3b82f6;text-decoration:none'>claude.ai</a>"
            " for an independent medicinal chemistry assessment.</p>",
            unsafe_allow_html=True,
        )
        st.text_area("Reviewer prompt", value=_build_ai_prompt(record), height=240, key="ai_prompt_box")


# ============================================================
# TAB 2 — Batch Upload
# ============================================================
with tab_batch:
    st.markdown("#### Batch Processing")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.8rem'>"
        "Upload a CSV with columns <code>molecule_id</code> and <code>smiles</code>. "
        "Results download as CSV or JSON Lines.</p>",
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        preset_label_b = st.selectbox("Threshold preset", list(_THRESHOLD_PRESETS.keys()), key="preset_batch")
    with col_b2:
        skip_sim_b = st.checkbox("Skip similarity check", value=False, key="skip_sim_batch")

    preset_key_b = _THRESHOLD_PRESETS[preset_label_b]
    sa_thresh_b, sim_flag_threshold_b = _get_threshold_objects(preset_key_b)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None

    if uploaded_file is not None:
        try:
            import pandas as pd
            df_input = pd.read_csv(uploaded_file)
            missing = {"molecule_id", "smiles"} - set(df_input.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.markdown(f"{len(df_input)} molecules loaded.")
                st.dataframe(df_input.head(5), use_container_width=True, hide_index=True)

                if st.button("Run Batch Triage", type="primary", key="run_batch"):
                    results = []
                    progress_bar = st.progress(0.0)
                    status_text  = st.empty()
                    os.makedirs(os.path.dirname(_BATCH_RUNS_FILE), exist_ok=True)

                    for i, row in df_input.iterrows():
                        mol_id = str(row["molecule_id"])
                        smiles = str(row["smiles"])
                        status_text.markdown(
                            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.76rem;"
                            f"color:#6b7280'>[{i+1}/{len(df_input)}] {mol_id}</span>",
                            unsafe_allow_html=True,
                        )
                        try:
                            state, explanation, record = _run_triage(
                                smiles=smiles,
                                molecule_id=mol_id,
                                sa_thresh=sa_thresh_b,
                                sim_flag_threshold=sim_flag_threshold_b,
                                skip_similarity=skip_sim_b,
                            )
                            save(record, _BATCH_RUNS_FILE)
                            results.append({
                                "molecule_id": mol_id,
                                "smiles":      smiles,
                                "decision":    record.final_decision,
                                "sa_score":    record.sa_score,
                                "nn_tanimoto": record.nn_tanimoto,
                                "reward":      record.reward,
                                "s_sa":        record.s_sa,
                                "s_nov":       record.s_nov,
                                "s_qed":       record.s_qed,
                                "summary":     record.explanation_summary,
                                "error":       None,
                            })
                        except Exception as e:
                            results.append({
                                "molecule_id": mol_id, "smiles": smiles,
                                "decision": "ERROR", "sa_score": None,
                                "nn_tanimoto": None, "reward": None,
                                "s_sa": None, "s_nov": None, "s_qed": None,
                                "summary": str(e), "error": str(e),
                            })
                        progress_bar.progress((i + 1) / len(df_input))

                    status_text.empty()
                    progress_bar.empty()
                    st.session_state.batch_results = results

        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if st.session_state.batch_results:
        import pandas as pd
        results  = st.session_state.batch_results
        df_res   = pd.DataFrame(results)
        total    = len(results)
        passes   = sum(1 for r in results if r["decision"] == "PASS")
        flags    = sum(1 for r in results if r["decision"] == "FLAG")
        discards = sum(1 for r in results if r["decision"] == "DISCARD")
        errors   = sum(1 for r in results if r["decision"] == "ERROR")

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("#### Results")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total",   total)
        m2.metric("Pass",    passes)
        m3.metric("Flag",    flags)
        m4.metric("Discard", discards)
        m5.metric("Error",   errors)

        st.dataframe(df_res, use_container_width=True, hide_index=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("Download CSV",  df_res.to_csv(index=False),
                file_name=f"triflag_batch_{ts}.csv",  mime="text/csv", key="dl_csv")
        with col_dl2:
            st.download_button("Download JSONL", "\n".join(json.dumps(r) for r in results),
                file_name=f"triflag_batch_{ts}.jsonl", mime="application/x-ndjson", key="dl_jsonl")


# ============================================================
# TAB 3 — Generation Analytics
# ============================================================
with tab_analytics:
    st.markdown("#### Generation Analytics")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.8rem'>"
        "Per-generation reward statistics and oracle dashboard plots. "
        "Data is populated automatically during ACEGEN training runs (Week 10).</p>",
        unsafe_allow_html=True,
    )

    if not os.path.exists(_DB_PATH):
        st.info(f"No database found at {_DB_PATH}. Run at least one molecule to initialise it.")
    else:
        try:
            from analysis.oracle_dashboard import generate_dashboard
            import pandas as pd

            with st.spinner("Querying database…"):
                summaries = generate_dashboard(db_path=_DB_PATH)

            if not summaries:
                st.info("No generation data recorded yet.")
            else:
                st.markdown("#### Per-Generation Summary")
                display_cols = [
                    "batch_id", "count", "mean_reward", "std_reward",
                    "pass_rate", "flag_rate", "discard_rate",
                    "unique_scaffolds", "convergence_warning",
                ]
                df_summary = pd.DataFrame([{k: s[k] for k in display_cols if k in s} for s in summaries])
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

                plot_dir = os.path.join(_HERE, "analysis", "output", "oracle")
                plot_specs = [
                    ("mean_reward_trajectory.png", "Mean Reward Trajectory"),
                    ("reward_distribution.png",    "Reward Distribution"),
                    ("decision_rates.png",          "Decision Rates"),
                    ("scaffold_diversity.png",      "Scaffold Diversity"),
                ]
                plots_found = [(f, t) for f, t in plot_specs if os.path.exists(os.path.join(plot_dir, f))]

                if plots_found:
                    st.markdown("#### Oracle Dashboard")
                    col_l, col_r = st.columns(2)
                    for idx, (fname, title) in enumerate(plots_found):
                        with (col_l if idx % 2 == 0 else col_r):
                            st.markdown(
                                f"<div style='font-size:0.76rem;color:#6b7280;margin-bottom:4px'>{title}</div>",
                                unsafe_allow_html=True,
                            )
                            st.image(os.path.join(plot_dir, fname))
                else:
                    st.info("Dashboard plots will appear here after the first ACEGEN run (Week 10).")

        except ImportError:
            try:
                import sqlite3, pandas as pd
                conn   = sqlite3.connect(_DB_PATH)
                df_raw = pd.read_sql_query(
                    """SELECT batch_id, COUNT(*) AS count,
                       ROUND(AVG(reward),4) AS mean_reward,
                       SUM(CASE WHEN final_decision='PASS'    THEN 1 ELSE 0 END) AS pass_count,
                       SUM(CASE WHEN final_decision='FLAG'    THEN 1 ELSE 0 END) AS flag_count,
                       SUM(CASE WHEN final_decision='DISCARD' THEN 1 ELSE 0 END) AS discard_count
                       FROM triage_runs GROUP BY batch_id ORDER BY batch_id""",
                    conn,
                )
                conn.close()
                st.dataframe(df_raw, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not query database: {e}")

        except Exception as e:
            st.error(f"Analytics error: {e}")