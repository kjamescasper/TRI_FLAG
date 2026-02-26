"""
streamlit_app.py — TRI_FLAG Week 7

Interactive triage UI with:
  - Single-molecule tab: SMILES input, decision badge, score table, rationale
  - Batch upload tab: CSV → run all → download results
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
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — must be before any triage_agent imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Triage pipeline imports
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
        AIReviewRecord,  # kept for RunRecord type compatibility
        RunRecord,
        RunRecordBuilder,
        save,
        load_as_dicts,
    )
    from tools.validity_tool import ValidityTool
    from tools.sa_score_tool import SAScoreTool
    from tools.similarity_tool import SimilarityTool

    _PIPELINE_OK = True
except ImportError as _e:
    _PIPELINE_OK = False
    _IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_RUNS_FILE = os.environ.get("TRIFLAG_RUNS_FILE", os.path.join(_HERE, "runs", "triage_runs.jsonl"))
_BATCH_RUNS_FILE = os.path.join(_HERE, "runs", "batch_streamlit.jsonl")

_THRESHOLD_PRESETS = {
    "Default (pass<6, flag<7)": "default",
    "Lead Optimization (pass<5, flag<6)": "lead_opt",
    "Natural Products (pass<7, flag<9)": "nat_prod",
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
    page_title="TRI_FLAG — Molecule Triage",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — industrial/utilitarian dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* App background */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}

h1 { color: #e6edf3; font-size: 1.6rem !important; }
h2 { color: #e6edf3; font-size: 1.1rem !important; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #8b949e; font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown { color: #8b949e; }

/* Metric tiles */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem !important; }

/* Input boxes */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem;
    border-radius: 6px;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* Buttons */
.stButton > button {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem;
    border-radius: 6px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background-color: #30363d !important;
    border-color: #58a6ff !important;
    color: #e6edf3 !important;
}
.stButton > button[kind="primary"] {
    background-color: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #388bfd !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace;
}

/* Code blocks */
code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: #161b22 !important;
    border: 1px solid #21262d;
    border-radius: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d1117;
    border-bottom: 1px solid #21262d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    padding: 8px 20px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background-color: transparent !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #161b22 !important;
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    border: 1px solid #21262d;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; overflow: hidden; }

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    border: 1px dashed #30363d;
    border-radius: 6px;
}

/* Progress / spinner */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* Decision badge styles */
.badge-pass {
    display: inline-block;
    background-color: #0d4429;
    border: 1px solid #1a7f37;
    color: #3fb950;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 8px 24px;
    border-radius: 6px;
    letter-spacing: 0.12em;
}
.badge-flag {
    display: inline-block;
    background-color: #3d2b00;
    border: 1px solid #9e6a03;
    color: #d29922;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 8px 24px;
    border-radius: 6px;
    letter-spacing: 0.12em;
}
.badge-discard {
    display: inline-block;
    background-color: #3d0d0d;
    border: 1px solid #8b1a1a;
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 8px 24px;
    border-radius: 6px;
    letter-spacing: 0.12em;
}
.badge-error {
    display: inline-block;
    background-color: #1c1c1c;
    border: 1px solid #484f58;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 8px 24px;
    border-radius: 6px;
    letter-spacing: 0.12em;
}

/* AI review box */
.ai-review-box {
    background-color: #0c2d6b;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 12px;
}
.ai-review-box h4 {
    color: #58a6ff !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.ai-review-box p {
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 4px 0;
}

/* Rationale pre block */
.rationale-block {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #30363d;
    border-radius: 4px;
    padding: 14px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    white-space: pre-wrap;
    line-height: 1.6;
    overflow-x: auto;
}

/* Sidebar history item */
.hist-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}
.hist-pass { border-left: 3px solid #3fb950; }
.hist-flag { border-left: 3px solid #d29922; }
.hist-discard { border-left: 3px solid #f85149; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _decision_badge(decision: str) -> str:
    icons = {"PASS": "🟢", "FLAG": "🟡", "DISCARD": "🔴"}
    classes = {"PASS": "badge-pass", "FLAG": "badge-flag", "DISCARD": "badge-discard"}
    icon = icons.get(decision, "⚪")
    cls = classes.get(decision, "badge-error")
    return f'<span class="{cls}">{icon} {decision}</span>'


def _decision_color(decision: str) -> str:
    return {"PASS": "#3fb950", "FLAG": "#d29922", "DISCARD": "#f85149"}.get(decision, "#8b949e")


def _get_threshold_objects(preset_key: str):
    """Returns (sa_thresholds, sim_flag_threshold_float)."""
    preset_map = {
        "default":  (DEFAULT_SA_THRESHOLDS,          0.85),
        "lead_opt": (LEAD_OPTIMIZATION_THRESHOLDS,   0.85),
        "nat_prod": (NATURAL_PRODUCT_THRESHOLDS,     0.85),
        "fragment": (FRAGMENT_SCREENING_THRESHOLDS,  0.85),
    }
    return preset_map.get(preset_key, preset_map["default"])


def _build_agent(sa_thresh, sim_flag_threshold: float, skip_similarity: bool):
    tools = [ValidityTool(), SAScoreTool(thresholds=sa_thresh)]
    if not skip_similarity:
        tools.append(SimilarityTool(flag_threshold=sim_flag_threshold))
    policy = PolicyEngine(sa_thresholds=sa_thresh)
    agent = TriageAgent(
        tools=tools,
        policy_engine=policy,
        logger=logging.getLogger("agent.triage_agent"),
    )
    return agent


def _run_triage(smiles: str, molecule_id: str, sa_thresh, sim_flag_threshold: float, skip_similarity: bool):
    """Run the full triage pipeline and return (state, explanation, record)."""
    agent = _build_agent(sa_thresh, sim_flag_threshold, skip_similarity)
    state = agent.run(molecule_id=molecule_id, raw_input=smiles)
    builder = RationaleBuilder()
    explanation = builder.build(state)
    record = RunRecordBuilder().build(state, explanation)
    os.makedirs(os.path.dirname(_RUNS_FILE), exist_ok=True)
    save(record, _RUNS_FILE)
    return state, explanation, record


def _build_ai_prompt(record: RunRecord) -> str:
    """Build the reviewer prompt from a RunRecord, ready to paste into Claude.ai."""
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
    """Load all run records from the JSONL file."""
    try:
        return load_as_dicts(_RUNS_FILE)
    except Exception:
        return []


def _render_sidebar_history():
    """Render session history in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Run History")
    records = _load_history()
    if not records:
        st.sidebar.markdown(
            "<p style='color:#484f58;font-size:0.78rem;font-family:IBM Plex Mono,monospace'>"
            "No runs yet.</p>",
            unsafe_allow_html=True,
        )
        return

    # Most recent first
    for r in reversed(records[-30:]):
        decision = r.get("final_decision", "?")
        mol_id = r.get("molecule_id", "?")
        smiles = r.get("smiles_canonical") or r.get("smiles_input") or "?"
        ts = r.get("run_timestamp", "")[:16].replace("T", " ")
        css_class = {"PASS": "hist-pass", "FLAG": "hist-flag", "DISCARD": "hist-discard"}.get(decision, "")
        icon = {"PASS": "🟢", "FLAG": "🟡", "DISCARD": "🔴"}.get(decision, "⚪")
        smiles_short = smiles[:22] + "…" if len(smiles) > 25 else smiles
        st.sidebar.markdown(
            f'<div class="hist-item {css_class}">'
            f'<span style="color:{_decision_color(decision)}">{icon} {decision}</span>'
            f'<br><span style="color:#e6edf3">{mol_id[:20]}</span>'
            f'<br><span style="color:#484f58">{smiles_short}</span>'
            f'<br><span style="color:#30363d">{ts}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Summary counts
    total = len(records)
    passes = sum(1 for r in records if r.get("final_decision") == "PASS")
    flags = sum(1 for r in records if r.get("final_decision") == "FLAG")
    discards = sum(1 for r in records if r.get("final_decision") == "DISCARD")
    st.sidebar.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#484f58;"
        f"border-top:1px solid #21262d;padding-top:8px;margin-top:4px'>"
        f"Total: {total} &nbsp;|&nbsp; "
        f"<span style='color:#3fb950'>P:{passes}</span> "
        f"<span style='color:#d29922'>F:{flags}</span> "
        f"<span style='color:#f85149'>D:{discards}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_score_table(record: RunRecord):
    """Render the 3-column score summary."""
    import pandas as pd

    rows = []

    # Validity
    v_status = "✅ VALID" if record.is_valid else "❌ INVALID"
    rows.append({
        "Check": "Validity",
        "Result": v_status,
        "Detail": f"{record.num_atoms or '?'} atoms, {record.num_bonds or '?'} bonds",
    })

    # SA Score
    sa_val = f"{record.sa_score:.2f}" if record.sa_score is not None else "N/A"
    sa_cat = record.synthesizability_category or "N/A"
    sa_dec = record.sa_decision or "N/A"
    sa_icons = {"PASS": "✅", "FLAG": "⚠️", "DISCARD": "❌", "ERROR": "💀"}
    rows.append({
        "Check": "SA Score",
        "Result": f"{sa_icons.get(sa_dec, '—')} {sa_val}",
        "Detail": f"{sa_cat} → {sa_dec}",
    })

    # Similarity
    nn_t = f"{record.nn_tanimoto:.3f}" if record.nn_tanimoto is not None else "N/A"
    sim_dec = record.similarity_decision or "N/A"
    nn_id = record.nn_id or "—"
    sim_icons = {"PASS": "✅", "FLAG": "⚠️", "ERROR": "⚠️"}
    rows.append({
        "Check": "Similarity / IP",
        "Result": f"{sim_icons.get(sim_dec, '—')} {nn_t}",
        "Detail": f"Nearest: {nn_id} ({record.nn_source or '—'}) → {sim_dec}",
    })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)



# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1>⚗️ TRI_FLAG <span style='color:#484f58;font-size:0.9rem;font-weight:400'>"
    "— Explainable Molecule Triage</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#484f58;font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
    "margin-top:-8px;margin-bottom:16px'>"
    "Chemical validity · SA score · IP similarity · AI review</p>",
    unsafe_allow_html=True,
)

if not _PIPELINE_OK:
    st.error(f"⛔ Pipeline import failed: `{_IMPORT_ERROR}`. Make sure you are running from `triage_agent/`.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar history
# ---------------------------------------------------------------------------
_render_sidebar_history()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔬 Single Molecule", "📦 Batch Upload"])

# ============================================================
# TAB 1: Single Molecule
# ============================================================
with tab_single:

    # -- Settings row
    col_settings1, col_settings2, col_settings3 = st.columns([2, 2, 1])
    with col_settings1:
        preset_label = st.selectbox(
            "Threshold preset",
            list(_THRESHOLD_PRESETS.keys()),
            key="preset_single",
        )
    with col_settings2:
        skip_sim = st.checkbox("Skip similarity check (faster, offline)", value=False, key="skip_sim_single")
    with col_settings3:
        st.markdown("<br>", unsafe_allow_html=True)

    preset_key = _THRESHOLD_PRESETS[preset_label]
    sa_thresh, sim_flag_threshold = _get_threshold_objects(preset_key)

    # -- Input row
    col_smiles, col_id = st.columns([3, 1])
    with col_smiles:
        smiles_input = st.text_input(
            "SMILES",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
            key="smiles_single",
        )
    with col_id:
        mol_id_input = st.text_input(
            "Molecule ID",
            value="mol_001",
            key="id_single",
        )

    run_btn = st.button("▶ Run Triage", type="primary", key="run_single")

    # -- Session state for single run
    if "single_result" not in st.session_state:
        st.session_state.single_result = None  # (state, explanation, record)

    if run_btn:
        if not smiles_input.strip():
            st.warning("Please enter a SMILES string.")
        else:
            with st.spinner("Running triage pipeline…"):
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

    # -- Results display
    if st.session_state.single_result is not None:
        state, explanation, record = st.session_state.single_result

        st.markdown("---")

        # Decision badge
        col_badge, col_summary = st.columns([1, 3])
        with col_badge:
            st.markdown(
                _decision_badge(record.final_decision),
                unsafe_allow_html=True,
            )
            if record.early_termination:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
                    f"color:#8b949e;margin-top:6px'>⚡ Early termination</div>",
                    unsafe_allow_html=True,
                )
        with col_summary:
            st.markdown(
                f"<p style='color:#c9d1d9;font-size:0.95rem;line-height:1.5;"
                f"margin-top:4px'>{explanation.summary}</p>",
                unsafe_allow_html=True,
            )

        # Score table
        st.markdown("#### Score Summary")
        _render_score_table(record)

        # Flags
        if record.flags_raised:
            st.markdown("#### ⚑ Flags Raised")
            for flag in record.flags_raised:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;"
                    f"color:#d29922;padding:4px 0'>"
                    f"[{flag.get('source', '?')}] {flag.get('reason', '?')}</div>",
                    unsafe_allow_html=True,
                )

        # Rationale report (collapsible)
        with st.expander("📄 Full Rationale Report", expanded=False):
            try:
                rationale_text = format_text(explanation)
            except Exception:
                rationale_text = record.rule_rationale
            st.markdown(
                f'<div class="rationale-block">{rationale_text}</div>',
                unsafe_allow_html=True,
            )

        # Raw run record (collapsible)
        with st.expander("🗄 Raw Run Record (JSON)", expanded=False):
            st.json({
                k: v for k, v in record.__dict__.items()
                if not k.startswith("_") and k != "ai_review"
            })

        # AI reviewer section
        st.markdown("---")
        st.markdown("#### 🤖 AI Reviewer")
        st.markdown(
            "<p style='color:#8b949e;font-family:IBM Plex Mono,monospace;font-size:0.78rem'>"
            "Copy the prompt below and paste it into "
            "<a href='https://claude.ai' target='_blank' style='color:#58a6ff'>claude.ai</a> "
            "for an independent medicinal chemistry review.</p>",
            unsafe_allow_html=True,
        )
        ai_prompt = _build_ai_prompt(record)
        st.text_area(
            "Reviewer prompt (copy → paste into Claude.ai)",
            value=ai_prompt,
            height=260,
            key="ai_prompt_box",
        )


# ============================================================
# TAB 2: Batch Upload
# ============================================================
with tab_batch:
    st.markdown("### Batch Processing")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem'>"
        "Upload a CSV with columns: <code>molecule_id</code>, <code>smiles</code>. "
        "Optional: <code>name</code>. "
        "Results will be available for download as JSON Lines.</p>",
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        preset_label_b = st.selectbox(
            "Threshold preset",
            list(_THRESHOLD_PRESETS.keys()),
            key="preset_batch",
        )
    with col_b2:
        skip_sim_b = st.checkbox("Skip similarity check", value=False, key="skip_sim_batch")

    preset_key_b = _THRESHOLD_PRESETS[preset_label_b]
    sa_thresh_b, sim_flag_threshold_b = _get_threshold_objects(preset_key_b)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_upload",
    )

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None

    if uploaded_file is not None:
        try:
            import pandas as pd
            df_input = pd.read_csv(uploaded_file)
            required_cols = {"molecule_id", "smiles"}
            missing = required_cols - set(df_input.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.markdown(f"**{len(df_input)} molecules loaded.**")
                st.dataframe(df_input.head(5), use_container_width=True, hide_index=True)

                run_batch_btn = st.button("▶ Run Batch Triage", type="primary", key="run_batch")

                if run_batch_btn:
                    results = []
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    os.makedirs(os.path.dirname(_BATCH_RUNS_FILE), exist_ok=True)

                    for i, row in df_input.iterrows():
                        mol_id = str(row["molecule_id"])
                        smiles = str(row["smiles"])
                        status_text.markdown(
                            f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;"
                            f"color:#8b949e'>Processing [{i+1}/{len(df_input)}]: {mol_id}</span>",
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
                                "smiles": smiles,
                                "decision": record.final_decision,
                                "sa_score": record.sa_score,
                                "nn_tanimoto": record.nn_tanimoto,
                                "summary": record.explanation_summary,
                                "error": None,
                            })
                        except Exception as e:
                            results.append({
                                "molecule_id": mol_id,
                                "smiles": smiles,
                                "decision": "ERROR",
                                "sa_score": None,
                                "nn_tanimoto": None,
                                "summary": str(e),
                                "error": str(e),
                            })
                        progress_bar.progress((i + 1) / len(df_input))

                    status_text.empty()
                    progress_bar.empty()
                    st.session_state.batch_results = results

        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # Batch results display
    if st.session_state.batch_results:
        import pandas as pd
        results = st.session_state.batch_results
        df_res = pd.DataFrame(results)

        # Summary metrics
        total = len(results)
        passes = sum(1 for r in results if r["decision"] == "PASS")
        flags = sum(1 for r in results if r["decision"] == "FLAG")
        discards = sum(1 for r in results if r["decision"] == "DISCARD")
        errors = sum(1 for r in results if r["decision"] == "ERROR")

        st.markdown("---")
        st.markdown("### Batch Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total", total)
        m2.metric("🟢 PASS", passes)
        m3.metric("🟡 FLAG", flags)
        m4.metric("🔴 DISCARD", discards)
        m5.metric("⚪ ERROR", errors)

        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_out = df_res.to_csv(index=False)
            st.download_button(
                label="⬇ Download CSV",
                data=csv_out,
                file_name=f"triflag_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_csv",
            )
        with col_dl2:
            jsonl_out = "\n".join(json.dumps(r) for r in results)
            st.download_button(
                label="⬇ Download JSONL",
                data=jsonl_out,
                file_name=f"triflag_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl",
                mime="application/x-ndjson",
                key="dl_jsonl",
            )