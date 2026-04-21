# TRI_FLAG

**Agentic molecular triage and generative pipeline for BACE1-targeted drug discovery**

*University of Alabama at Birmingham | OSDD2 / UAB SPARC*
*Kieran Casper & Caylee Barnes | Advisor: Dr. Jake Chen | April 2026*

TRI_FLAG wraps [ACEGEN](https://github.com/Acellera/acegen-open) reinforcement learning and [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) binding prediction inside a five-tool molecular triage pipeline that scores, filters, and stores every generated molecule in a live SQLite database. The reward function enforces synthesis feasibility, IP novelty, drug-likeness, and BACE1 binding affinity simultaneously. Across 8 generation runs the pipeline has scored **34,915 molecules**.

**Oracle APEX public URL:** https://oracleapex.com/ords/r/info_triflag/tri-flag/home

*More information is available through Notion upon request.*

```
ACEGEN generates SMILES
    → triflag_score() evaluates each one
        → R = S_sa × S_nov × S_qed × S_act
            → reward updates RL policy
                → molecule stored in SQLite with full provenance
```

---

## Database Access

The complete TRI_FLAG molecule database (34,915 molecules, 8 generations) is available three ways:

| Method | Location | What's there |
|--------|----------|--------------|
| **Browse online (Oracle APEX)** | https://oracleapex.com/ords/r/info_triflag/tri-flag/home | Searchable, filterable web interface — generations summary, top candidates, full molecule table. No login required. |
| **Download full database (ZIP)** | [`exports/triflag_data.zip`](exports/triflag_data.zip) in this repo | `triflag.db` (SQLite, all 34,915 molecules) + 4 CSVs + manifest. Download and open with DB Browser for SQLite or any SQLite client. |
| **Programmatic access** | Clone repo → `triage_agent/runs/triflag.db` after unzipping | `from database.db import DatabaseManager` — see Usage section |

> **Why two access methods?** Oracle APEX free tier has a row/storage limit — the full 34,915-molecule database exceeds it. The APEX instance hosts curated views (generation summaries, top candidates, key physicochemical properties). The complete database with all columns and all rows is in `exports/triflag_data.zip`.

---

## Results

| Gen | Molecules | Mean Reward | Max Reward | Pass% | Notes |
|-----|-----------|-------------|------------|-------|-------|
| gen_000 | 4,172 | 0.4346* | 0.4957 | 95.4% | 3-component reward — not comparable |
| gen_001 | 4,018 | 0.1094 | 0.5337 | 93.4% | First full 4-component run |
| gen_002 | 1,530 | 0.0778 | 0.5388 | 90.3% | Truncated |
| gen_003 | 1,283 | 0.0792 | 0.5388 | 90.9% | Truncated |
| gen_004 | 2,420 | 0.0940 | 0.5388 | 90.7% | Partial |
| gen_005 | 6,403 | 0.1325 | **0.7049** | 92.8% | First full 6,400-mol run |
| gen_006 | 6,521 | 0.1314 | **0.7049** | 92.7% | Cold-start; matched Gen 5 max; first checkpoint saved |
| gen_007 | 8,551 | **0.2242** | 0.6422 | 96.0% | **First warm-start: +69% mean reward** |

*Gen 0 excluded from RL trend analysis — used 3-component reward without S_act.*

**The primary RL finding:** loading `gen_006_policy.pt` into Generation 7 produced a +69% improvement in mean reward (0.1314 → 0.2242), with 8 of the all-time top 10 candidates emerging from that single warm-start run — at scaffolds never seen in any cold-start generation.

---

## Quick Start

**Requirements:** Windows, conda, Python 3.12

```bash
# Clone with ACEGEN submodule
git clone --recurse-submodules https://github.com/kjamescasper/TRI_FLAG.git
cd TRI_FLAG

# Create environment
conda create -n triflag python=3.12
conda activate triflag

# Core dependencies
pip install rdkit requests aiohttp streamlit fastmcp

# ACEGEN (TorchRL-based)
pip install torchrl==0.8.1 tensordict==0.8.1
pip install -e acegen-open/

# DeepPurpose (optional — required for S_act scoring)
# Must set env var FIRST to avoid Windows shm.dll crash (see Known Issues)
set TRIFLAG_ENABLE_DEEPPURPOSE=1
pip install DeepPurpose

# Verify
cd triage_agent
python -m pytest tests/ -x -q
# Expected: 374 passed, 2 skipped
```

---

## Usage

**Triage a single molecule:**
```bash
cd triage_agent && conda activate triflag
python main.py --smiles "O=C(NC1CC1)c1ccc(-c2ccnc2)cc1" --molecule-id my_mol
```

**Streamlit dashboard:**
```bash
cd triage_agent && conda activate triflag
streamlit run streamlit_app.py
# → http://localhost:8501
```

**Run a new generation:**
```bash
cd triage_agent && conda activate triflag
python acegen_scripts/run_generation.py               # cold-start (auto-increments)
python acegen_scripts/run_generation.py --warm-start  # load previous checkpoint
python acegen_scripts/run_generation.py --gen 7       # rerun a specific generation
```

**Query the database directly:**
```bash
cd triage_agent && conda activate triflag
python -c "
from database.db import DatabaseManager
db = DatabaseManager()
for r in db.get_top_n_by_reward(10):
    d = dict(r)
    print(d['molecule_id'], d['reward'])
"
```

**MCP server (Claude Desktop):**
```bash
cd triage_agent && conda activate triflag
python mcp_server.py
# Add to Claude Desktop → Settings → MCP Servers
```

---

## How It Works

### Reward Function

```
R = S_sa × S_nov × S_qed × S_act
```

All four factors must be good simultaneously — a synthetically inaccessible molecule or a patent match scores near zero regardless of how well it binds BACE1.

| Factor | What it measures | Implementation |
|--------|-----------------|----------------|
| **S_sa** | Synthetic accessibility | Ertl-Schuffenhauer SA score via RDKit; sigmoid transform (midpoint 4.5, k=1.5); hard discard at SA > 7 |
| **S_nov** | IP novelty | ECFP4 Tanimoto vs ChEMBL (~2.4M) + SureChEMBL (~28M patents); FLAG at Tanimoto ≥ 0.70 |
| **S_qed** | Drug-likeness | Bickerton QED via RDKit; composite of 8 physicochemical descriptors |
| **S_act** | BACE1 binding affinity | DeepPurpose MPNN_CNN_BindingDB; pIC50 normalized to [0,1] (4.0→0.0, 10.0→1.0) |

The all-time max reward of 0.7049 is **S_act-suppressed** — a proof in `tests/test_scoring_validation.py` shows S_sa × S_nov × S_qed exceeds 0.7049 for the record molecule, implying S_act ≈ 0.797 (predicted BACE1 pIC50 ~8.8 nM).

### Triage Pipeline (Module 1)

Five tools run in sequence with early-termination: an invalid SMILES or SA > 7 halts immediately.

```
ValidityTool     → RDKit SMILES parsing + canonicalization → DISCARD if invalid
DescriptorTool   → MW, logP, TPSA, HBD, HBA, rotatable bonds, Ro5, Murcko scaffold
SAScoreTool      → Ertl-Schuffenhauer score → DISCARD if SA > 7; sigmoid → S_sa
SimilarityTool   → ECFP4 Tanimoto vs ChEMBL / SureChEMBL / PubChem → FLAG if ≥ 0.70
PAINSTool        → PAINS_A/B/C via RDKit FilterCatalog → advisory FLAG
```

Every decision includes a plain-English rationale from `RationaleBuilder`. Mean SA score across all 34,915 molecules: **2.996 / 10** — well below the discard threshold of 7, confirming the RL agent consistently produces synthetically feasible candidates.

### ACEGEN Warm-Start (Module 2)

ACEGEN calls `triflag_score(smiles: List[str]) -> List[float]` in `loop/triflag_scorer.py` as its reward oracle. ACEGEN's `logger_backend: null` saves no native checkpoints. TRI_FLAG enables warm-start via an in-memory two-point patch of `reinvent.py` before `exec()`:

- **Point 1** — overrides `ckpt_path` to load from `runs/checkpoints/gen_NNN_policy.pt` instead of the frozen prior
- **Point 2** — injects a global reference to expose actor weights post-run for saving

### DeepPurpose BACE1 Scoring (Module 3)

Model `MPNN_CNN_BindingDB` encodes molecules as message-passing neural networks and BACE1 (UniProt P56817, 501 residues) via 1D CNN, trained on BindingDB IC50 data. Pre-trained weights at `save_folder/pretrained_models/model_MPNN_CNN/model.pt`.

> **Windows:** DeepPurpose import must be fully lazy (inside `load_model()` only). A C-level process abort (`0xC0000139`, `shm.dll`) occurs on bare import and cannot be caught by Python. Set `TRIFLAG_ENABLE_DEEPPURPOSE=1` before any other imports.

---

## Top Candidates

Verified live from `runs/triflag.db`, April 20 2026. All are Lipinski Ro5 compliant, PAINS-free, S_nov = 1.000.

| Rank | Molecule ID | Reward | S_sa | S_qed | MW | TPSA (Å²) | logP | Gen |
|------|-------------|--------|------|-------|-----|-----------|------|-----|
| 1 | acegen_eaec28e1178d | 0.7049 | 0.937 | 0.943 | 318.4 | 84.0 | 3.12 | 6 |
| 2 | acegen_dc1d5ec0d55b | 0.7049 | 0.937 | 0.943 | 318.4 | 84.0 | 3.12 | 5 |
| 3 | acegen_0f2cf957e964 | 0.6422 | 0.955 | 0.913 | 299.6 | 52.9 | 2.89 | 7 |
| 4 | acegen_96889fee11c1 | 0.6269 | 0.967 | 0.902 | 326.4 | 70.0 | 3.97 | 7 |
| 5 | acegen_016df7dbf3bb | 0.6164 | 0.960 | 0.866 | 379.5 | 80.3 | 2.54 | 7 |

**Priority docking candidate:** `acegen_a7888cc3a940`
SMILES: `O=C(NC1CC1)c1ccc(C#Cc2ccc(C3CC3)nc2)cc1` | Reward: 0.5558 | TPSA: 42.0 Å²

Contains all four BACE1 pharmacophoric elements: basic nitrogen (Asp32/Asp228 catalytic dyad), cyclopropyl (S1 pocket), pyridine (S3 pocket), anilide carbonyl (flap region). First docking target against PDB 5CLM.

---

## Repository Structure

```
TRI_FLAG/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── save_folder/                            # DeepPurpose pretrained model weights
│   └── pretrained_models/
│       ├── model_mpnn_cnn_bindingdb.zip    # Zipped weights for distribution
│       └── model_MPNN_CNN/
│           ├── model.pt                    # MPNN_CNN_BindingDB weights
│           └── config.pkl                  # Model architecture config
│
└── triage_agent/                           # All pipeline code
    │
    ├── main.py                             # CLI entry point — triage a single molecule
    ├── streamlit_app.py                    # Interactive dashboard (4 tabs)
    ├── mcp_server.py                       # FastMCP server — 12 tools for Claude Desktop
    ├── molecule.py                         # Root-level molecule dataclass (legacy, preserved)
    ├── backfill_inchikeys.py               # One-time script: retroactively computed InChIKeys
    ├── export_for_apex.py                  # Curated CSVs for Oracle APEX (joined/filtered views)
    ├── export_for_apex_full.py             # Faithful 1:1 SQLite dump — one CSV per table, no filtering
    ├── test_molecules.csv                  # Small molecule test set for pipeline validation
    ├── pytest.ini                          # pytest configuration
    │
    ├── agent/                              # Triage agent core
    │   ├── agent_state.py                  # AgentState: single source of truth across all tools
    │   ├── triage_agent.py                 # TriageAgent: orchestrates tool sequence + early exit
    │   └── decision.py                     # DecisionType enum (PASS/FLAG/DISCARD) + Decision dataclass
    │
    ├── tools/                              # The five triage tools
    │   ├── base_tool.py                    # BaseTool ABC: run(), name, description
    │   ├── validity_tool.py                # RDKit SMILES parsing + canonicalization
    │   ├── descriptor_tool.py              # MW, logP, TPSA, HBD/HBA, Ro5 check, Murcko scaffold
    │   ├── sa_score_tool.py                # Ertl-Schuffenhauer score + sigmoid transform → S_sa
    │   ├── similarity_tool.py              # ECFP4 Tanimoto vs ChEMBL / SureChEMBL / PubChem
    │   ├── similarity_tool_backup_gen1.py  # Pre-SureChEMBL-rebuild backup (preserved)
    │   └── pains_tool.py                   # PAINS_A/B/C via RDKit FilterCatalog
    │
    ├── chemistry/                          # Low-level chemistry utilities
    │   ├── fingerprints.py                 # ECFP4 fingerprint generation + Tanimoto coefficient
    │   ├── sa_score.py                     # Ertl-Schuffenhauer fragment contribution model
    │   └── molecule_utils.py               # SMILES canonicalization, InChI/InChIKey, formula
    │
    ├── policies/                           # Decision logic
    │   ├── policy_engine.py                # PolicyEngine: aggregates tool outputs → final decision
    │   └── thresholds.py                   # SAScoreThresholds: default / lead_opt / nat_prod / fragment
    │
    ├── reporting/                          # Output assembly
    │   ├── scoring.py                      # compute_reward() → RewardResult (R = S_sa×S_nov×S_qed×S_act)
    │   ├── run_record.py                   # RunRecord + RunRecordBuilder + save() + load_as_dicts()
    │   └── rationale_builder.py            # Plain-English decision explanations
    │
    ├── loop/                               # ACEGEN reward function interface
    │   ├── config_triflag.yaml             # ACEGEN YAML config pointing to triflag_scorer
    │   └── triflag_scorer.py               # triflag_score(smiles: List[str]) → List[float]
    │
    ├── acegen_scripts/                     # Generation runner scripts
    │   ├── run_generation.py               # Main runner: cold-start / --warm-start / --gen N
    │   ├── run_gen0.py                     # Legacy Gen 0 script (preserved for reference)
    │   ├── run_gen1.py                     # Legacy Gen 1 script (pre-unified runner)
    │   └── run_gen2.py                     # Gen 2 script (first with 2024 SureChEMBL async API)
    │
    ├── target/                             # DeepPurpose BACE1 integration
    │   ├── target_config.py                # UniProt P56817 (501 aa), pIC50 bounds, model name
    │   └── deeppurpose_model.py            # Lazy model loader + S_act normalization
    │
    ├── database/                           # SQLite persistence layer
    │   ├── schema.py                       # CREATE TABLE statements + column name constants
    │   └── db.py                           # DatabaseManager: all reads/writes (no raw SQL elsewhere)
    │
    ├── analysis/                           # Post-hoc analytics
    │   ├── diversity.py                    # Murcko scaffold diversity + mode-collapse detection
    │   ├── oracle_dashboard.py             # 4 matplotlib plots + generation_summary.csv
    │   ├── sa_score_distribution.py        # SA score distribution across generations
    │   └── output/oracle/                  # Generated outputs (populated after first run)
    │       ├── mean_reward_trajectory.png
    │       ├── reward_distribution.png
    │       ├── decision_rates.png
    │       ├── scaffold_diversity.png
    │       └── generation_summary.csv
    │
    ├── runs/                               # Live data (NOT tracked in git — see exports/ for database)
    │   ├── triflag.db                      # SQLite database — download from exports/triflag_data.zip
    │   ├── triage_runs.jsonl               # JSONL run log for Streamlit session history
    │   ├── batch_streamlit.jsonl           # Batch results from Streamlit UI
    │   ├── batch_test.jsonl                # Batch results from test runs
    │   ├── out.jsonl                       # CLI output log
    │   └── checkpoints/
    │       ├── gen_006_policy.pt           # Gen 6 cold-start policy (baseline for warm-start)
    │       └── gen_007_policy.pt           # Gen 7 warm-start policy (current best)
    │
    ├── exports/                            # Database exports — full data available here
    │   ├── triflag_data.zip                # *** FULL DATABASE *** triflag.db + all 4 CSVs (download this)
    │   ├── triflag_molecules.csv           # Curated: all molecules with physicochemical properties
    │   ├── triflag_generations.csv         # Per-generation summary statistics
    │   ├── triflag_top_candidates.csv      # Top candidates shortlist (reward ≥ 0.4, PASS only)
    │   ├── export_manifest.txt             # Export metadata: timestamp, row counts, schema
    │   ├── apex_molecules.csv              # Full 1:1 dump of molecules table (7 cols)
    │   ├── apex_triage_runs.csv            # Full 1:1 dump of triage_runs table (34 cols)
    │   ├── apex_batches.csv                # Full 1:1 dump of batches table
    │   └── apex_target_predictions.csv     # Full 1:1 dump of target_predictions table
    │
    └── tests/                              # 374 passing, 2 skipped (network)
        ├── test_agent.py                   # TriageAgent integration tests
        ├── test_agent_similarity.py        # SimilarityTool tests (2 network-skipped)
        ├── test_sa_score.py                # SAScoreTool unit tests
        ├── test_scoring_validation.py      # Reward function + S_act ceiling mathematical proof
        ├── test_week6.py                   # ValidityTool, SAScoreTool, SimilarityTool (~80 tests)
        ├── test_week8.py                   # DatabaseManager, schema, reward function (30 tests)
        ├── test_week9.py                   # DescriptorTool, PAINSTool, oracle dashboard (~60 tests)
        ├── test_week10.py                  # ACEGEN loop, triflag_scorer, batch tracking (~64 tests)
        └── test_week11.py                  # DeepPurpose S_act, affinity normalization (~40 tests)
```

---

## Database Schema

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `molecules` | One row per unique molecule | `molecule_id`, `canonical_smiles`, `inchi`, `inchikey`, `molecular_formula` |
| `triage_runs` | One row per triage evaluation | `reward`, `s_sa`, `s_nov`, `s_qed`, `s_act`, `final_decision`, `batch_id`, `rationale`, `scaffold_smiles`, `predicted_affinity` |
| `batches` | One row per generation batch | `batch_id`, `generation_number`, `mean_reward`, `pass_count`, `flag_count`, `discard_count` |
| `target_predictions` | DeepPurpose outputs | `molecule_id`, `target_id` (P56817), `predicted_pic50`, `model_name` |

Access: `from database.db import DatabaseManager` — `DatabaseManager` is the single gateway; no raw SQL is written outside `db.py`.

**Note on `nn_source` / `nn_id`:** these are intentionally blank for PASS molecules with S_nov = 1.000. No near-neighbor was found above the FLAG threshold, so there is no compound to report. They are only populated for FLAG molecules (patent hits).

---

## Streamlit Dashboard

Four tabs, all reading live from `runs/triflag.db`:

| Tab | Description |
|-----|-------------|
| **Single Molecule** | SMILES input → decision badge, score breakdown, descriptors, PAINS status, rationale |
| **Batch Upload** | CSV of SMILES → run all → download results as CSV or JSONL |
| **Generation Analytics** | Loop status panel, top candidates table, reward trajectory and diversity plots |
| **AI Reviewer** | Generate a structured prompt for Claude to independently review a triage decision |

---

## MCP Server

12 tools via [FastMCP](https://github.com/jlowin/fastmcp) for Claude Desktop:

| Tool | Description |
|------|-------------|
| `triage_molecule` | Run full pipeline on a SMILES string |
| `get_top_candidates` | Top-N by reward (all batches or filtered by batch/min_reward) |
| `analyze_top_candidates` | Top-N with scientific interpretation and pharmacophore notes |
| `get_pass_candidates` | PASS-only shortlist (FLAGS excluded) |
| `analyze_pass_candidates` | PASS-only with full biological interpretation |
| `get_database_summary` | Total molecules, mean/max reward, decision counts |
| `get_all_generations_summary` | Cross-generation reward trend table |
| `get_generation_progress` | Live progress for an active run |
| `get_generation_stats` | Detailed stats for one generation |
| `launch_generation` | Spawn ACEGEN as a detached subprocess |
| `search_by_scaffold` | Find all molecules sharing a given Murcko scaffold |
| `get_decision_breakdown` | PASS/FLAG/DISCARD counts broken down by reason |

---

## Known Issues

**`get_generation_progress` denominator** — Uses ~4,160 as its baseline, not the `total_smiles` config value of 6,400. Use `get_generation_stats` for accurate completion percentage.

**Gen 0 not comparable to Gens 1–7** — Gen 0 used a 3-component reward without S_act and with `SKIP_SIMILARITY=1`. S_act was added in Gen 1, which structurally reduces mean reward (fourth multiplicative factor in [0,1]). RL learning trend analysis starts at Gen 1.

**SureChEMBL API availability** — The 2024 async API rebuild removed the legacy sync endpoint. If EBI is unreachable, `triage_molecule` fails at connection stage. Use `skip_similarity=True` for offline use (disables IP screening). The pre-rebuild implementation is preserved at `tools/similarity_tool_backup_gen1.py`.

**DeepPurpose on Windows** — Set `TRIFLAG_ENABLE_DEEPPURPOSE=1` before all other imports. Exit code `0xC0000139` (`shm.dll`) on bare import cannot be caught by Python `try/except`.

**ACEGEN warm-start patching** — Do not modify files inside `acegen-open/`. The warm-start mechanism patches `reinvent.py` in memory at runtime via string injection. Source changes will silently break the injection points.

**BACE1 sequence** — Must be the full 501-residue UniProt P56817 sequence in `target/target_config.py`. A 487-residue truncated version (missing the catalytic aspartyl dyad) was corrected during development.

---

## References

1. Bou A, et al. ACEGEN: Reinforcement Learning of Generative Chemical Agents for Drug Discovery. *J Chem Inf Model.* 2024;64(15):5900–5911.
2. Huang K, et al. DeepPurpose: a deep learning library for drug-target interaction prediction. *Bioinformatics.* 2021;36(22-23):5545–5547.
3. Naidu A, et al. Safety concerns associated with BACE1 inhibitors. *Expert Opin Drug Saf.* 2025;24(7):767–772.
4. Gao W, Luo S, Coley CW. Generative AI for navigating synthesizable chemical space. *PNAS.* 2025;122(41):e2415665122.
5. Ertl P, Schuffenhauer A. Estimation of synthetic accessibility score of drug-like molecules. *J Cheminform.* 2009;1:8.
6. Bickerton GR, et al. Quantifying the chemical beauty of drugs. *Nat Chem.* 2012;4(2):90–98.
7. Mendez D, et al. ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Res.* 2019;47(D1):D930–D940.

---

*Built within the [OSDD2](https://osdd2.org/) framework at UAB SPARC. All 34,915 molecules and full source available at https://github.com/kjamescasper/TRI_FLAG.*
