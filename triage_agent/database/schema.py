"""
database/schema.py

Single source of truth for the TRI_FLAG SQLite schema.

Contains:
  - CREATE TABLE SQL strings (imported by DatabaseManager.init_db())
  - Column name constants (imported everywhere to avoid magic strings)

Biolink alignment:
  molecules  → biolink:SmallMolecule  (subclass of ChemicalEntity)
  triage_runs → biolink:Association   (subject=molecule, predicate=triaged_as)

Schema is forward-compatible: nullable descriptor columns (logP, TPSA, MW,
HBD, HBA, rotatable_bonds, scaffold_smiles, pains_alert, predicted_affinity,
target_id) are reserved here and populated in Weeks 9–11 without migration.

Week 8 — do not modify column layout; extend by adding nullable columns only.
Week 11 — target_predictions table activated (was stub); affinity index added.
"""

# ===========================================================================
# molecules table
# Biolink class: biolink:SmallMolecule
# ===========================================================================

CREATE_MOLECULES_SQL = """
CREATE TABLE IF NOT EXISTS molecules (
    molecule_id         TEXT PRIMARY KEY,
    canonical_smiles    TEXT NOT NULL,
    inchi               TEXT,
    inchikey            TEXT,
    molecular_formula   TEXT,
    source              TEXT DEFAULT 'triflag',
    created_at          TEXT NOT NULL
);
"""

# ===========================================================================
# triage_runs table
# Biolink class: biolink:Association
#   subject   = molecule_id
#   predicate = triflag:triaged_as
#   object    = final_decision
# ===========================================================================

CREATE_TRIAGE_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS triage_runs (
    run_id              TEXT PRIMARY KEY,
    molecule_id         TEXT NOT NULL REFERENCES molecules(molecule_id),

    -- Association provenance
    batch_id            TEXT,
    generation_number   INTEGER,
    triaged_at          TEXT NOT NULL,
    entry_point         TEXT,

    -- Triage outcome
    final_decision      TEXT NOT NULL,
    reward              REAL,

    -- Reward components (stored for dashboard decomposition)
    s_sa                REAL,
    s_nov               REAL,
    s_qed               REAL,
    s_act               REAL,

    -- SA score  (biolink:has_attribute → SynthesizabilityScore)
    sa_score            REAL,
    sa_decision         TEXT,
    sa_category         TEXT,

    -- IP similarity (nearest known biolink:ChemicalEntity)
    nn_tanimoto         REAL,
    nn_source           TEXT,
    nn_id               TEXT,
    nn_name             TEXT,
    similarity_decision TEXT,

    -- Validity
    is_valid            INTEGER NOT NULL,
    validity_error      TEXT,

    -- Week 9 descriptor columns (nullable until DescriptorTool added)
    mol_weight          REAL,
    logp                REAL,
    tpsa                REAL,
    hbd                 INTEGER,
    hba                 INTEGER,
    rotatable_bonds     INTEGER,
    scaffold_smiles     TEXT,

    -- Week 9 PAINS column (nullable until PAINSTool added)
    pains_alert         INTEGER,

    -- Week 11 target-binding columns (populated by DeepPurpose)
    predicted_affinity  REAL,
    target_id           TEXT,

    -- Full rationale + raw record
    rationale           TEXT,
    raw_json            TEXT
);
"""

# ===========================================================================
# batches table — tracks ACEGEN generation metadata
# ===========================================================================

CREATE_BATCHES_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id            TEXT PRIMARY KEY,
    generation_number   INTEGER,
    source              TEXT,
    created_at          TEXT NOT NULL,
    molecule_count      INTEGER,
    pass_count          INTEGER,
    flag_count          INTEGER,
    discard_count       INTEGER,
    mean_reward         REAL
);
"""

# ===========================================================================
# target_predictions table — Week 11 (activated from stub)
# Biolink class: biolink:ChemicalToTargetAssociation
# ===========================================================================

CREATE_TARGET_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS target_predictions (
    prediction_id       TEXT PRIMARY KEY,
    molecule_id         TEXT NOT NULL REFERENCES molecules(molecule_id),
    target_id           TEXT NOT NULL,
    predicted_pic50     REAL,
    predicted_affinity  REAL,
    model_name          TEXT,
    predicted_at        TEXT NOT NULL
);
"""

# ===========================================================================
# Index DDL — created once by init_db()
# ===========================================================================

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_runs_molecule   ON triage_runs(molecule_id);",
    "CREATE INDEX IF NOT EXISTS idx_runs_batch      ON triage_runs(batch_id);",
    "CREATE INDEX IF NOT EXISTS idx_runs_decision   ON triage_runs(final_decision);",
    "CREATE INDEX IF NOT EXISTS idx_runs_reward     ON triage_runs(reward);",
    "CREATE INDEX IF NOT EXISTS idx_runs_generation ON triage_runs(generation_number);",
    # Week 11: index on predicted_affinity for get_top_n_by_affinity() queries
    "CREATE INDEX IF NOT EXISTS idx_runs_affinity   ON triage_runs(predicted_affinity);",
    # Week 11: index on target_predictions for per-target lookups
    "CREATE INDEX IF NOT EXISTS idx_tp_molecule     ON target_predictions(molecule_id);",
    "CREATE INDEX IF NOT EXISTS idx_tp_target       ON target_predictions(target_id);",
]

# ===========================================================================
# Column name constants — molecules table
# ===========================================================================

COL_MOLECULE_ID       = "molecule_id"
COL_CANONICAL_SMILES  = "canonical_smiles"
COL_INCHI             = "inchi"
COL_INCHIKEY          = "inchikey"
COL_MOLECULAR_FORMULA = "molecular_formula"
COL_SOURCE            = "source"
COL_CREATED_AT        = "created_at"

# ===========================================================================
# Column name constants — triage_runs table
# ===========================================================================

COL_RUN_ID            = "run_id"
COL_BATCH_ID          = "batch_id"
COL_GENERATION_NUMBER = "generation_number"
COL_TRIAGED_AT        = "triaged_at"
COL_ENTRY_POINT       = "entry_point"
COL_FINAL_DECISION    = "final_decision"
COL_REWARD            = "reward"
COL_S_SA              = "s_sa"
COL_S_NOV             = "s_nov"
COL_S_QED             = "s_qed"
COL_S_ACT             = "s_act"
COL_SA_SCORE          = "sa_score"
COL_SA_DECISION       = "sa_decision"
COL_SA_CATEGORY       = "sa_category"
COL_NN_TANIMOTO       = "nn_tanimoto"
COL_NN_SOURCE         = "nn_source"
COL_NN_ID             = "nn_id"
COL_NN_NAME           = "nn_name"
COL_SIMILARITY_DECISION = "similarity_decision"
COL_IS_VALID          = "is_valid"
COL_VALIDITY_ERROR    = "validity_error"
COL_MOL_WEIGHT        = "mol_weight"
COL_LOGP              = "logp"
COL_TPSA              = "tpsa"
COL_HBD               = "hbd"
COL_HBA               = "hba"
COL_ROTATABLE_BONDS   = "rotatable_bonds"
COL_SCAFFOLD_SMILES   = "scaffold_smiles"
COL_PAINS_ALERT       = "pains_alert"
COL_PREDICTED_AFFINITY = "predicted_affinity"
COL_TARGET_ID         = "target_id"
COL_RATIONALE         = "rationale"
COL_RAW_JSON          = "raw_json"

# ===========================================================================
# Column name constants — batches table
# ===========================================================================

COL_MOLECULE_COUNT    = "molecule_count"
COL_PASS_COUNT        = "pass_count"
COL_FLAG_COUNT        = "flag_count"
COL_DISCARD_COUNT     = "discard_count"
COL_MEAN_REWARD       = "mean_reward"

# ===========================================================================
# All table DDL in init order (dependency-safe)
# ===========================================================================

ALL_CREATE_STATEMENTS = [
    CREATE_MOLECULES_SQL,
    CREATE_TRIAGE_RUNS_SQL,
    CREATE_BATCHES_SQL,
    CREATE_TARGET_PREDICTIONS_SQL,
]