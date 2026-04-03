"""
database/db.py

DatabaseManager: the single gateway for all TRI_FLAG SQLite reads and writes.

Design principles:
  - Everything else in the codebase calls DatabaseManager methods.
    No raw SQL is written outside this file.
  - WAL journal mode is configured once on connection, enabling concurrent
    access from Streamlit, CLI, and MCP without locking errors.
  - save_run() writes molecules + triage_runs in a single transaction,
    preserving referential integrity even on crash.

Field name mapping (real RunRecord vs test mock):
    Real RunRecord              Test MockRunRecord
    smiles_canonical            canonical_smiles
    run_timestamp               triaged_at
    synthesizability_category   sa_category
    rule_rationale              rationale

save_run() resolves both via getattr fallback chains so tests and the
real pipeline both work without any changes.

Week 8 — initial implementation.
Week 11 — save_run() extended to write affinity fields;
           get_top_n_by_affinity() added. No structural changes needed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from database.schema import (
    ALL_CREATE_STATEMENTS,
    CREATE_INDEXES_SQL,
    COL_BATCH_ID,
    COL_CANONICAL_SMILES,
    COL_CREATED_AT,
    COL_DISCARD_COUNT,
    COL_ENTRY_POINT,
    COL_FINAL_DECISION,
    COL_FLAG_COUNT,
    COL_GENERATION_NUMBER,
    COL_HBA,
    COL_HBD,
    COL_INCHI,
    COL_INCHIKEY,
    COL_IS_VALID,
    COL_LOGP,
    COL_MEAN_REWARD,
    COL_MOL_WEIGHT,
    COL_MOLECULE_COUNT,
    COL_MOLECULE_ID,
    COL_MOLECULAR_FORMULA,
    COL_NN_ID,
    COL_NN_NAME,
    COL_NN_SOURCE,
    COL_NN_TANIMOTO,
    COL_PAINS_ALERT,
    COL_PASS_COUNT,
    COL_PREDICTED_AFFINITY,
    COL_RATIONALE,
    COL_RAW_JSON,
    COL_ROTATABLE_BONDS,
    COL_RUN_ID,
    COL_SA_CATEGORY,
    COL_SA_DECISION,
    COL_SA_SCORE,
    COL_SCAFFOLD_SMILES,
    COL_SIMILARITY_DECISION,
    COL_SOURCE,
    COL_S_ACT,
    COL_S_NOV,
    COL_S_QED,
    COL_S_SA,
    COL_TARGET_ID,
    COL_TPSA,
    COL_TRIAGED_AT,
    COL_VALIDITY_ERROR,
    COL_REWARD,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "runs/triflag.db"


class DatabaseManager:
    """
    Wraps all SQLite reads and writes for TRI_FLAG.

    Instantiate once per process (or once per request in Streamlit) and
    reuse. The connection is opened lazily on the first call to _conn().

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite file. Created automatically if absent.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None
        self.init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._connection = conn
            logger.debug("Opened SQLite connection: %s", self.db_path)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables and indexes (CREATE IF NOT EXISTS — safe to repeat)."""
        conn = self._conn()
        with conn:
            for ddl in ALL_CREATE_STATEMENTS:
                conn.execute(ddl)
            for idx_sql in CREATE_INDEXES_SQL:
                conn.execute(idx_sql)
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_run(self, record) -> None:
        """
        Persist a RunRecord to SQLite in a single transaction.

        Resolves field name differences between the real RunRecord and the
        lightweight test mock so both work without modification:

            Real RunRecord field       Mock alias
            smiles_canonical      -->  canonical_smiles
            run_timestamp         -->  triaged_at
            synthesizability_cat  -->  sa_category
            rule_rationale        -->  rationale

        Parameters
        ----------
        record : RunRecord
            The completed triage record from RunRecordBuilder.build().
        """
        conn = self._conn()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Serialise full record for raw_json column
        try:
            raw_json_str = json.dumps(
                record.to_dict() if hasattr(record, "to_dict") else {}
            )
        except Exception:
            raw_json_str = "{}"

        # Resolve field name differences
        canonical_smiles = (
            getattr(record, "smiles_canonical", None)
            or getattr(record, "canonical_smiles", None)
            or getattr(record, "smiles_input", None)
        )
        triaged_at = (
            getattr(record, "run_timestamp", None)
            or getattr(record, "triaged_at", None)
            or now_iso
        )
        sa_category = (
            getattr(record, "synthesizability_category", None)
            or getattr(record, "sa_category", None)
        )
        rationale = (
            getattr(record, "rule_rationale", None)
            or getattr(record, "rationale", None)
        )
        is_valid_val = int(bool(getattr(record, "is_valid", False)))
        pains = getattr(record, "pains_alert", None)
        pains_int = int(pains) if pains is not None else None

        with conn:
            # molecules row
            conn.execute(
                f"""
                INSERT OR IGNORE INTO molecules (
                    {COL_MOLECULE_ID},
                    {COL_CANONICAL_SMILES},
                    {COL_INCHI},
                    {COL_INCHIKEY},
                    {COL_MOLECULAR_FORMULA},
                    {COL_SOURCE},
                    {COL_CREATED_AT}
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.molecule_id,
                    canonical_smiles,
                    getattr(record, "inchi", None),
                    getattr(record, "inchikey", None),
                    getattr(record, "molecular_formula", None),
                    getattr(record, "source", "triflag"),
                    now_iso,
                ),
            )

            # triage_runs row
            conn.execute(
                f"""
                INSERT INTO triage_runs (
                    {COL_RUN_ID},
                    {COL_MOLECULE_ID},
                    {COL_BATCH_ID},
                    {COL_GENERATION_NUMBER},
                    {COL_TRIAGED_AT},
                    {COL_ENTRY_POINT},
                    {COL_FINAL_DECISION},
                    {COL_REWARD},
                    {COL_S_SA},
                    {COL_S_NOV},
                    {COL_S_QED},
                    {COL_S_ACT},
                    {COL_SA_SCORE},
                    {COL_SA_DECISION},
                    {COL_SA_CATEGORY},
                    {COL_NN_TANIMOTO},
                    {COL_NN_SOURCE},
                    {COL_NN_ID},
                    {COL_NN_NAME},
                    {COL_SIMILARITY_DECISION},
                    {COL_IS_VALID},
                    {COL_VALIDITY_ERROR},
                    {COL_MOL_WEIGHT},
                    {COL_LOGP},
                    {COL_TPSA},
                    {COL_HBD},
                    {COL_HBA},
                    {COL_ROTATABLE_BONDS},
                    {COL_SCAFFOLD_SMILES},
                    {COL_PAINS_ALERT},
                    {COL_PREDICTED_AFFINITY},
                    {COL_TARGET_ID},
                    {COL_RATIONALE},
                    {COL_RAW_JSON}
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    record.run_id,
                    record.molecule_id,
                    getattr(record, "batch_id", None),
                    getattr(record, "generation_number", None),
                    triaged_at,
                    getattr(record, "entry_point", None),
                    record.final_decision,
                    getattr(record, "reward", None),
                    getattr(record, "s_sa", None),
                    getattr(record, "s_nov", None),
                    getattr(record, "s_qed", None),
                    getattr(record, "s_act", None),
                    getattr(record, "sa_score", None),
                    getattr(record, "sa_decision", None),
                    sa_category,
                    getattr(record, "nn_tanimoto", None),
                    getattr(record, "nn_source", None),
                    getattr(record, "nn_id", None),
                    getattr(record, "nn_name", None),
                    getattr(record, "similarity_decision", None),
                    is_valid_val,
                    getattr(record, "validity_error", None),
                    getattr(record, "mol_weight", None),
                    getattr(record, "logp", None),
                    getattr(record, "tpsa", None),
                    getattr(record, "hbd", None),
                    getattr(record, "hba", None),
                    getattr(record, "rotatable_bonds", None),
                    getattr(record, "scaffold_smiles", None),
                    pains_int,
                    getattr(record, "predicted_affinity", None),
                    getattr(record, "target_id", None),
                    rationale,
                    raw_json_str,
                ),
            )

        logger.info(
            "Saved run %s (molecule=%s, decision=%s, reward=%s)",
            record.run_id,
            record.molecule_id,
            record.final_decision,
            getattr(record, "reward", None),
        )

    def save_batch(self, batch_id: str, stats: dict) -> None:
        """
        Upsert a batch summary row.

        Parameters
        ----------
        batch_id : str
            Unique identifier for the ACEGEN generation batch.
        stats : dict
            Expected keys: generation_number, source, molecule_count,
            pass_count, flag_count, discard_count, mean_reward.
        """
        conn = self._conn()
        now_iso = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO batches (
                    {COL_BATCH_ID},
                    {COL_GENERATION_NUMBER},
                    {COL_SOURCE},
                    {COL_CREATED_AT},
                    {COL_MOLECULE_COUNT},
                    {COL_PASS_COUNT},
                    {COL_FLAG_COUNT},
                    {COL_DISCARD_COUNT},
                    {COL_MEAN_REWARD}
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    stats.get("generation_number"),
                    stats.get("source"),
                    now_iso,
                    stats.get("molecule_count", 0),
                    stats.get("pass_count", 0),
                    stats.get("flag_count", 0),
                    stats.get("discard_count", 0),
                    stats.get("mean_reward"),
                ),
            )
        logger.info("Saved batch %s (%s molecules)", batch_id, stats.get("molecule_count"))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_runs_by_batch(self, batch_id: str) -> List[sqlite3.Row]:
        """Return all triage_runs rows for a given batch_id, oldest first."""
        conn = self._conn()
        cursor = conn.execute(
            f"""
            SELECT tr.*, m.{COL_CANONICAL_SMILES}, m.{COL_INCHI},
                   m.{COL_INCHIKEY}, m.{COL_MOLECULAR_FORMULA}
            FROM triage_runs tr
            JOIN molecules m USING ({COL_MOLECULE_ID})
            WHERE tr.{COL_BATCH_ID} = ?
            ORDER BY tr.{COL_TRIAGED_AT}
            """,
            (batch_id,),
        )
        rows = cursor.fetchall()
        logger.debug("load_runs_by_batch(%s): %d rows", batch_id, len(rows))
        return rows

    def get_top_n_by_reward(
        self,
        n: int,
        batch_id: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """
        Return top-N runs ranked by reward DESC.
        This is the query ACEGEN uses to pick positive training examples.
        """
        conn = self._conn()
        if batch_id is not None:
            cursor = conn.execute(
                f"""
                SELECT tr.*, m.{COL_CANONICAL_SMILES}
                FROM triage_runs tr
                JOIN molecules m USING ({COL_MOLECULE_ID})
                WHERE tr.{COL_BATCH_ID} = ?
                  AND tr.{COL_REWARD} IS NOT NULL
                ORDER BY tr.{COL_REWARD} DESC
                LIMIT ?
                """,
                (batch_id, n),
            )
        else:
            cursor = conn.execute(
                f"""
                SELECT tr.*, m.{COL_CANONICAL_SMILES}
                FROM triage_runs tr
                JOIN molecules m USING ({COL_MOLECULE_ID})
                WHERE tr.{COL_REWARD} IS NOT NULL
                ORDER BY tr.{COL_REWARD} DESC
                LIMIT ?
                """,
                (n,),
            )
        rows = cursor.fetchall()
        logger.debug("get_top_n_by_reward(n=%d, batch=%s): %d rows", n, batch_id, len(rows))
        return rows

    def get_top_n_by_affinity(
        self,
        n: int,
        target_id: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """
        Return top-N runs ranked by predicted_affinity DESC.

        Week 11 addition. Used by the Streamlit shortlist tab and MCP
        get_shortlist() to surface the highest-affinity candidates.

        Parameters
        ----------
        n : int
            Maximum number of rows to return.
        target_id : str | None
            If provided, filter to runs where target_id matches.
            If None, returns top-N across all targets.

        Returns
        -------
        list of sqlite3.Row
            Rows from triage_runs joined with molecules, ordered by
            predicted_affinity DESC. Rows with NULL predicted_affinity
            are excluded.
        """
        conn = self._conn()
        if target_id is not None:
            cursor = conn.execute(
                f"""
                SELECT tr.*, m.{COL_CANONICAL_SMILES}
                FROM triage_runs tr
                JOIN molecules m USING ({COL_MOLECULE_ID})
                WHERE tr.{COL_PREDICTED_AFFINITY} IS NOT NULL
                  AND tr.{COL_TARGET_ID} = ?
                ORDER BY tr.{COL_PREDICTED_AFFINITY} DESC
                LIMIT ?
                """,
                (target_id, n),
            )
        else:
            cursor = conn.execute(
                f"""
                SELECT tr.*, m.{COL_CANONICAL_SMILES}
                FROM triage_runs tr
                JOIN molecules m USING ({COL_MOLECULE_ID})
                WHERE tr.{COL_PREDICTED_AFFINITY} IS NOT NULL
                ORDER BY tr.{COL_PREDICTED_AFFINITY} DESC
                LIMIT ?
                """,
                (n,),
            )
        rows = cursor.fetchall()
        logger.debug(
            "get_top_n_by_affinity(n=%d, target=%s): %d rows",
            n, target_id, len(rows),
        )
        return rows

    def get_all_runs(
        self,
        limit: Optional[int] = None,
        decision_filter: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """Return triage_runs rows, newest first."""
        conn = self._conn()
        where_clauses = []
        params: list = []

        if decision_filter:
            where_clauses.append(f"tr.{COL_FINAL_DECISION} = ?")
            params.append(decision_filter)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""

        cursor = conn.execute(
            f"""
            SELECT tr.*, m.{COL_CANONICAL_SMILES}
            FROM triage_runs tr
            JOIN molecules m USING ({COL_MOLECULE_ID})
            {where_sql}
            ORDER BY tr.{COL_TRIAGED_AT} DESC
            {limit_sql}
            """,
            params,
        )
        return cursor.fetchall()

    def get_run_by_id(self, run_id: str) -> Optional[sqlite3.Row]:
        """Fetch a single run by run_id, or None if not found."""
        conn = self._conn()
        cursor = conn.execute(
            f"""
            SELECT tr.*, m.{COL_CANONICAL_SMILES}, m.{COL_INCHI}, m.{COL_INCHIKEY}
            FROM triage_runs tr
            JOIN molecules m USING ({COL_MOLECULE_ID})
            WHERE tr.{COL_RUN_ID} = ?
            """,
            (run_id,),
        )
        return cursor.fetchone()

    def get_generation_stats(self, generation_number: int) -> dict:
        """
        Aggregate statistics for a given generation number.
        Used by oracle dashboard (Week 9+) and MCP get_generation_stats (Week 10).
        """
        conn = self._conn()
        cursor = conn.execute(
            f"""
            SELECT
                COUNT(*)                                          AS total,
                SUM(CASE WHEN {COL_FINAL_DECISION}='PASS'    THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN {COL_FINAL_DECISION}='FLAG'    THEN 1 ELSE 0 END) AS flag_count,
                SUM(CASE WHEN {COL_FINAL_DECISION}='DISCARD' THEN 1 ELSE 0 END) AS discard_count,
                AVG({COL_REWARD})                                 AS mean_reward,
                MAX({COL_REWARD})                                 AS max_reward,
                MIN({COL_REWARD})                                 AS min_reward,
                AVG({COL_SA_SCORE})                               AS mean_sa_score,
                AVG({COL_NN_TANIMOTO})                            AS mean_tanimoto
            FROM triage_runs
            WHERE {COL_GENERATION_NUMBER} = ?
            """,
            (generation_number,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}