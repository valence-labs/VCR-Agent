import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import dotenv
import duckdb
import pandas as pd
from loguru import logger

dotenv.load_dotenv()
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", ":memory:")


class DuckDBCache:
    """
    DuckDB cache for ChEMBL compounds storing CURATED JSON only, plus extra JSON docs.

    Tables:
      - chembl_compounds (
            inchikey   TEXT PRIMARY KEY,
            chembl_id  TEXT,
            name       TEXT,
            smiles     TEXT,
            payload    JSON,            -- curated compound JSON
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
      - chembl_synonyms (inchikey TEXT, synonym TEXT, PRIMARY KEY (inchikey, synonym))
      - chembl_payloads (
            inchikey   TEXT,
            kind       TEXT,               -- e.g., 'bioactivities'
            source_id  TEXT NOT NULL DEFAULT '',  -- optional identifier
            payload    JSON,               -- arbitrary JSON blob
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            PRIMARY KEY (inchikey, kind, source_id)
        )

    Public API:
      - save_compound(record)  # expects keys: inchikey, chembl_id, name, smiles, synonyms?, payload(curated)?
      - save_bioactivities(inchikey, payload, source_id=None)
      - query(inchikey) -> dict | None
      - query_by_synonym(name) -> dict | None
      - query_payloads(inchikey, kind='bioactivities') -> list[dict]
      - create_from_df(df)  # optional bulk load (df may contain 'synonyms' and 'payload')
    """

    def __init__(self, cache_path: str | Path = DUCKDB_PATH, key_col: str = "inchikey"):
        self.key_col = key_col  # kept for compatibility
        self._in_memory = (cache_path is None) or (str(cache_path) == ":memory:")
        self.cache_path = None if self._in_memory else Path(cache_path)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._con = duckdb.connect(":memory:" if self._in_memory else str(self.cache_path))
        self._ensure_schema()

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("DuckDBCache connection is closed. Create a new instance or call reopen().")
        return self._con

    def _ensure_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS chembl_compounds (
                inchikey   TEXT PRIMARY KEY,
                chembl_id  TEXT,
                name       TEXT,
                smiles     TEXT,
                payload    JSON,
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS chembl_synonyms (
                inchikey TEXT,
                synonym  TEXT,
                PRIMARY KEY (inchikey, synonym)
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS chembl_payloads (
                inchikey   TEXT,
                kind       TEXT,
                source_id  TEXT NOT NULL DEFAULT '',
                payload    JSON,
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP,
                PRIMARY KEY (inchikey, kind, source_id)
            );
        """)

        # Helpful indexes
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_compounds_chembl_id ON chembl_compounds(chembl_id);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_compounds_name      ON chembl_compounds(name);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_compounds_smiles    ON chembl_compounds(smiles);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_synonym_lower       ON chembl_synonyms(lower(synonym));")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_payloads_kind       ON chembl_payloads(kind);")

    def _norm_record(self, record: dict) -> tuple[dict, list[str], str | None]:
        """
        Normalize an incoming record into:
          - compound row (inchikey, chembl_id, name, smiles)
          - synonyms list
          - payload_json_str (full JSON of compound) or None
        """
        row = {
            "inchikey": record.get("inchikey"),
            "chembl_id": record.get("chembl_id"),
            "name": record.get("name"),
            "smiles": record.get("smiles"),
        }
        if not row["inchikey"]:
            raise ValueError("Record must contain 'inchikey'.")

        # Synonyms normalization (list or scalar)
        syns = record.get("synonyms") or []
        if isinstance(syns, str):
            syns = [syns]
        seen, uniq = set(), []
        for s in syns:
            if not isinstance(s, str):
                continue
            s2 = s.strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                uniq.append(s2)

        # Payload handling (store the WHOLE JSON if provided)
        payload = record.get("payload") or record.get("raw_payload") or record.get("_raw") or None
        payload_json_str = None
        if payload is not None:
            if isinstance(payload, dict | list):
                payload_json_str = json.dumps(payload, ensure_ascii=False)
            elif isinstance(payload, str):
                # accept pre-serialized JSON
                payload_json_str = payload
            else:
                payload_json_str = None

        return row, uniq, payload_json_str

    def _upsert_compound(self, row: dict, payload_json_str: str | None):
        df = pd.DataFrame([{**row, "_payload_text": payload_json_str}])
        self.con.register("rec_compound", df)

        # Update existing
        self.con.execute("""
            UPDATE chembl_compounds AS t
            SET
                chembl_id  = COALESCE(s.chembl_id,  t.chembl_id),
                name       = COALESCE(s.name,       t.name),
                smiles     = COALESCE(s.smiles,     t.smiles),
                payload    = COALESCE(CAST(s._payload_text AS JSON), t.payload),
                updated_at = now()
            FROM rec_compound AS s
            WHERE t.inchikey = s.inchikey;
        """)

        # Insert new
        self.con.execute("""
            INSERT INTO chembl_compounds (inchikey, chembl_id, name, smiles, payload, created_at)
            SELECT s.inchikey, s.chembl_id, s.name, s.smiles, CAST(s._payload_text AS JSON), now()
            FROM rec_compound AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM chembl_compounds t WHERE t.inchikey = s.inchikey
            );
        """)

        self.con.unregister("rec_compound")

    def _merge_synonyms(self, inchikey: str, synonyms: Iterable[str]):
        if not synonyms:
            return
        df = pd.DataFrame({"inchikey": [inchikey] * len(synonyms), "synonym": list(synonyms)})
        self.con.register("rec_syns", df)

        # Insert only new (inchikey, synonym) pairs
        self.con.execute("""
            INSERT INTO chembl_synonyms (inchikey, synonym)
            SELECT s.inchikey, s.synonym
            FROM rec_syns AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM chembl_synonyms t
                WHERE t.inchikey = s.inchikey AND t.synonym = s.synonym
            );
        """)

        self.con.unregister("rec_syns")

    def save_compound(self, record: dict):
        """
        Upsert a compound and its synonyms, and store the CURATED compound JSON payload if provided.

        Expected keys on `record`:
          - inchikey (required)
          - chembl_id, name, smiles (optional but recommended)
          - synonyms (list[str] or str) (optional)
          - payload / raw_payload / _raw (dict | str) (optional, curated compound JSON)
        """
        if not record:
            return
        try:
            row, syns, payload_json_str = self._norm_record(record)
            self._upsert_compound(row, payload_json_str)
            self._merge_synonyms(row["inchikey"], syns)
        except duckdb.Error as e:
            logger.error(f"Failed to append record: {e}")

    def save_bioactivities(self, inchikey: str, payload: Any, source_id: str | None = None):
        """
        Save a JSON blob of bioactivities (or any additional doc) keyed by inchikey.

        If a row with the same (inchikey, kind='bioactivities', source_id) exists, it is updated.
        """
        if not inchikey:
            raise ValueError("inchikey is required")

        payload_text = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
        df = pd.DataFrame(
            [
                {
                    "inchikey": inchikey,
                    "kind": "bioactivities",
                    "source_id": source_id or "",  # must be non-null for PK
                    "_payload_text": payload_text,
                }
            ]
        )
        self.con.register("rec_payload", df)

        # Update existing
        self.con.execute("""
            UPDATE chembl_payloads AS t
            SET payload = CAST(s._payload_text AS JSON),
                updated_at = now()
            FROM rec_payload AS s
            WHERE t.inchikey = s.inchikey
              AND t.kind      = s.kind
              AND t.source_id = s.source_id;
        """)

        # Insert new
        self.con.execute("""
            INSERT INTO chembl_payloads (inchikey, kind, source_id, payload, created_at)
            SELECT s.inchikey, s.kind, s.source_id, CAST(s._payload_text AS JSON), now()
            FROM rec_payload AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM chembl_payloads t
                WHERE t.inchikey = s.inchikey
                  AND t.kind      = s.kind
                  AND t.source_id = s.source_id
            );
        """)

        self.con.unregister("rec_payload")

    def query(self, inchikey: str) -> dict | None:
        """Return compound dict with `synonyms` (list[str]) and parsed `payload` (dict) if present."""
        try:
            res = self.con.execute(
                "SELECT inchikey, chembl_id, name, smiles, payload, created_at, updated_at "
                "FROM chembl_compounds WHERE inchikey = ? LIMIT 1;",
                [inchikey],
            )
            row = res.fetchone()
            if not row:
                return None

            cols = [d[0] for d in res.description]
            compound = dict(zip(cols, row, strict=True))

            # Parse JSON payload to Python dict (DuckDB may return str or dict depending on version)
            payload_val = compound.get("payload")
            if isinstance(payload_val, str):
                try:
                    compound["payload"] = json.loads(payload_val)
                except Exception:
                    pass  # leave as-is if not valid JSON

            syn_rows = self.con.execute(
                "SELECT synonym FROM chembl_synonyms WHERE inchikey = ? ORDER BY lower(synonym);",
                [inchikey],
            ).fetchall()
            compound["synonyms"] = [s[0] for s in syn_rows] if syn_rows else None

            # strip metadata if you want lean output
            compound.pop("created_at", None)
            compound.pop("updated_at", None)
            return compound
        except duckdb.Error as e:
            logger.error(f"DuckDB query error: {e}")
            return None

    def query_by_synonym(self, name: str) -> dict | None:
        """Lookup a compound by an exact synonym match (case-insensitive)."""
        try:
            res = self.con.execute(
                """
                SELECT c.inchikey, c.chembl_id, c.name, c.smiles, c.payload
                FROM chembl_compounds c
                JOIN chembl_synonyms s ON c.inchikey = s.inchikey
                WHERE lower(s.synonym) = lower(?)
                LIMIT 1;
                """,
                [name],
            )
            row = res.fetchone()
            if not row:
                return None

            cols = [d[0] for d in res.description]
            compound = dict(zip(cols, row, strict=True))

            payload_val = compound.get("payload")
            if isinstance(payload_val, str):
                try:
                    compound["payload"] = json.loads(payload_val)
                except Exception:
                    pass

            syn_rows = self.con.execute(
                "SELECT synonym FROM chembl_synonyms WHERE inchikey = ? ORDER BY lower(synonym);",
                [compound["inchikey"]],
            ).fetchall()
            compound["synonyms"] = [s[0] for s in syn_rows] if syn_rows else None

            return compound
        except duckdb.Error as e:
            logger.error(f"DuckDB query_by_synonym error: {e}")
            return None

    def query_payloads(self, inchikey: str, kind: str = "bioactivities") -> list[dict]:
        """Return a list of parsed JSON docs for the given kind (default: 'bioactivities')."""
        try:
            rows = self.con.execute(
                "SELECT payload FROM chembl_payloads WHERE inchikey = ? AND kind = ? ORDER BY created_at;",
                [inchikey, kind],
            ).fetchall()
            out = []
            for (p,) in rows:
                if isinstance(p, str):
                    try:
                        out.append(json.loads(p))
                    except Exception:
                        out.append({"_raw": p})
                else:
                    out.append(p)
            return out
        except duckdb.Error as e:
            logger.error(f"DuckDB query_payloads error: {e}")
            return []

    def create_from_df(self, df: pd.DataFrame):
        """
        Bulk (re)create compounds from a DataFrame.
        Columns used if present: inchikey, chembl_id, name, smiles, synonyms, payload.
        """
        self.con.execute("DELETE FROM chembl_synonyms;")
        self.con.execute("DELETE FROM chembl_compounds;")
        for _, rec in df.iterrows():
            self.save_compound(rec.to_dict())

    def close(self):
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
