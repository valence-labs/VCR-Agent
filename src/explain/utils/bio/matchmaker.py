import polars as pl
from google.cloud import bigquery
from loguru import logger

from ._base_dti import BaseDTIClient, DTIResult


class MatchMakerClient(BaseDTIClient):
    """
    Client for MatchMaker drug-target interaction scoring.
    Searches BOTH matchmaker_rxrx AND matchmaker_chembl tables with compound resolution.
    """

    def __init__(
        self, project: str = "datalake-prod-ef49c0c9", drop_duplicates: bool = True, binding_threshold: float = 2.0
    ):
        super().__init__(binding_threshold=binding_threshold)  # Z-score threshold (default)
        self.project = project
        self.mm_tables = [f"{project}.cyclica_90d.matchmaker_rxrx", f"{project}.cyclica_90d.matchmaker_chembl"]
        self.compounds_table = f"{project}.cauldron__cauldron.compounds"
        self.mm_score_threshold = 0.5  # mm_score likelihood threshold
        self.drop_duplicates = drop_duplicates

    def _get_table_name_short(self, table_path: str) -> str:
        """Extract short table name for source tracking."""
        if "rxrx" in table_path:
            return "rxrx"
        elif "chembl" in table_path:
            return "chembl"
        else:
            return "unknown"

    def _build_union_query(
        self, select_clause: str, where_clause: str, join_compound_table: bool = True, include_smiles: bool = False
    ) -> str:
        """Build UNION query across all MatchMaker tables."""
        union_parts = []

        for table in self.mm_tables:
            table_name = self._get_table_name_short(table)

            if join_compound_table and include_smiles:
                # Join with compounds table to get smiles
                query_part = f"""
                SELECT
                    {select_clause},
                    cm.smiles,
                    '{table_name}' as source_table
                FROM `{table}` mm
                INNER JOIN compound_matches cm ON mm.rec_id = cm.rec_id
                WHERE {where_clause}
                """
            elif join_compound_table:
                query_part = f"""
                SELECT
                    {select_clause},
                    '{table_name}' as source_table
                FROM `{table}` mm
                INNER JOIN compound_matches cm ON mm.rec_id = cm.rec_id
                WHERE {where_clause}
                """
            else:
                query_part = f"""
                SELECT
                    {select_clause},
                    '{table_name}' as source_table
                FROM `{table}` mm
                WHERE {where_clause}
                """

            union_parts.append(query_part)

        return " UNION ALL ".join(union_parts)

    def _get_target_variants(self, target_id: str) -> str:
        """
        Get target ID variants for matching (original and with _HUMAN suffix).

        Args:
            target_id: Original target identifier

        Returns:
            SQL condition to match both original and _HUMAN variant
        """
        if target_id.endswith("_HUMAN"):
            # Already has _HUMAN, just use as-is
            return f"mm.protein = '{target_id}'"
        else:
            # Try both original and with _HUMAN appended
            human_variant = f"{target_id}_HUMAN"
            return f"(mm.protein = '{target_id}' OR mm.protein = '{human_variant}')"

    def get_pli(self, target_id: str, compound_id: str, use_mm_score: bool = False, **kwargs) -> DTIResult:
        """
        Get MatchMaker score for a protein-compound pair from both RXRX and ChEMBL tables.

        Args:
            target_id: UniProt accession of the target protein
            compound_id: REC_ID, InChI key, SMILES, or compound name
            use_mm_score: If True use mm_score (likelihood), else z_score (default)

        Returns:
            DTIResult with best MatchMaker score from either table
        """
        try:
            params = [bigquery.ScalarQueryParameter("compound_id", "STRING", compound_id)]

            # Build compound matches CTE and union query
            select_clause = """
                mm.protein,
                mm.rec_id,
                mm.mm_score,
                mm.z_score
            """
            # Use target variants to handle _HUMAN suffix
            target_condition = self._get_target_variants(target_id)
            where_clause = target_condition
            union_query = self._build_union_query(select_clause, where_clause, join_compound_table=True)

            sql = f"""
            WITH compound_matches AS (
              SELECT rec_id
              FROM `{self.compounds_table}`
              WHERE rec_id = @compound_id
                 OR inchi_key = @compound_id
                 OR rdkit_inchikey = @compound_id
                 OR smiles = @compound_id
                 OR iupac_name = @compound_id
            )
            {union_query}
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            results = list(job.result())

            if results:
                # If multiple results (from both tables), use the best score
                if use_mm_score:
                    # For mm_score, higher is better (likelihood)
                    best_result = max(results, key=lambda x: x["mm_score"] or 0)
                    score = float(best_result["mm_score"])
                    threshold = self.mm_score_threshold
                    unit = "mm_score"
                    confidence = 1.0  # Always 1.0 for MatchMaker as requested
                    is_binding = score > threshold
                else:
                    # For z_score, higher absolute value is better (default)
                    best_result = max(results, key=lambda x: abs(x["z_score"] or 0))
                    score = float(best_result["z_score"])
                    threshold = self.default_threshold  # 2.0 for z_score
                    unit = "z_score"
                    confidence = 1.0  # Always 1.0 for MatchMaker as requested
                    is_binding = abs(score) > threshold

                return DTIResult(
                    target_id=target_id,
                    compound_id=compound_id,
                    score=score,
                    unit=unit,
                    method="matchmaker",
                    binding_type="binding",  # MatchMaker predicts general binding
                    confidence=confidence,
                    is_binding=is_binding,
                    raw_data={
                        "best_result": dict(best_result),
                        "all_results": [dict(r) for r in results],
                        "num_sources": len(results),
                    },
                )
            else:
                unit = "mm_score" if use_mm_score else "z_score"
                return DTIResult(
                    target_id=target_id,
                    compound_id=compound_id,
                    unit=unit,
                    method="matchmaker",
                    binding_type="binding",
                    confidence=1.0,  # Always 1.0 for MatchMaker as requested
                )

        except Exception as e:
            logger.warning(f"Failed to get MatchMaker score: {e}")
            unit = "mm_score" if use_mm_score else "z_score"
            return DTIResult(
                target_id=target_id,
                compound_id=compound_id,
                unit=unit,
                method="matchmaker",
                binding_type="binding",
                confidence=1.0,  # Always 1.0 for MatchMaker as requested
            )

    def get_targets(self, compound_id: str, limit: int = 10, use_mm_score: bool = False) -> pl.DataFrame:
        """
        Get top target proteins for a compound from both RXRX and ChEMBL tables.

        Args:
            compound_id: REC_ID, InChI key, SMILES, or compound name
            use_mm_score: If True rank by mm_score, else z_score (default)

        Returns:
            DataFrame with columns: target_id, mm_score, z_score, smiles, source_table
        """
        try:
            params = [
                bigquery.ScalarQueryParameter("compound_id", "STRING", compound_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]

            order_col = "mm_score DESC" if use_mm_score else "ABS(z_score) DESC"

            # Build union query dynamically from table list - include smiles from compounds table
            select_clause = """
                mm.protein as target_id,
                mm.mm_score,
                mm.z_score
            """
            where_clause = "TRUE"  # No additional WHERE needed for targets query
            union_query = self._build_union_query(
                select_clause, where_clause, join_compound_table=True, include_smiles=True
            )

            sql = f"""
            WITH compound_matches AS (
              SELECT rec_id, smiles
              FROM `{self.compounds_table}`
              WHERE rec_id = @compound_id
                 OR inchi_key = @compound_id
                 OR rdkit_inchikey = @compound_id
                 OR smiles = @compound_id
                 OR iupac_name = @compound_id
            )
            SELECT *
            FROM (
                {union_query}
            )
            ORDER BY {order_col}
            LIMIT @limit
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            df = pl.from_arrow(job.result().to_arrow())

            # Apply drop_duplicates if enabled
            if self.drop_duplicates and df.height > 0:
                df = df.unique(subset=["target_id"])

            return df

        except Exception as e:
            logger.error(f"Error getting targets for compound: {e}")
            return pl.DataFrame()

    def get_compounds(self, target_id: str, limit: int = 10, use_mm_score: bool = False) -> pl.DataFrame:
        """
        Get top compounds for a protein target from both RXRX and ChEMBL tables.

        Args:
            target_id: UniProt accession of the target protein
            use_mm_score: If True rank by mm_score, else z_score (default)

        Returns:
            DataFrame with columns: compound_id, mm_score, z_score, smiles, source_table
        """
        try:
            params = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]

            order_col = "mm_score DESC" if use_mm_score else "ABS(z_score) DESC"

            # Build union query with target variants and join with compounds for smiles
            select_clause = """
                rec_id as compound_id,
                mm_score,
                z_score
            """
            target_condition = self._get_target_variants(target_id)
            where_clause = target_condition
            union_query = self._build_union_query(select_clause, where_clause, join_compound_table=False)

            sql = f"""
            WITH combined_results AS (
                {union_query}
            )
            SELECT
                cr.*,
                c.smiles
            FROM combined_results cr
            LEFT JOIN `{self.compounds_table}` c ON cr.compound_id = c.rec_id
            ORDER BY {order_col}
            LIMIT @limit
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            df = pl.from_arrow(job.result().to_arrow())

            # Apply drop_duplicates if enabled
            if self.drop_duplicates and df.height > 0:
                df = df.unique(subset=["compound_id"])

            return df

        except Exception as e:
            logger.error(f"Error getting compounds for target: {e}")
            return pl.DataFrame()

    def get_full_pli_info(self, target_id: str, compound_id: str, use_mm_score: bool = False, **kwargs) -> pl.DataFrame:
        """
        Get comprehensive MatchMaker information from both RXRX and ChEMBL tables.

        Args:
            target_id: UniProt accession of the target protein
            compound_id: REC_ID, InChI key, SMILES, or compound name

        Returns:
            DataFrame with all MatchMaker predictions from both sources
        """
        try:
            params = [bigquery.ScalarQueryParameter("compound_id", "STRING", compound_id)]

            # Build union query with target variants and get smiles from compounds table
            select_clause = f"""
                mm.protein as target_id,
                mm.rec_id as compound_id,
                mm.mm_score,
                mm.z_score,
                ABS(mm.z_score) as abs_z_score,
                CASE
                    WHEN mm.mm_score > {self.mm_score_threshold} THEN 'likely_binding'
                    WHEN ABS(mm.z_score) >= {self.binding_threshold} THEN 'significant_z_score'
                    ELSE 'weak_prediction'
                END as prediction_category
            """
            target_condition = self._get_target_variants(target_id)
            where_clause = target_condition
            union_query = self._build_union_query(
                select_clause, where_clause, join_compound_table=True, include_smiles=True
            )

            sql = f"""
            WITH compound_matches AS (
              SELECT rec_id, smiles
              FROM `{self.compounds_table}`
              WHERE rec_id = @compound_id
                 OR inchi_key = @compound_id
                 OR rdkit_inchikey = @compound_id
                 OR smiles = @compound_id
                 OR iupac_name = @compound_id
            )
            {union_query}
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            df = pl.from_arrow(job.result().to_arrow())

            # Apply drop_duplicates if enabled (keep best score per target-compound pair)
            if self.drop_duplicates and df.height > 0:
                if use_mm_score:
                    # Keep row with highest mm_score per target-compound pair
                    df = df.sort("mm_score", descending=True).unique(subset=["target_id", "compound_id"])
                else:
                    # Keep row with highest abs(z_score) per target-compound pair
                    df = df.sort("abs_z_score", descending=True).unique(subset=["target_id", "compound_id"])

            return df

        except Exception as e:
            logger.error(f"Error getting full MatchMaker PLI info: {e}")
            return pl.DataFrame()
