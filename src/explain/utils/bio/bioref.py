import polars as pl
from google.cloud import bigquery
from loguru import logger

from ._base_dti import BaseDTIClient, DTIResult


class BiorefClient(BaseDTIClient):
    """
    Bioref DTI client using ONLY standardized measurements:
      - bioref_std_measurement_unit must be 'nM'
      - bioref_std_measurement_value is used as the potency value
    Organism filter is applied ONLY if provided (organism is not required).
    """

    # binding_threshold is in nM
    def __init__(self, project: str = "datalake-dev-ea785c15", binding_threshold: float = 10000.0, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.table = f"{project}.about_a_molecule.kce_details_vw"
        self.binding_threshold = binding_threshold

    def get_pli(self, target_id: str, compound_id: str, organism: str | None = None, **kwargs) -> DTIResult:
        """
        Get binding score for target-compound pair with aggregation.

        Args:
            target_id: Protein accession ID, gene name, or UniProt name
            compound_id: InChI key or REC_ID
            organism: Organism filter (optional)

        Returns:
            DTIResult with aggregated binding information in nM
        """
        try:
            params = [
                bigquery.ScalarQueryParameter("target_id", "STRING", target_id),
                bigquery.ScalarQueryParameter("compound_id", "STRING", compound_id),
            ]

            where_parts = [
                "bioref_std_measurement_unit = 'nM'",
                "bioref_std_measurement_value IS NOT NULL",
                "(bioref_protein_accession_id = @target_id OR bioref_gene_name = @target_id OR bioref_protein = @target_id)",
                "(inchi_key = @compound_id OR rec_id = @compound_id)",
            ]

            # Add organism filter if provided
            if organism:
                where_parts.append("bioref_organism = @organism")
                params.append(bigquery.ScalarQueryParameter("organism", "STRING", organism))

            where_sql = " AND ".join(where_parts)

            sql = f"""
            WITH base AS (
              SELECT
                SAFE_CAST(bioref_std_measurement_value AS FLOAT64) AS value_nM,
                mce_biological_activity
              FROM `{self.table}`
              WHERE {where_sql}
            )
            SELECT
              COUNT(*) AS n_rows,
              COUNTIF(value_nM IS NOT NULL) AS n_obs,
              AVG(value_nM) AS mean_nM,
              APPROX_QUANTILES(value_nM, 100)[OFFSET(50)] AS median_nM,
              MIN(value_nM) AS best_nM,
              ARRAY_AGG(DISTINCT mce_biological_activity IGNORE NULLS) AS activities
            FROM base
            """

            client = bigquery.Client(project=self.project)
            result = list(client.query(sql, bigquery.QueryJobConfig(query_parameters=params)).result())

            if not result or result[0]["n_obs"] == 0:
                return DTIResult(
                    target_id=target_id,
                    compound_id=compound_id,
                    unit="nM",
                    method="bioref",
                    binding_type="unknown",
                    confidence=0.0,
                )

            r = result[0]
            median_nM = r["median_nM"]

            # Keep score as float64 in nM (don't convert to µM)
            score_nM = float(median_nM) if median_nM is not None else None

            # Simple confidence based on number of observations
            n_obs = r["n_obs"]
            confidence = min(float(n_obs) / 10.0, 1.0)

            # Get additional data for binding type determination
            binding_info_sql = f"""
            SELECT
                ARRAY_AGG(DISTINCT mce_biological_activity IGNORE NULLS) AS activities,
                ARRAY_AGG(DISTINCT bioref_measurement_type IGNORE NULLS) AS measurement_types
            FROM `{self.table}`
            WHERE {where_sql}
            """

            binding_info_result = list(
                client.query(binding_info_sql, bigquery.QueryJobConfig(query_parameters=params)).result()
            )
            binding_info = binding_info_result[0] if binding_info_result else {}

            # Determine binding type using both activities and measurement types
            binding_type = self._determine_binding_type(
                binding_info.get("activities", []) or [], binding_info.get("measurement_types", []) or []
            )
            # Binding prediction using nM threshold
            is_binding = score_nM is not None and score_nM <= self.binding_threshold

            return DTIResult(
                target_id=target_id,
                compound_id=compound_id,
                score=score_nM,
                unit="nM",
                method="bioref",
                binding_type=binding_type,
                confidence=confidence,
                is_binding=is_binding,
                raw_data={"median_nM": median_nM, "mean_nM": r["mean_nM"], "best_nM": r["best_nM"], "n_obs": n_obs},
            )

        except Exception as e:
            logger.error(f"Error querying bioref database: {e}")
            return DTIResult(
                target_id=target_id,
                compound_id=compound_id,
                unit="nM",
                method="bioref",
                binding_type="unknown",
                confidence=0.0,
            )

    def get_targets(self, compound_id: str, limit: int = 10, organism: str | None = None, **kwargs) -> pl.DataFrame:
        """
        Get targets for a compound as DataFrame with detailed information.

        Returns:
            DataFrame with columns: target_id, uniprot_id, gene_name, target_name, n_obs, median_nM, mean_nM, best_nM
        """
        try:
            params = [
                bigquery.ScalarQueryParameter("compound_id", "STRING", compound_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]

            where_parts = [
                "bioref_std_measurement_unit = 'nM'",
                "bioref_std_measurement_value IS NOT NULL",
                "(inchi_key = @compound_id OR rec_id = @compound_id)",
            ]

            # Add organism filter if provided
            if organism:
                where_parts.append("bioref_organism = @organism")
                params.append(bigquery.ScalarQueryParameter("organism", "STRING", organism))

            where_sql = " AND ".join(where_parts)

            sql = f"""
            WITH base AS (
              SELECT
                COALESCE(bioref_protein_accession_id, bioref_gene_name, bioref_protein) AS target_id,
                bioref_protein_accession_id AS uniprot_id,
                bioref_gene_name AS gene_name,
                bioref_protein_name AS target_name,
                SAFE_CAST(bioref_std_measurement_value AS FLOAT64) AS value_nM
              FROM `{self.table}`
              WHERE {where_sql}
            )
            SELECT
              target_id,
              ANY_VALUE(uniprot_id) AS uniprot_id,
              ANY_VALUE(gene_name) AS gene_name,
              ANY_VALUE(target_name) AS target_name,
              COUNT(*) AS n_obs,
              AVG(value_nM) AS mean_nM,
              APPROX_QUANTILES(value_nM, 100)[OFFSET(50)] AS median_nM,
              MIN(value_nM) AS best_nM
            FROM base
            WHERE target_id IS NOT NULL
            GROUP BY target_id
            ORDER BY median_nM ASC, n_obs DESC
            LIMIT @limit
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            return pl.from_arrow(job.result().to_arrow())

        except Exception as e:
            logger.error(f"Error getting targets dataframe for compound: {e}")
            return pl.DataFrame()

    def get_compounds(self, target_id: str, limit: int = 10, organism: str | None = None, **kwargs) -> pl.DataFrame:
        """
        Get compounds for a target as DataFrame with detailed information.

        Returns:
            DataFrame with columns: compound_id, n_obs, median_nM, mean_nM, best_nM, smiles
        """
        try:
            params = [
                bigquery.ScalarQueryParameter("target_id", "STRING", target_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]

            where_parts = [
                "bioref_std_measurement_unit = 'nM'",
                "bioref_std_measurement_value IS NOT NULL",
                "(bioref_protein_accession_id = @target_id OR bioref_gene_name = @target_id OR bioref_protein = @target_id)",
            ]

            # Add organism filter if provided
            if organism:
                where_parts.append("bioref_organism = @organism")
                params.append(bigquery.ScalarQueryParameter("organism", "STRING", organism))

            where_sql = " AND ".join(where_parts)

            sql = f"""
            WITH base AS (
              SELECT
                COALESCE(inchi_key, rec_id) AS compound_id,
                SAFE_CAST(bioref_std_measurement_value AS FLOAT64) AS value_nM,
                smiles
              FROM `{self.table}`
              WHERE {where_sql}
            )
            SELECT
              compound_id,
              COUNT(*) AS n_obs,
              AVG(value_nM) AS mean_nM,
              APPROX_QUANTILES(value_nM, 100)[OFFSET(50)] AS median_nM,
              MIN(value_nM) AS best_nM,
              ANY_VALUE(smiles) AS smiles
            FROM base
            WHERE compound_id IS NOT NULL
            GROUP BY compound_id
            ORDER BY median_nM ASC, n_obs DESC
            LIMIT @limit
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            return pl.from_arrow(job.result().to_arrow())

        except Exception as e:
            logger.error(f"Error getting compounds dataframe for target: {e}")
            return pl.DataFrame()

    def get_full_pli_info(
        self, target_id: str, compound_id: str, organism: str | None = None, limit: int = 100, **kwargs
    ) -> pl.DataFrame:
        """
        Get comprehensive DTI information as a clean DataFrame.

        Args:
            target_id: Target protein identifier
            compound_id: Compound identifier
            organism: Organism filter (optional)
            limit: Number of records to return

        Returns:
            Polars DataFrame with essential DTI information
        """
        try:
            params = [
                bigquery.ScalarQueryParameter("target_id", "STRING", target_id),
                bigquery.ScalarQueryParameter("compound_id", "STRING", compound_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]

            where_parts = [
                "(bioref_protein_accession_id = @target_id OR bioref_gene_name = @target_id OR bioref_protein = @target_id)",
                "(inchi_key = @compound_id OR rec_id = @compound_id)",
            ]

            # Add organism filter if provided
            if organism:
                where_parts.append("bioref_organism = @organism")
                params.append(bigquery.ScalarQueryParameter("organism", "STRING", organism))

            where_sql = " AND ".join(where_parts)

            # Get essential DTI information
            sql = f"""
            SELECT
              -- Identifiers
              inchi_key,
              rec_id,
              smiles,
              bioref_protein_accession_id as uniprot_id,
              bioref_gene_name as gene_name,
              bioref_protein_name as target_name,
              
              -- Original measurements (fallback)
              bioref_std_measurement_value as value,
              bioref_std_measurement_unit as unit,
              
              -- Assay context
              bioref_measurement_type as measurement_type,
              bioref_assay_type as assay_type,
              mce_biological_activity as activity,
              mce_research_area as research_area,
              mce_target as kce_targets,
              mce_pathway as pathway,
              general_kce as is_kce,
              
              -- Source tracking
              bioref_organism as organism,
              bioref_article_doi as doi,
              
            FROM `{self.table}`
            WHERE {where_sql}
            ORDER BY bioref_std_measurement_value ASC, bioref_original_measurement_value ASC
            LIMIT @limit
            """

            client = bigquery.Client(project=self.project)
            job = client.query(sql, bigquery.QueryJobConfig(query_parameters=params))
            return pl.from_arrow(job.result().to_arrow())

        except Exception as e:
            logger.error(f"Error fetching full PLI info: {e}")
            return pl.DataFrame()

    def _determine_binding_type(self, activities: list[str], measurement_types: list[str]) -> str:
        """
        Determine binding type from both biological activities and measurement types.

        Args:
            activities: mce_biological_activity values
            measurement_types: bioref_measurement_type values (e.g., IC50, EC50)

        Returns:
            Most specific binding type
        """
        all_indicators = activities + measurement_types

        if not all_indicators:
            return "binding"

        # Convert to lowercase for matching
        lower_indicators = [ind.lower() for ind in all_indicators if ind]

        # Mapping of measurement types to binding types
        measurement_type_mapping = {
            "IC50": "inhibitor",
            "pIC50": "inhibitor",
            "ki": "inhibitor",
            "pki": "inhibitor",
            "EC50": "agonist",
            "pEC50": "agonist",
            "KD": "binding",
            "pKD": "binding",
            "KA": "binding",
            "pKA": "binding",
        }

        # Check measurement types first (more specific)
        for measurement, binding_type in measurement_type_mapping.items():
            if any(measurement.lower() in ind.lower() for ind in lower_indicators):
                return binding_type

        # Fallback to activity-based detection
        activity_priority = [
            "inhibitor",
            "antagonist",
            "blocker",
            "agonist",
            "activator",
            "modulator",
            "binder",
            "binding",
        ]

        for binding_type in activity_priority:
            for indicator in lower_indicators:
                if binding_type in indicator:
                    return binding_type

        # Return first available indicator if no match
        return lower_indicators[0] if lower_indicators else "binding"
