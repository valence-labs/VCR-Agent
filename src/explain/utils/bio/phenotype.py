import re

import pandas as pd
from loguru import logger

from explain.utils.bio._const import (
    GENE_TO_PHENOTYPE,
    PHENOPRINT_DART_SCORE,
    PHENOPRINT_LOOKUP,
)


class PhenotypeClient:
    """
    Simple client for phenotype analysis.
    Returns straightforward phenotype associations.
    """

    def __init__(self):
        self.dart_threshold = 0.5

    def get_gene_phenotypes(self, gene_id: str) -> list[str]:
        """
        Get phenotypes associated with a gene.

        Args:
            gene_id: Gene identifier (e.g., Entrez ID)

        Returns:
            List of phenotype names
        """
        all_phenotypes = []

        for source_name, config in GENE_TO_PHENOTYPE.items():
            try:
                df = pd.read_table(config["path"])
                phenotypes = df[df[config["gene_id_col"]] == gene_id][config["phenotype_col"]].unique().tolist()
                all_phenotypes.extend(phenotypes)

            except Exception as e:
                logger.debug(f"Failed to load phenotypes from {source_name}: {e}")

        return list(set(all_phenotypes))  # Remove duplicates

    def has_phenotype(self, gene_id: str, phenotype: str) -> bool:
        """
        Check if a gene is associated with a specific phenotype.

        Args:
            gene_id: Gene identifier
            phenotype: Phenotype name to check

        Returns:
            True if association exists, False otherwise
        """
        gene_phenotypes = self.get_gene_phenotypes(gene_id)
        phenotype_lower = phenotype.lower()

        # Check for exact or partial matches
        for gene_phenotype in gene_phenotypes:
            if phenotype_lower in gene_phenotype.lower() or gene_phenotype.lower() in phenotype_lower:
                return True

        return False

    def has_phenoprint_effect(self, perturbation_name: str, cell_type: str = "HUVEC") -> bool:
        """
        Check if a perturbation has phenoprint effects.

        Args:
            perturbation_name: Name of the perturbation
            cell_type: Cell type context

        Returns:
            True if effects detected, False otherwise
        """
        try:
            # Check PH2-CP map
            phenoprint_path = PHENOPRINT_LOOKUP.get("PH2-CP")
            if not phenoprint_path:
                return False

            df = pd.read_parquet(phenoprint_path)
            pattern = rf"^{re.escape(perturbation_name)}(@[^;]+)?$"
            matches = df[df["perturbation_name"].str.match(pattern, na=False)]

            return not matches.empty

        except Exception as e:
            logger.debug(f"Error checking phenoprint effects: {e}")
            return False

    def has_therapeutic_potential(self, compound_rec_id: str, cell_type: str = "HUVEC") -> bool:
        """
        Check if compound has therapeutic potential using DART scores.

        Args:
            compound_rec_id: REC_ID of the compound
            cell_type: Cell type context

        Returns:
            True if therapeutic potential detected, False otherwise
        """
        try:
            dart_path = PHENOPRINT_DART_SCORE.get(cell_type)
            if not dart_path:
                return False

            df = pd.read_csv(dart_path)
            # Use default sigma=30
            score_col = "Compound Score (30 𝜎)"

            if score_col not in df.columns:
                return False

            compound_scores = df[df["Rec ID"] == compound_rec_id][score_col]

            if compound_scores.empty:
                return False

            max_score = compound_scores.max()
            return max_score > self.dart_threshold

        except Exception as e:
            logger.debug(f"Error checking DART scores: {e}")
            return False

    def get_dart_score(self, compound_rec_id: str, cell_type: str = "HUVEC") -> float | None:
        """
        Get DART score for a compound.

        Returns:
            DART score if available, None otherwise
        """
        try:
            dart_path = PHENOPRINT_DART_SCORE.get(cell_type)
            if not dart_path:
                return None

            df = pd.read_csv(dart_path)
            score_col = "Compound Score (30 𝜎)"

            if score_col not in df.columns:
                return None

            compound_scores = df[df["Rec ID"] == compound_rec_id][score_col]

            if compound_scores.empty:
                return None

            return float(compound_scores.max())

        except Exception as e:
            logger.debug(f"Error getting DART score: {e}")
            return None
