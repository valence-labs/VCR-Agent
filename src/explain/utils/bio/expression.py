import os

import anndata
import numpy as np
import pandas as pd
from diskcache import FanoutCache
from loguru import logger
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from explain.utils.bio._const import H5AD_PATH_DICT, PRECOMPUTED_FC
from explain.utils.bq import retrieve_from_bigquery

_EXPR_CACHE_DIR = os.path.expanduser("~/.expression_cache")
cache = FanoutCache(_EXPR_CACHE_DIR, timeout=2)


class GeneExpressionClient:
    """
    Simple client for gene expression analysis.
    Returns straightforward regulation results.
    """

    def __init__(self, pval_threshold: float = 0.05, log2fc_threshold: float = 1.0):
        self.pval_threshold = pval_threshold
        self.log2fc_threshold = log2fc_threshold

    def is_upregulated(self, gene: str, perturbation: str, reference: str = "DMSO", cell_type: str = "HUVEC") -> bool:
        """
        Check if a gene is upregulated under perturbation.

        Returns:
            True if gene is significantly upregulated, False otherwise
        """
        upregulated, _ = self.get_regulated_genes(perturbation, reference, cell_type)
        return gene in upregulated

    def is_downregulated(self, gene: str, perturbation: str, reference: str = "DMSO", cell_type: str = "HUVEC") -> bool:
        """
        Check if a gene is downregulated under perturbation.

        Returns:
            True if gene is significantly downregulated, False otherwise
        """
        _, downregulated = self.get_regulated_genes(perturbation, reference, cell_type)
        return gene in downregulated

    def get_regulated_genes(
        self, perturbation: str, reference: str = "DMSO", cell_type: str = "HUVEC"
    ) -> tuple[list[str], list[str]]:
        """
        Get lists of up/down regulated genes for a perturbation.

        Returns:
            Tuple of (upregulated_genes, downregulated_genes)
        """
        # Try multiple data sources
        de_results = self._get_de_results(perturbation, reference, cell_type)

        if de_results.empty:
            return [], []

        # Filter for significant results
        upregulated = de_results[
            (de_results["padj"] < self.pval_threshold) & (de_results["log2_foldchange"] > self.log2fc_threshold)
        ]["gene_id"].tolist()

        downregulated = de_results[
            (de_results["padj"] < self.pval_threshold) & (de_results["log2_foldchange"] < -self.log2fc_threshold)
        ]["gene_id"].tolist()

        return upregulated, downregulated

    def get_fold_change(
        self, gene: str, perturbation: str, reference: str = "DMSO", cell_type: str = "HUVEC"
    ) -> float | None:
        """
        Get log2 fold change for a specific gene.

        Returns:
            Log2 fold change if available, None if not found
        """
        de_results = self._get_de_results(perturbation, reference, cell_type)

        if de_results.empty:
            return None

        gene_result = de_results[de_results["gene_id"] == gene]

        if len(gene_result) > 0:
            return float(gene_result.iloc[0]["log2_foldchange"])

        return None

    @cache.memoize(expire=None, tag="de_call")
    def _get_de_results(self, perturbation: str, reference: str, cell_type: str) -> pd.DataFrame:
        """Get differential expression results from available sources."""

        # Try datalake first
        try:
            sql = f"""
                SELECT gene_id, log2_foldchange, padj
                FROM `trekseq_diffexp.trek_differentialexpression`
                WHERE reference_description LIKE '%{reference}%'
                AND test_description LIKE '{perturbation}%'
                AND UPPER(experiment_label) LIKE UPPER('%{cell_type}%')
                AND padj IS NOT NULL
            """

            result = retrieve_from_bigquery(sql)
            if len(result) > 0:
                return result

        except Exception as e:
            logger.debug(f"Datalake query failed: {e}")

        # Try precomputed files
        try:
            precomputed_path = PRECOMPUTED_FC.get(cell_type)
            if precomputed_path and precomputed_path.exists():
                df = pd.read_parquet(precomputed_path)
                result = df[
                    (df["reference_description"] == reference) & (df["test_description"].str.startswith(perturbation))
                ]
                if len(result) > 0:
                    return result

        except Exception as e:
            logger.debug(f"Precomputed data failed: {e}")

        # Last resort: compute on-demand (expensive)
        try:
            return self._compute_de_results(perturbation, reference, cell_type)
        except Exception as e:
            logger.warning(f"On-demand computation failed: {e}")
            return pd.DataFrame()

    def _compute_de_results(self, perturbation: str, reference: str, cell_type: str) -> pd.DataFrame:
        """Compute differential expression on-demand using DeSeq2."""

        h5ad_path = H5AD_PATH_DICT.get(cell_type)
        if not h5ad_path or not h5ad_path.exists():
            logger.warning(f"No expression data available for {cell_type}")
            return pd.DataFrame()

        try:
            adata = anndata.read_h5ad(h5ad_path)

            # Validate conditions exist
            available_conditions = set(adata.obs["condition"].unique())
            if not {perturbation, reference}.issubset(available_conditions):
                logger.warning(f"Conditions not found. Available: {available_conditions}")
                return pd.DataFrame()

            # Convert sparse matrix if needed
            if not isinstance(adata.X, np.ndarray):
                adata.X = adata.X.toarray()

            # Set up DeSeq2
            design = "~experiment_label + condition" if "experiment_label" in adata.obs else "~condition"

            dds = DeseqDataSet(
                adata=adata,
                design=design,
                quiet=True,
                refit_cooks=True,
                inference=DefaultInference(n_cpus=8),
            )
            dds.deseq2()

            # Get results
            stat_res = DeseqStats(
                dds,
                contrast=["condition", perturbation, reference],
                independent_filter=False,
                n_cpus=8,
            )
            stat_res.summary()

            de_results = stat_res.results_df
            de_results.rename(columns={"log2FoldChange": "log2_foldchange"}, inplace=True)
            de_results.reset_index(names="gene_id", inplace=True)

            return de_results

        except Exception as e:
            logger.error(f"DeSeq2 computation failed: {e}")
            return pd.DataFrame()
