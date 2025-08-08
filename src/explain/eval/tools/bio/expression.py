from typing import Any

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from google.cloud import bigquery
from loguru import logger
from pydantic import BaseModel, Field
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from explain.eval.tools._base import ToolVerifier


class GeneExpressionArgs(BaseModel):
    """Arguments for checking regulation expression."""

    source_entity: str = Field(description="The source entity causing regulation (e.g., compound, protein, pathway)")
    upregulated_genes: list[str] = Field(default=[], description="List of genes expected to be upregulated")
    downregulated_genes: list[str] = Field(default=[], description="List of genes expected to be downregulated")
    cell_type: str | None = Field(
        default=None, description="Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes)"
    )
    gkos: list[str] = Field(default=[], description="List of gene knockouts applied as perturbations")
    compounds: list[str] = Field(default=[], description="List of compound perturbations applied prior to measurement")
    dose: float | None = Field(default=None, description="Dose/concentration used in uM")
    # todo:
    # - check if there is perturbations with multiple compounds in different dosees
    # - unify the gene identifiers
    # - unify the compound identifiers
    # - unify the cell type identifiers
    # - unify and map reference , e.g. DMSO, centering-introns


class GeneExpressionVerifier(ToolVerifier):
    """Tool for checking if a source entity regulates gene expression in specific conditions.

    This tool verifies claims about gene regulation by querying knowledge bases
    and experimental databases, accounting for cellular context and perturbation conditions.
    """

    effect_pval = 0.05
    name = "check_gene_expression"
    description = "Check if a source entity regulates gene expression under specific conditions"
    args_schema = GeneExpressionArgs

    def _tool_logic(self, args: GeneExpressionArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking gene regulation.
        """
        if not args.upregulated_genes and not args.downregulated_genes:
            return 0.0, {"error": "At least one gene list (upregulated or downregulated) must be provided"}

        # get gene expression regulation
        upregulated_genes, downregulated_genes = self._expression_regulation_results(args)

        up_regulation_results = [gene in upregulated_genes for gene in args.upregulated_genes]
        down_regulation_results = [gene in downregulated_genes for gene in args.downregulated_genes]

        all_results = up_regulation_results + down_regulation_results
        reward = float(np.mean(all_results)) if all_results else 0.0

        feedback = {
            "source_entity": args.source_entity,
            "downregulated": {
                "requested": args.downregulated_genes,
                "results": down_regulation_results,
            },
            "upregulated": {
                "requested": args.upregulated_genes,
                "results": up_regulation_results,
            },
        }
        return reward, feedback

    def _process_perturbation(self, args):
        perturbation = ""
        if args.gkos:
            perturbation += "-".join(args.gkos)
            # reference/control for gene perturbation
            reference = "centering-introns"

        if args.compounds and args.dose:
            perturbation += "_" + "-".join(args.compounds) + "@" + args.dose
            # reference/control for compound perturbation
            reference = "DMSO"

        return perturbation, reference

    def _expression_regulation_results(self, args):
        # get gene_KO and reference
        perturbation, reference = self._process_perturbation(args)
        upregulated = []
        downregulated = []

        query_genes = args.upregulated_genes + args.downregulated_genes

        # rxrx precomputed
        up_reg, down_reg, presence_1 = self._expression_regulation(
            perturbation, reference, "rxrx_deseq2", args.cell_type, query_genes
        )
        upregulated.extend(up_reg)
        downregulated.extend(down_reg)

        # deseq2 precomputed
        upreg, downreg, presence_2 = self._expression_regulation(
            perturbation, reference, "deseq2_precomputed", args.cell_type, query_genes
        )
        upregulated.extend(up_reg)
        downregulated.extend(down_reg)

        # compyte deseq2
        presence = {g: presence_1[g] or presence_2[g] for g in query_genes}
        missing_genes = [g for g in query_genes if not presence[g]]
        # if there are genes does exist in any source
        if len(missing_genes) > 0:
            # the computing takes significant amount of time.
            # This part requires optimization.
            upreg, downreg, presence_3 = self._expression_regulation(
                perturbation, reference, "compute", args.cell_type, missing_genes
            )
            upregulated.extend(up_reg)
            downregulated.extend(down_reg)

            presence = {g: presence[g] or presence_3[g] for g in query_genes}
            missing_genes = [g for g in query_genes if not presence[g]]
        logger.info(f"Genes can not be found in the datasets: {missing_genes}.")

        return upregulated, downregulated

    def _expression_regulation(self, perturbation: str, reference: str, source: str, cell_type: str, query_genes: list):
        """
        Exame whether a perturbation of the source gene would regulate the expression of target gene(s).
        """
        upregulated_genes, downregulated_genes = [], []
        de_res = pd.DataFrame()
        presence = {g: False for g in query_genes}

        if source == "rxrx_deseq2":
            # get precomputed DE from rxrx precomputed
            de_res = get_precomputed_DE_datalake(perturbation, reference, cell_type)

        if source == "deseq2_precomputed":
            de_res = get_precomputed_DE_table(perturbation, reference, cell_type)

        if source == "compute":
            de_res = compute_DeSeq2(perturbation, reference, cell_type, query_genes)

        if de_res.shape[0] > 0:
            presence.update({g: g in de_res["gene_id"].unique() for g in query_genes})

            # get fc threshold based on fc distribution
            # we can use threshold
            # - default log2FC > 1 or log2FC< -1
            # log2fc_threshold = 1

            # - top K %
            # - dynamic threshold based on distribution.
            log2fc_threshold = get_log2fc_threshold(de_res, log2FC_col="log2_foldchange")
            logger.info(f"Apply log2FC threshold: {log2fc_threshold}")

            # get list of upregulated gene
            upregulated_genes = de_res.query(f"padj < {self.effect_pval} & log2_foldchange > {log2fc_threshold}")[
                "gene_id"
            ].unique()

            # get list of downregulated gene
            downregulated_genes = de_res.query(
                f"padj < {self.effect_pval} & log2_foldchange < {-1 * log2fc_threshold}"
            )["gene_id"].unique()

        return upregulated_genes, downregulated_genes, presence


def get_experiment_labels(cell_type: str = "HUVEC") -> list:
    # todo: update the cell line information
    EXPT_LABEL_DICT = {"HUVEC": ["250207-trek-1045-huvec-ipg-HSP90AB1-mimic_p1-a"]}

    return EXPT_LABEL_DICT.get(cell_type)


def get_precomputed_DE_datalake(perturbation: str, reference: str, cell_type: str = None) -> pd.DataFrame:
    """
    Retrieve DE data from RXRX DeSeq2 via SQL query
    """

    logger.info("Retrive DE data from DataLake")
    client = bigquery.Client(project="datalake-prod-ef49c0c9")

    # todo: get_experiment_labels for the specified cell_type
    # expt_labels = get_experiment_labels(cell_type)
    # specify the experiment with """AND experiment_label IN ('{"', '".join(expt_labels)}') """
    # check whether gene of interest exist precomputed database
    sql = f"""
        SELECT gene_id, log2_foldchange, padj FROM `trekseq_diffexp.trek_differentialexpression`
        WHERE reference_description LIKE '%{reference}%'
        AND test_description LIKE '{perturbation}%'
    """
    if cell_type:
        sql += f"   AND UPPER(experiment_label) LIKE UPPER('%{cell_type}%')"

    query_job = client.query(sql)  # Make an API request.

    de_res = query_job.to_dataframe()

    return de_res


def get_precomputed_DE_table(perturbation: str, reference: str, cell_type: str) -> pd.DataFrame:
    r"""
    Retrieve DE data from precomputed DeSeq2 FC from paquet file
    """
    # todo:
    #  - precomputing expression regulations for all the Tx data we have, rxrx and Tahor.)
    #  - update the data path dictionary after pre-computation
    logger.info("Retrive DE data from precomputed data sheets.")

    PRECOMPUTED_FC = {
        "HUVEC":  #  merged data from Tahor_deseq2, trekseq, pertubseq_deseq2 etc.
        "/rxrx/data/user/lu.zhu/outgoing/hooke-explain/Data/Expression/precomputed_deseq2_test.parquet",
        # "cell type 2": [...]
    }

    de_df = pd.read_parquet(PRECOMPUTED_FC.get(cell_type))
    # todo: update the file loading after deseq2 precomputation

    de_res = de_df[
        (de_df["reference_description"] == reference) & (de_df["test_description"].str.startswith(perturbation))
    ]
    return de_res


def get_log2fc_threshold(result, bin_index: int = 0, log2FC_col="log2_foldchange"):
    r"""
    Calculate the fold-change threshold for classifying differentially expressed genes.

    This function computes a threshold value for log2 fold-change based on the histogram of the provided data.
    The threshold is determined by selecting a bin index (`bin_index`) from the histogram of the specified column.

    Args:
        result (pd.DataFrame): DataFrame containing gene expression results, including a log2 fold-change column.
        fold_threshold (int, optional): Index of the histogram bin to use for threshold calculation. Defaults to 0.
        log2FC_col (str, optional): Name of the column containing log2 fold-change values. Defaults to "log2_foldchange".

    Returns:
        float: Calculated fold-change threshold value
    """

    foldp = np.histogram(result[log2FC_col].dropna())
    fc_threshold = (
        foldp[1][np.where(foldp[1] > 0)[0][bin_index]] + foldp[1][np.where(foldp[1] > 0)[0][bin_index + 1]]
    ) / 2
    return fc_threshold


def get_expression_data(cell_type: str) -> AnnData:
    # todo: update the path dictionary
    H5AD_PATH_DICT = {"HUVEC": "/rxrx/data/user/lu.zhu/outgoing/hooke-explain/Data/Expression/adata_test.h5ad"}

    h5ad_path = H5AD_PATH_DICT[cell_type]
    if h5ad_path:
        return anndata.read_h5ad(h5ad_path)
    return None


def compute_DeSeq2(
    perturbation: str, reference: str, cell_type: str, query_genes: list, n_cpus: int = 16
) -> pd.DataFrame:
    """
    Performs differential expression analysis between a perturbation condition and a reference condition.

    This function retrieves gene expression data, verifies that the specified conditions exist in the dataset,
    sets up the appropriate experimental design, and runs DESeq2 analysis to identify differentially expressed genes.

    Args:
        perturbation (str): The name of the perturbation condition to compare.
        reference (str): The name of the reference (control) condition.
        n_cpus (int, optional): Number of CPUs to use for parallel computation. Defaults to 16.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the differential expression analysis,
                    including statistics such as log fold change, p-values, and adjusted p-values.

    Raises:
        ValueError: If the specified conditions do not match those present in the dataset.
    """
    de_res = pd.DataFrame()

    # AnnData
    adata = get_expression_data(cell_type)

    if len(set(adata.var.index.tolist()).intersection(query_genes)) > 0:
        # double check the gene_ko and referecne in the dataset
        if set([perturbation, reference]) != set(adata.obs["condition"].unique()):
            raise ValueError(f"The dataset doesn't match the condition: {perturbation, reference}.")

        # assuming condition (perturnbation, reference) is already set in the dataset.
        if "experiment_label" in adata.obs:
            # assuming significant varience across experiments
            design = "~experiment_label * condition"
        else:
            design = "~condition"

        inference = DefaultInference(n_cpus=n_cpus)

        if not isinstance(adata.X, np.ndarray):
            X_array = adata.X.toarray()
            adata.X = None
            adata.X = X_array

        dds = DeseqDataSet(
            adata=adata,
            design=design,
            quiet=False,
            refit_cooks=True,
            inference=inference,
        )
        # run deseq2
        dds.deseq2()

        # return the regulation direction
        stat_res = DeseqStats(
            dds,
            contrast=["condition", perturbation, reference],
            independent_filter=False,
            n_cpus=n_cpus,
        )

        # get summary
        stat_res.summary()

        de_res = stat_res.results_df
        de_res.rename(columns={"log2FoldChange": "log2_foldchange"}, inplace=True)
        de_res.reset_index(names="gene_id", inplace=True)
    else:
        logger.info(f"Query genes ({query_genes}) are not in the GE datasets")
    return de_res


# Todo: add Harmonizome
# todo: add
