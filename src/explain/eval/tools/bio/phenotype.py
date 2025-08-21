import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from explain.eval.tools._base import ToolVerifier
from explain.eval.tools.bio.entity import (
    CellTypeEntity,
    CompoundEntity,
    GeneEntity,
    PerturbationEntity,
    compare_phenotypes,
    get_entity_type,
)
from explain.eval.tools.bio.utils import (
    GENE_TO_PHENOTYPE,
    PHENOPRINT_DART_SCORE,
    PHENOPRINT_INFERENCE,
    PHENOPRINT_LOOKUP,
    PHENOPRINT_SIMILARITY_MATRIX,
    get_llm_embeddings,
)


class PhenotypeArgs(BaseModel):
    """
    Arguments for checking phenotype.

    Attributes:
        source_entity (str): The source entity causing phenotype (e.g., compound, protein, pathway).
        phenotype (list[str]): The phenotypes to check for given the perturbation.
        direction (Literal["induce", "alleviate"]): Direction of phenotype change.
        cell_type (str | None): Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes).
        gkos (list[str]): List of gene knockouts applied as perturbations.
        compounds (list[str]): List of compound perturbations applied prior to measurement.
        dose (float | None): Dose/concentration used in uM.
        cell_type_entity (CellTypeEntity): Entity object for the cell type.
        gkos_entities (list): List of gene knockout entity objects.
        compounds_entities (list): List of compound entity objects.
        perturbation_entity (PerturbationEntity): Entity object for the perturbation.
    """

    source_entity: str = Field(description="The source entity causing phenotype (e.g., compound, protein, pathway)")
    phenotype: list[str] = Field(description="The phenotypes to check for given the perturbation")
    direction: Literal["induce", "alleviate"] = Field(
        description="Direction of phenotype change. Value: 'induce' or 'alleviate'."
    )
    cell_type: str | None = Field(
        default=None, description="Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes)"
    )

    gkos: list[str] = Field(default_factory=list, description="List of gene knockouts applied as perturbations")
    compounds: list[str] = Field(
        default_factory=list, description="List of compound perturbations applied prior to measurement"
    )
    dose: float | None = Field(default=None, description="Dose/concentration used in uM")

    cell_type_entity: CellTypeEntity = None
    gkos_entities: list = Field(default_factory=list)
    compounds_entities: list = Field(default_factory=list)
    perturbation_entity: PerturbationEntity = None

    def model_post_init(self, __context__=None):
        """
        Post-initialization to convert string fields to entity objects and
        populate entity-related attributes for downstream phenotype checking.
        """
        # Create entity for source_entity
        entity_type = get_entity_type(self.source_entity)
        name = self.source_entity
        if entity_type == "compound":
            self.source_entity = CompoundEntity(name=name)
        elif entity_type == "protein":
            self.source_entity = GeneEntity(name=name)
        elif entity_type == "pathway":
            pass  # Pathway entity handling can be added here

        self.source_entity.retrieve_identifiers()

        # Create entity for cell type
        self.cell_type_entity = CellTypeEntity(name=self.cell_type)


class PhenotypeVerifier(ToolVerifier):
    """
    Tool for checking if a source entity causes a specific phenotype in specific conditions.

    This tool verifies claims about phenotype by querying knowledge bases
    and experimental databases, accounting for cellular context and perturbation conditions.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool.
        args_schema (BaseModel): Pydantic schema for tool arguments.
        llm_client: Optional LLM client for semantic comparison.
        phenosim_threshold (float): Threshold for phenotype similarity.
        dart_score_threshold (float): Threshold for DART score.
    """

    name = "check_phenotype"
    description = "Check if a source entity causes a specific phenotype under specific conditions"
    args_schema = PhenotypeArgs
    phenosim_threshold = 0.4
    dart_score_threshold = 0.5

    def _tool_logic(self, args: PhenotypeArgs, llm_client=None) -> tuple[float, dict[str, Any]]:
        """
        Main logic for checking phenotype.

        Args:
            args (PhenotypeArgs): Arguments for phenotype checking.
            llm_client: Optional LLM client for semantic comparison.

        Returns:
            tuple[float, dict[str, Any]]: Reward score and feedback dictionary.
        """
        # Check the correctness of predicted phenotype
        if isinstance(args.source_entity, GeneEntity):
            retrieved_phenotypes = self._check_phenotype_for_genes(args)
        elif isinstance(args.source_entity, CompoundEntity):
            retrieved_phenotypes = self._check_phenotype_for_compound(args)

        phenotype_results = [
            self._compare_query_phenotypes(phenotype, retrieved_phenotypes, llm_client) for phenotype in args.phenotype
        ]

        # Check the correctness of change direction
        pheno_direction_bool = False
        if args.direction == "induce":
            pheno_direction_bool = self._check_phenoprint_change(args)
        elif args.direction == "alleviate":
            pheno_direction_bool = self._check_pheno_dart_score(args)

        reward = float(np.mean(phenotype_results)) if pheno_direction_bool else 0.0
        is_verified = any(phenotype_results) and pheno_direction_bool

        feedback = {
            "source_entity": args.source_entity,
            "phenotype": {
                "requested": args.phenotype,
                "results": retrieved_phenotypes,
            },
            "direction": {"requested": args.direction, "results": pheno_direction_bool},
            "verification_status": "VERIFIED" if is_verified else "NOT_VERIFIED",
        }
        return reward, feedback

    def _get_compound_moa(self, compound_id, cell_type):
        """
        Retrieve mechanism of action (MoA) gene targets for a compound in a given cell type.

        Args:
            compound_id (str): InChI key or identifier for the compound.
            cell_type (str): Cell type or cell line.

        Returns:
            np.ndarray: Array of unique MoA gene targets.
        """
        moa_df = pd.read_parquet(PHENOPRINT_INFERENCE.get(cell_type))
        moa_rows = moa_df.query("inchi_key == @compound_id")
        return moa_rows["MoA:target"].unique()

    def _check_phenotype_for_compound(self, args):
        """
        Check if a compound causes the queried phenotype(s) by MoA or phenoprint similarity.

        Args:
            args (PhenotypeArgs): Arguments for phenotype checking.

        Returns:
            tuple[list, np.ndarray | list]: Retrieved phenotypes and phenotype result booleans.
        """
        # Check if the MoA is available for compound; if so, get the genes and phenotypes
        targets = self._get_compound_moa(compound_id=args.source_entity.inchi_key, cell_type=args.cell_type)
        target_ent = [GeneEntity(name=tar).retrieve_identifiers() for tar in targets]
        target_ids = [ent.GeneID for ent in target_ent]

        retrieved_phenotypes = self._get_phenotypes_from_gene(target_ids) if len(targets) > 0 else []
        if not retrieved_phenotypes:
            # If not, get the most similar phenotypes if phenoprint is available
            retrieved_phenotypes = self._phenotype_inference(args)

        return retrieved_phenotypes

    def _fetch_consine_similarity(self, query_perturb_id, reference_perturb_ids, cell_type):
        """
        Fetch precomputed cosine similarity between a query perturbation and reference perturbations.

        Args:
            query_perturb_id (str): Query perturbation identifier.
            reference_perturb_ids (list[str]): List of reference perturbation identifiers.
            cell_type (str): Cell type or cell line.

        Returns:
            np.ndarray: Similarity scores.
        """
        cosim_mat = pd.read_parquet(PHENOPRINT_SIMILARITY_MATRIX.get(cell_type))
        return cosim_mat.loc[query_perturb_id, reference_perturb_ids].values

    def _phenotype_inference(self, args, topk: int = 10):
        """
        Infer phenotype similarity using LLM embeddings and precomputed phenoprints.

        Args:
            args (PhenotypeArgs): Arguments for phenotype checking.
            topk (int): Number of top similar phenotypes to consider.

        Returns:
            np.ndarray: Similarity scores for the top phenotypes.
        """
        pred_embedding = get_llm_embeddings(args.phenotype)
        df_known_phenotype = pd.read_parquet(PHENOPRINT_INFERENCE.get(args.cell_type))
        known_embeddings = np.stack(df_known_phenotype["phenotype_embedding"].values)

        # Add similarity scores to the dataframe
        df_known_phenotype["similarity_to_pred"] = cosine_similarity([pred_embedding], known_embeddings)[0]
        df_known_top10 = df_known_phenotype.nlargest(topk, "similarity_to_pred")

        # Build perturbation IDs for similarity lookup
        if isinstance(args.source_entity, GeneEntity):
            query_perturb_id = f"GENE:{args.source_entity.GeneID}"
            reference_perturb_ids = [f"GENE:{gene_name}" for gene_name in df_known_top10["MoA:target"]]
        elif isinstance(args.source_entity, CompoundEntity):
            query_perturb_id = f"MOL:{args.source_entity.REC_ID}"
            reference_perturb_ids = [f"MOL:{rec_id}" for rec_id in df_known_top10["rec_id"].unique()]
        else:
            return np.array([])

        similarities = self._fetch_consine_similarity(query_perturb_id, reference_perturb_ids, args.cell_type)
        df_known_top10["cosim_bool"] = similarities > self.phenosim_threshold

        return df_known_top10["phenotype"].unique()

    def _check_phenotype_for_genes(self, args):
        """
        Check if a gene causes the queried phenotype(s) by direct annotation or phenoprint similarity.

        Args:
            args (PhenotypeArgs): Arguments for phenotype checking.

        Returns:
            tuple[list, np.ndarray | list]: Retrieved phenotypes and phenotype result booleans.
        """
        retrieved_phenotypes = self._get_phenotypes_from_gene(args.source_entity.GeneID)
        if retrieved_phenotypes:
            phenotype_results = [
                self._compare_query_phenotypes(phenotype, retrieved_phenotypes, self.llm_client)
                for phenotype in args.phenotype
            ]
        else:
            # If not, use phenoprint inference
            similarities = self._phenotype_inference(args)
            phenotype_results = similarities > self.phenosim_threshold
        return retrieved_phenotypes, phenotype_results

    def _check_phenoprint_change(self, args: PhenotypeArgs, map_name: str | None = "PH2-CP"):
        """
        Check if the perturbation causes a phenoprint change using a lookup table.

        Args:
            args (PhenotypeArgs): Arguments for phenotype checking.
            map_name (str | None): Name of the phenoprint lookup map.

        Returns:
            bool: True if a phenoprint change is detected, False otherwise.
        """
        pert_name = args.perturbation_entity.perturbation
        pattern = rf"^{re.escape(pert_name)}(@[^;]+)?$"
        phenom_df = pd.read_parquet(PHENOPRINT_LOOKUP[map_name])
        matches = phenom_df[phenom_df["perturbation_name"].str.match(pattern, na=False)]
        return not matches.empty

    def _compare_query_phenotypes(self, query_phenotype: str, retrieved_phenotypes: list[str], llm_client=None):
        """
        Compare a queried phenotype with a list of retrieved phenotypes using semantic comparison.

        Args:
            query_phenotype (str): The phenotype to check.
            retrieved_phenotypes (list[str]): List of phenotypes retrieved from knowledge base.
            llm_client: Optional LLM client for semantic comparison.

        Returns:
            bool: True if any retrieved phenotype matches the query, False otherwise.
        """
        return any(compare_phenotypes(query_phenotype, retrieved, llm_client) for retrieved in retrieved_phenotypes)

    def _get_phenotypes_from_gene(self, gene_ids: list[str]):
        """
        Retrieve phenotypes associated with a gene from knowledge bases.

        Args:
            gene_id (str): Gene identifier.
            phenotype (list): List of phenotype queries.

        Returns:
            list: List of phenotypes associated with the gene.
        """
        all_phenotypes = []
        for _, val in GENE_TO_PHENOTYPE.items():
            df_gene_px = pd.read_table(val["path"])
            px = df_gene_px[df_gene_px[val["gene_id_col"]].isin(gene_ids)][val["phenotype_col"]].unique().tolist()
            all_phenotypes.extend(px)
        return all_phenotypes

    def _check_pheno_dart_score(self, args, sigma=30):
        """
        Retrieve the DART scores of the query compound and check if it exceeds the threshold.

        Args:
            args (PhenotypeArgs): Arguments for phenotype checking.
            sigma (int): Sigma value for DART score column.

        Returns:
            bool: True if any DART score exceeds the threshold, False otherwise.
        """
        compound_id = args.source_entity.REC_ID
        df_dart = pd.read_csv(PHENOPRINT_DART_SCORE.get(args.cell_type))
        dart_scores = df_dart.loc[df_dart["Rec ID"] == compound_id, f"Compound Score ({sigma} 𝜎)"]
        return (dart_scores > self.dart_score_threshold).any()
