from pathlib import Path

import numpy as np
from google import genai

# todo: to be relocated
DATA_DIR_ROOT = Path("/rxrx/data/user/lu.zhu/outgoing/hooke-explain/Data")  # TO BE RELOCTED
ChEMBL_PATH = DATA_DIR_ROOT / "Binding" / "ChEMBL_35_Drug_Mechanism.csv"
H5AD_PATH_DICT = {"HUVEC": DATA_DIR_ROOT / "Expression/adata_test.h5ad"}
# todo: update the path dictionary

PRECOMPUTED_FC = {
    "HUVEC":  #  merged data from Tahor_deseq2, trekseq, pertubseq_deseq2 etc.
    DATA_DIR_ROOT / "Expression/precomputed_deseq2_test.parquet",
    # "cell type 2": [...]
}

GENE_TO_PHENOTYPE = {
    "HumanPhenotypeOntology": {
        "path": DATA_DIR_ROOT / "Phenotype/HumanPhenotypeOntology-genes_to_phenotype.txt",
        "gene_id_col": "ncbi_gene_id",
        "phenotype_col": "hpo_name",
    }
}


PHENOPRINT_LOOKUP = {
    "PH2-CP": DATA_DIR_ROOT / "Phenoprint/HUVEC-v1-Phenom-2-CP-2025-08-18_with-significant-effect.parquet",
    "PH2-BF": DATA_DIR_ROOT / "Phenoprint/HUVEC-v3-Phenom-2-BF-2025-06-26_with-significant-effect.parquet",
    "PH1": DATA_DIR_ROOT
    / "Phenoprint/HUVEC-tvn_v20_prox_bias_reduced-Phenom-1-2025-08-18_with-significant-effect.parquet",
    "DL2": DATA_DIR_ROOT / "Phenoprint/HUVEC-tvn_v20_prox_bias_reduced-DL2-2025-08-18_with-significant-effect.parquet",
}

TX_LOOKUP = {"TxFM": None}

PHENOPRINT_INFERENCE = {"HUVEC": DATA_DIR_ROOT / "Phenotype/phenotype_known_test.parquet"}

PHENOPRINT_SIMILARITY_MATRIX = {"HUVEC": DATA_DIR_ROOT / "Phenoprint/precomputed_cosim_matrix_test.parquet"}

PHENOPRINT_DART_SCORE = {"HUVEC": DATA_DIR_ROOT / "Phenoprint/DART/dart_huvec.csv"}


def get_llm_embeddings(text: str, client: genai.client.Client | None = None):
    if not client:
        client = genai.Client()
    embedding_response = client.models.embed_content(model="models/text-embedding-004", contents=text)
    return np.array(embedding_response.embeddings[0].values)
