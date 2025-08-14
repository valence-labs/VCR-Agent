from pathlib import Path

from google.cloud import bigquery

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


def retrieve_from_bigquery(sql: str, as_dataframe: bool = True):
    """Retrive any data from datalake-prod-ef49c0c9 database"""
    client = bigquery.Client(project="datalake-prod-ef49c0c9")

    query_job = client.query(sql)
    if as_dataframe:
        results = query_job.to_dataframe()

    return results
