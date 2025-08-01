from typing import Dict
from google.cloud import bigquery
import os
from pathlib import Path
import requests
import time
import json

from flair.models import EntityMentionLinker
from flair.nn import Classifier
from flair.data import Sentence


DATA_DIR_ROOT = Path("/rxrx/data/user/lu.zhu/outgoing/hooke-explain/Data")  # TO BE RELOCTED
ChEMBL_PATH = DATA_DIR_ROOT / "Binding" / "ChEMBL_35_Drug_Mechanism.csv"

GENE_ALIAS_SOURCES = ["ChEMBL", "Ensembl", "HGNC", "RefSeq", "GeneID", "RefSeq_Protein", "HPA"]  # TO BE EXTEDNED


def GetGeneEntity(entity: str) -> Dict:
    """
    Retrieves gene entity information for a given entity string.

    Attempts to obtain gene identifiers using `get_gene_ids`. If unsuccessful, it uses a gene entity linker
    to predict and extract gene information from the input. The function returns a dictionary containing
    the query, NCBI gene ID, and gene name if found, otherwise returns None.

    Args:
        entity (str): The entity string representing a gene.

    Returns:
        Dict or None: A dictionary with keys 'query', 'ncbi_gene_id', and 'name' if the gene entity is found,
        otherwise None.
    """
    entity_dict = get_gene_ids(entity)
    if not entity_dict:
        gene_linker = EntityMentionLinker.load("gene-linker")
        gene_linker.predict(sentence)

        # NCBI Gene

        label_str = str(sentence.get_labels("link")[0])

        pattern = r'Span\[\d+:\d+\]: "(?P<query>[^"]+)" → (?P<ncbi_gene_id>\d+)/name=(?P<name>[^ ]+)'
        match = re.match(pattern, s)
        if match:
            entity_dict = {
                "query": match.group("query"),
                "ncbi_gene_id": int(match.group("ncbi_gene_id")),
                "name": match.group("name"),
            }
        else:
            entity_dict = None
    return entity_dict


def get_gene_ids(gene_name) -> Dict:
    """Returns canonical and common aliases for a gene across multiple databases."""

    # The gene/protein identifier you start with"
    initial_id_type = "Gene_Name"  # e.g., 'Gene_Name', 'RefSeq_Protein', 'Ensembl'

    # --------------------------------------------------------------------------
    # STEP 1: Use the ID Mapping API to find potential UniProtKB Accessions
    # --------------------------------------------------------------------------

    # 1.1 Submit the mapping job
    submit_url = "https://rest.uniprot.org/idmapping/run"
    submission = requests.post(
        submit_url,
        data={"from": initial_id_type, "to": "UniProtKB-Swiss-Prot", "ids": gene_name, "taxId": 9606},
    )
    job_id = submission.json()["jobId"]

    # 1.2 Poll for job completion
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        status_res = requests.get(status_url).json()
        if "jobStatus" in status_res and status_res["jobStatus"] == "COMPLETE":
            break
        if "results" in status_res:  # For very fast jobs
            break
        time.sleep(1)

    # 1.3 Get the mapping results
    results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
    results = requests.get(results_url).json()

    # Make sure the result
    assert len(results) == 1

    uniprot_accession = results.get("results")[0]["to"]

    id_dict = {"uniProtkbAccession": uniprot_accession}
    # --------------------------------------------------------------------------
    # STEP 3: Use the UniProtKB API to get the full entry and all IDs
    # --------------------------------------------------------------------------

    entry_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}"
    entry_data = requests.get(entry_url).json()
    id_dict["uniProtkbId"] = entry_data["uniProtkbId"]

    # Parse and display the dbReferences
    if "uniProtKBCrossReferences" in entry_data:
        for row in entry_data["uniProtKBCrossReferences"]:
            for db in GENE_ALIAS_SOURCES:
                if row["database"] == db:
                    id_dict[db] = row["id"]

    return id_dict


def retrieve_from_bigquery(sql: str):
    """Retrive any data from datalake-prod-ef49c0c9 database"""
    client = bigquery.Client(project="datalake-prod-ef49c0c9")

    query_job = client.query(sql)
    results = query_job.to_dataframe()

    return results


def GetCompoundEntity(entity: str) -> dict:
    """
    Retrieves compound entity information for a given entity string using the chemical-linker model.

    Args:
        entity (str): The entity string to be analyzed.

    Returns:
        dict: The predicted compound entity information as returned by the chemical-linker model.
    """
    chemical_linker = EntityMentionLinker.load("chemical-linker")
    chemical_linker.predict(entity)

    label_str = str(sentence.get_labels("link")[0])
    pattern = r'Span\[\d+:\d+\]: "(?P<query>[^"]+)"[^>]*→ (?P<mesh>MESH:[^/]+)/name=(?P<name>[^(\n]+)'

    match = re.search(pattern, label_str)
    if match:
        return dict(query=match.group("query"), mesh_id=match.group("mesh"), name=match.group("name").strip())
