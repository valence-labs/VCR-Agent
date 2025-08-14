import json
import re
import time
from abc import abstractmethod
from typing import Any

import datamol as dm
import requests
from flair.data import Sentence
from flair.models import EntityMentionLinker
from flair.nn import Classifier
from loguru import logger
from pydantic import BaseModel

from explain.eval.tools.bio.utils import retrieve_from_bigquery


class NERModel(BaseModel):
    model_name: str
    model: Any = None

    @abstractmethod
    def _predict_gene(self):
        return

    @abstractmethod
    def _predict_compound(self):
        return


class FlairModel(NERModel):
    # todo: to be replaced
    model_name: str = "hunflair2"
    model: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.model = Classifier.load(self.model_name)

    def _predict_gene(self, name: str):
        sentence = Sentence(name)
        self.model.predict(sentence)

        gene_linker = EntityMentionLinker.load("gene-linker")
        gene_linker.predict(sentence)

        label_str = str(sentence.get_labels("link")[0])

        pattern = r'Span\[\d+:\d+\]: "(?P<query>[^"]+)" → (?P<ncbi_gene_id>\d+)/name=(?P<name>[^ ]+)'
        match = re.match(pattern, label_str)

        if match:
            gene_id = int(match.group("ncbi_gene_id"))
            gene_symobol = match.group("name")

        return gene_id, gene_symobol

    def _predict_compound(self, name: str):
        sentence = Sentence(name)
        self.model.predict(sentence)

        chemical_linker = EntityMentionLinker.load("chemical-linker")
        chemical_linker.predict(sentence)

        label_str = str(sentence.get_labels("link")[0])
        pattern = r'Span\[\d+:\d+\]: "(?P<query>[^"]+)"[^>]*→ (?P<mesh>MESH:[^/]+)/name=(?P<name>[^(\n]+)'

        match = re.search(pattern, label_str)
        if match:
            mesh_id = match.group("mesh")
            chem_name = match.group("name").strip()

        return mesh_id, chem_name


class BaseEntity(BaseModel):
    """Base class for biological Entities."""

    name: str

    @abstractmethod
    def _get_entity(self):
        pass

    def retrieve_identifiers(self, ner_model: NERModel | None):
        return self

    def compare_string(self, query: str):
        entity_dict = self.model_dump()
        return query in entity_dict.values()


class GeneEntity(BaseEntity):
    name: str
    ChEMBL: str = None
    Ensembl: str = None
    HGNC: str = None
    RefSeq: str = None
    HPA: str = None
    GeneID: str = None  # ncbi
    GeneSymbol: str = None  # ncbi
    UniprotAC: str = None
    UniprotID: str = None

    def _get_entity(self, ner_model: NERModel):
        # todo: replace the llm NER model
        gene_id, gene_symobol = ner_model._predict_gene(self.name)
        self.GeneID = gene_id
        self.GeneSymbol = gene_symobol

    def _ID_mapping(self, query_gene_id: str, query_id_type: str = "GeneID", tax_id: int = 9606):
        logger.info("ID mapping.")

        # 1.1 Submit the mapping job
        submit_url = "https://rest.uniprot.org/idmapping/run"
        submission = requests.post(
            submit_url,
            data={"from": query_id_type, "to": "UniProtKB-Swiss-Prot", "ids": query_gene_id, "taxId": tax_id},
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
        results = requests.get(results_url).json().get("results")

        if len(results) > 0:
            self.UniprotAC = results[0]["to"]

            entry_url = f"https://rest.uniprot.org/uniprotkb/{self.UniprotAC}"
            entry_data = requests.get(entry_url).json()
            self.UniprotID = entry_data["uniProtkbId"]

            # Parse and display the dbReferences
            if "uniProtKBCrossReferences" in entry_data:
                for row in entry_data["uniProtKBCrossReferences"]:
                    # for db in GENE_ALIAS_SOURCES:
                    if row["database"] == "ChEMBL":
                        self.ChEMBL = row["id"]
                    elif row["database"] == "Ensembl":
                        self.Ensembl = row["id"]
                    elif row["database"] == "HGNC":
                        self.HGNC = row["id"]
                    elif row["database"] == "RefSeq":
                        self.RefSeq = row["id"]
                    elif row["database"] == "HPA":
                        self.HPA = row["id"]
                    # extend if neccessary

    def retrieve_identifiers(self, ner_model: NERModel | None = None) -> None:
        if not ner_model:
            ner_model = FlairModel()
            logger.info(f"NER model info: {ner_model.model_name}")
        # use llm to identify the gene id
        self._get_entity(ner_model)

        if not self.GeneID:
            raise ValueError("Make sure to run `GeneEntity._llm_get_entity` first.")

        # use ncbi gene id to get other identifier through uniprot mapping API
        self._ID_mapping(query_gene_id=self.GeneID, query_id_type="GeneID", tax_id=9606)
        return self

    def compare_entity(self, query: BaseEntity):
        # use `GeneID` as key ID
        return self.GeneID == query.GeneID


class CompoundEntity(BaseEntity):
    name: str = None
    ChEMBL: str = None
    PubChem: str = None
    REC_ID: str = None
    inchi_key: str = None
    smiles: str = None
    std_smiles: str = None
    mesh_id: str = None
    iupac_name: str = None

    def retrieve_identifiers(self, ner_model: NERModel = None):
        if self.smiles:
            self.std_smiles = dm.standardize_smiles(self.smiles)
            self.inchi_key = dm.to_inchikey(self.std_smiles)
            self.iupac_name = get_iupac_name(self.inchi_key)

        elif self.name:
            response = get_compound_info_from_synonym(self.name)
            if "error" in response and ner_model:
                response = self._get_entity(ner_model)

            if "error" not in response:
                self.smiles = response["SMILES"]
                self.inchi_key = response["InChIKey"]
                self.iupac_name = response["IUPACName"]

        # get chembl id
        response = get_compound_info_from_inchikey(self.inchi_key)
        if "error" not in response:
            self.PubChem = response["CID"]
            self.ChEMBL = response["ChEMBL"]

        if self.inchi_key:
            self._fetch_rxrx_id()

        return self

    def _get_entity(self, ner_model: NERModel):
        self.mesh_id, ner_name = ner_model._predict_compound(self.name)
        response = get_compound_info_from_synonym(ner_name)

        return response

    def _fetch_rxrx_id(self):
        logger.info(f"Fetching REC_ID for {self.inchi_key}")

        sql = f"""
            SELECT rec_id
            FROM cauldron__cauldron.compounds
            WHERE inchi_key = '{self.inchi_key}'
        """
        res = retrieve_from_bigquery(sql)

        if len(res) == 1:
            self.REC_ID = res.loc[0, "rec_id"]
        else:
            logger.info("Molecule is unavaiable in RXRX datalake.")

    def _ID_mapping(self):
        if not self.inchi_key:
            raise ValueError("Make sure to run 'CompoundEntity._get_entity()' first.")


class CellTypeEntity(BaseEntity):
    name: str
    rxrx_name: str = None
    std_name: str = None

    def _get_entity(self):
        # todo: add cell line mapping table.
        return self.name


class PerturbationEntity(BaseEntity):
    name: str | None = ""
    gkos: list[GeneEntity] | None = None
    compounds: list[CompoundEntity] | None = None
    cell_type: CellTypeEntity | None = None
    dose: float | None = None

    def _get_entity(self, source: str = "rxrx"):
        """
        Get the perturbation information and reference based on the data source.
        """
        if source == "rxrx":
            perturbation = ""
            if self.gkos:
                gko_ids = [gko.GeneSymbol for gko in self.gkos]
                perturbation += "-".join(gko_ids)
                # reference/control for gene perturbation
                reference = "centering-introns"

            if self.compounds and self.dose:
                compounds_ids = [cpd.REC_ID for cpd in self.compounds]
                perturbation += "_" + "-".join(compounds_ids) + "_" + self.dose
                # reference/control for compound perturbation
                reference = "DMSO"

            return perturbation, reference


def get_compound_info_from_inchikey(inchikey: str):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    url = f"{base_url}/compound/inchikey/{inchikey}/xrefs/RegistryID/JSON"

    try:
        response = requests.get(url, timeout=10)  # 10-second timeout
        data = response.json()

        # Check for successful request
        if response.status_code == 200:
            # CID
            CID = data["InformationList"]["Information"][0]["CID"]

            # other identifiers
            all_ids = data["InformationList"]["Information"][0]["RegistryID"]

            # get chembl ID
            ChEMBL = [rid for rid in all_ids if rid.startswith("CHEMBL")]

            return {"CID": CID, "ChEMBL": ChEMBL[0] if len(ChEMBL) == 1 else ChEMBL}

        # Handle common errors gracefully
        elif response.status_code == 404:
            return {"error": f"Result: Identifier {inchikey} not found in PubChem."}
        else:
            return {"error": f"Received status code {response.status_code} from PubChem."}

    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred during the API request: {e}"}


def get_iupac_name(inchikey: str) -> dict:
    """
    Fetches the IUPAC name for a given InChIKey from the PubChem PUG REST API.

    Args:
        inchikey (str): The InChIKey of the compound.

    Returns:
        str: The IUPAC name if found, otherwise an error message.
    """

    # Construct the API URL for a plain text response
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/IUPACName/TXT"

    try:
        # Send the GET request
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # The response content is the plain text IUPAC name
            return response.text.strip()
        else:
            # Handle potential errors (e.g., InChIKey not found)
            return {
                "error": f"Error: Failed to retrieve data. Status code: {response.status_code}\nResponse: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        return {"error": f"Error: A network error occurred: {e}"}


def get_compound_info_from_synonym(synonym: str):
    """
    Retrieves a compound's information from a synonym using the PubChem API.

    Args:
        synonym (str): A common name or synonym for the compound.

    Returns:
        dict: A dictionary containing key compound information (CID, name,
              formula, SMILES), or an error message.
    """
    # Step 1: Get the CID from the synonym
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{synonym}/cids/TXT"
    try:
        cid_response = requests.get(cid_url)
        if cid_response.status_code != 200:
            return {"error": f"Could not find a compound ID for '{synonym}'. Status: {cid_response.status_code}"}

        # The response text is the CID
        cid = cid_response.text.strip()
        print(f"Found CID for '{synonym}': {cid}")

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error during CID retrieval: {e}"}

    # Step 2: Get compound properties using the CID
    # We'll use the property endpoint for a clean JSON response
    props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES,InChIKey,IUPACName/JSON"
    try:
        props_response = requests.get(props_url)
        if props_response.status_code != 200:
            return {"error": f"Failed to get properties for CID {cid}. Status: {props_response.status_code}"}

        props_data = props_response.json()

        # Extract the relevant data from the JSON structure
        compound_properties = props_data["PropertyTable"]["Properties"][0]

        return {
            "CID": cid,
            "SMILES": compound_properties["SMILES"],
            "InChIKey": compound_properties["InChIKey"],
            "IUPACName": compound_properties["IUPACName"],
        }

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        return {"error": f"Error retrieving or parsing compound data: {e}"}
