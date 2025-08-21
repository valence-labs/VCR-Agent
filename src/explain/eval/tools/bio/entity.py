import json
import re
import time
from abc import abstractmethod
from typing import Any, Optional, List, Union

import datamol as dm
import requests
from flair.data import Sentence
from flair.models import EntityMentionLinker
from flair.nn import Classifier
from loguru import logger
from pydantic import BaseModel

from explain.eval.tools.bio.utils import retrieve_from_bigquery
from explain.llm import create_client
from explain.llm._client import LLMClient


class NERModel(BaseModel):
    model_name: str
    model: Any = None

    @abstractmethod
    def _predict_gene(self, name: str):
        """Predict gene entity from input name."""
        pass

    @abstractmethod
    def _predict_compound(self, name: str):
        """Predict compound entity from input name."""
        pass


class FlairModel(NERModel):
    # todo: to be replaced
    model_name: str = "hunflair2"
    model: Any = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Flair model."""
        if self.model is None:
            self.model = Classifier.load(self.model_name)

    def _predict_gene(self, name: str):
        """
        Predict gene entity and return NCBI gene ID and gene symbol.

        Args:
            name (str): Gene name.

        Returns:
            tuple: (gene_id, gene_symbol)
        """
        sentence = Sentence(name)
        self.model.predict(sentence)

        gene_linker = EntityMentionLinker.load("gene-linker")
        gene_linker.predict(sentence)

        label_str = str(sentence.get_labels("link")[0])

        pattern = r'Span\[\d+:\d+\]: "(?P<query>[^"]+)" → (?P<ncbi_gene_id>\d+)/name=(?P<name>[^ ]+)'
        match = re.match(pattern, label_str)

        if match:
            gene_id = int(match.group("ncbi_gene_id"))
            gene_symbol = match.group("name")
            return gene_id, gene_symbol
        return None, None

    def _predict_compound(self, name: str):
        """
        Predict compound entity and return MeSH ID and compound name.

        Args:
            name (str): Compound name.

        Returns:
            tuple: (mesh_id, compound_name)
        """
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
        return None, None


class BaseEntity(BaseModel):
    """Base class for biological entities."""

    name: str

    def retrieve_identifiers(self, ner_model: Optional[NERModel] = None):
        """
        Retrieve identifiers for the entity.

        Args:
            ner_model (NERModel | None): Named entity recognition model.

        Returns:
            self
        """
        return self

    def compare_string(self, query: str) -> bool:
        """
        Compare a query string to the entity's attributes.

        Args:
            query (str): Query string.

        Returns:
            bool: True if query matches any attribute value, else False.
        """
        entity_dict = self.model_dump()
        return query in entity_dict.values()


class GeneEntity(BaseEntity):
    name: str
    ChEMBL: Optional[str] = None
    Ensembl: Optional[str] = None
    HGNC: Optional[str] = None
    RefSeq: Optional[str] = None
    HPA: Optional[str] = None
    GeneID: Optional[str] = None  # ncbi
    GeneSymbol: Optional[str] = None  # ncbi
    UniprotAC: Optional[str] = None
    UniprotID: Optional[str] = None

    def _get_entity(self, ner_model: NERModel):
        """
        Use NER model to identify gene ID and symbol.
        """
        gene_id, gene_symbol = ner_model._predict_gene(self.name)
        self.GeneID = gene_id
        self.GeneSymbol = gene_symbol

    def _ID_mapping(self, query_gene_id: str, query_id_type: str = "GeneID", tax_id: int = 9606):
        """
        Map gene identifiers using UniProt API.

        Args:
            query_gene_id (str): Gene identifier.
            query_id_type (str): Type of gene identifier (default: "GeneID").
            tax_id (int): Taxonomy ID (default: 9606 for human).
        """
        logger.info("ID mapping.")

        submit_url = "https://rest.uniprot.org/idmapping/run"
        submission = requests.post(
            submit_url,
            data={"from": query_id_type, "to": "UniProtKB-Swiss-Prot", "ids": query_gene_id, "taxId": tax_id},
        )
        job_id = submission.json().get("jobId")
        if not job_id:
            logger.error("Failed to submit UniProt mapping job.")
            return

        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        while True:
            status_res = requests.get(status_url).json()
            if status_res.get("jobStatus") == "COMPLETE" or "results" in status_res:
                break
            time.sleep(1)

        results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
        results = requests.get(results_url).json().get("results", [])

        if results:
            self.UniprotAC = results[0].get("to")
            entry_url = f"https://rest.uniprot.org/uniprotkb/{self.UniprotAC}"
            entry_data = requests.get(entry_url).json()
            self.UniprotID = entry_data.get("uniProtkbId")

            for row in entry_data.get("uniProtKBCrossReferences", []):
                db = row.get("database")
                if db == "ChEMBL":
                    self.ChEMBL = row.get("id")
                elif db == "Ensembl":
                    self.Ensembl = row.get("id")
                elif db == "HGNC":
                    self.HGNC = row.get("id")
                elif db == "RefSeq":
                    self.RefSeq = row.get("id")
                elif db == "HPA":
                    self.HPA = row.get("id")

    def retrieve_identifiers(self, ner_model: Optional[NERModel] = None) -> "GeneEntity":
        """
        Retrieve all identifiers for the gene entity using NER and UniProt mapping.

        Args:
            ner_model (NERModel | None): Named entity recognition model.

        Returns:
            self
        """
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

    def compare_entity(self, query: "GeneEntity") -> bool:
        """
        Compare this gene entity to another by GeneID.

        Args:
            query (BaseEntity): Another gene entity.

        Returns:
            bool: True if GeneID matches, else False.
        """
        # use `GeneID` as key ID
        return self.GeneID == query.GeneID


class CompoundEntity(BaseEntity):
    name: Optional[str] = None
    ChEMBL: Optional[str] = None
    PubChem: Optional[str] = None
    REC_ID: Optional[str] = None
    inchi_key: Optional[str] = None
    smiles: Optional[str] = None
    std_smiles: Optional[str] = None
    mesh_id: Optional[str] = None
    iupac_name: Optional[str] = None
    dose: Optional[float] = None

    def retrieve_identifiers(self, ner_model: Optional[NERModel] = None) -> "CompoundEntity":
        """
        Retrieve all identifiers for the compound entity.

        Args:
            ner_model (NERModel, optional): Named entity recognition model.

        Returns:
            self
        """
        if self.smiles:
            self.std_smiles = dm.standardize_smiles(self.smiles)
            self.inchi_key = dm.to_inchikey(self.std_smiles)
            self.iupac_name = get_iupac_name(self.inchi_key)
        elif self.name:
            response = get_compound_info_from_synonym(self.name)
            if "error" in response and ner_model:
                response = self._get_entity(ner_model)
            if "error" not in response:
                self.smiles = response.get("SMILES")
                self.inchi_key = response.get("InChIKey")
                self.iupac_name = response.get("IUPACName")

        if self.inchi_key:
            response = get_compound_info_from_inchikey(self.inchi_key)
            if "error" not in response:
                self.PubChem = response.get("CID")
                self.ChEMBL = response.get("ChEMBL")
            self._fetch_rxrx_id()

        return self

    def _get_entity(self, ner_model: NERModel):
        """
        Use NER model to identify compound MeSH ID and name, then fetch compound info.

        Args:
            ner_model (NERModel): Named entity recognition model.

        Returns:
            dict: Compound information.
        """
        self.mesh_id, ner_name = ner_model._predict_compound(self.name)
        return get_compound_info_from_synonym(ner_name)

    def _fetch_rxrx_id(self):
        """
        Fetch REC_ID for the compound from the RXRX datalake.
        """
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
            logger.info("Molecule is unavailable in RXRX datalake.")

    def _ID_mapping(self):
        """
        Placeholder for compound ID mapping logic.

        Raises:
            ValueError: If inchi_key is not set.
        """
        if not self.inchi_key:
            raise ValueError("Make sure to run 'CompoundEntity._get_entity()' first.")


class CellTypeEntity(BaseEntity):
    name: str
    rxrx_name: Optional[str] = None
    std_name: Optional[str] = None

    def _get_entity(self):
        """
        Placeholder for cell line mapping logic.

        Returns:
            str: Standardized cell type name.
        """
        # todo: add cell line mapping table.
        return self.name


class PerturbationEntity(BaseEntity):
    name: Optional[str] = ""
    gkos: Optional[List[GeneEntity]] = None
    compounds: Optional[List[CompoundEntity]] = None
    cell_type: Optional[CellTypeEntity] = None
    reference: Optional[str] = None
    perturbation: Optional[str] = None

    def model_post_init(self, __context__=None):
        """
        Generate perturbation string and reference based on gene knockouts and compounds.

        Args:
            __context__ (optional): Pydantic context (unused).
        """
        self.perturbation = ""
        self.reference = None
        if self.gkos and len(self.gkos) > 0:
            self.perturbation += "GENE:"
            gko_ids = [gko.GeneSymbol for gko in self.gkos if gko.GeneSymbol]
            self.perturbation += ";".join(gko_ids)
            self.reference = "centering-introns"

        if self.compounds and len(self.compounds) > 0:
            if self.perturbation:
                self.perturbation += "_"
            self.perturbation += "MOL:"
            compounds_ids = [
                f"{cpd.REC_ID}@{cpd.dose}" if cpd.dose else cpd.REC_ID
                for cpd in self.compounds if cpd.REC_ID
            ]
            self.perturbation += ";".join(compounds_ids)
            self.reference = "DMSO"


def get_compound_info_from_inchikey(inchikey: str):
    """
    Retrieve compound information from PubChem using an InChIKey.

    Args:
        inchikey (str): The InChIKey of the compound.

    Returns:
        dict: Dictionary with CID and ChEMBL ID, or error message.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    url = f"{base_url}/compound/inchikey/{inchikey}/xrefs/RegistryID/JSON"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            info = data["InformationList"]["Information"][0]
            CID = info.get("CID")
            all_ids = info.get("RegistryID", [])
            ChEMBL = [rid for rid in all_ids if rid.startswith("CHEMBL")]
            return {"CID": CID, "ChEMBL": ChEMBL[0] if len(ChEMBL) == 1 else ChEMBL}

        # Handle common errors gracefully
        elif response.status_code == 404:
            return {"error": f"Result: Identifier {inchikey} not found in PubChem."}
        else:
            return {"error": f"Received status code {response.status_code} from PubChem."}
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred during the API request: {e}"}


def get_iupac_name(inchikey: str) -> Union[str, dict]:
    """
    Fetch the IUPAC name for a given InChIKey from the PubChem PUG REST API.

    Args:
        inchikey (str): The InChIKey of the compound.

    Returns:
        str or dict: The IUPAC name if found, otherwise an error message.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/IUPACName/TXT"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text.strip()
        else:
            return {
                "error": f"Error: Failed to retrieve data. Status code: {response.status_code}\nResponse: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {"error": f"Error: A network error occurred: {e}"}


def get_compound_info_from_synonym(synonym: str):
    """
    Retrieve compound information from PubChem using a synonym.

    Args:
        synonym (str): A common name or synonym for the compound.

    Returns:
        dict: Dictionary with CID, SMILES, InChIKey, IUPACName, or error message.
    """
    # Step 1: Get the CID from the synonym
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{synonym}/cids/TXT"
    try:
        cid_response = requests.get(cid_url)
        if cid_response.status_code != 200:
            return {"error": f"Could not find a compound ID for '{synonym}'. Status: {cid_response.status_code}"}

        # The response text is the CID
        cid = cid_response.text.strip()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error during CID retrieval: {e}"}

    # Step 2: Get compound properties using the CID
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
            "SMILES": compound_properties.get("SMILES"),
            "InChIKey": compound_properties.get("InChIKey"),
            "IUPACName": compound_properties.get("IUPACName"),
        }
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        return {"error": f"Error retrieving or parsing compound data: {e}"}


def get_entity_type(input: str, client: Optional[LLMClient] = None) -> str:
    """
    Classify input text as 'compound', 'protein', or 'pathway' using an LLM.

    Args:
        input (str): Input text to classify.
        client (LLMClient, optional): LLM client instance.

    Returns:
        str: The predicted entity type ('compound', 'protein', or 'pathway').
    """
    if not client:
        client = create_client(provider="litellm", model="gemini-2.5-flash")

    llm_instruction = """
        You are an expert bioinformatics entity classifier. Your task is to analyze the input text and determine if it represents a `compound`, a `protein`, or a `pathway`.

        Follow these classification guidelines:
        1.  **compound**: Refers to a chemical substance, small molecule, or drug. Examples include "aspirin", "glucose", "ethanol", "metformin".
        2.  **protein**: Typically a gene name or its protein product. Often represented by a capitalized acronym that may include numbers. Examples include "MTOR1", "TP53", "EGFR", "AKT1".
        3.  **pathway**: Describes a biological process, a series of interactions, or a metabolic route. Often contains keywords like "signaling", "pathway", "cascade", "metabolism", or "cycle". Examples include "MTOR signaling", "Glycolysis", "Apoptosis", "p53 signaling pathway".

        Your output MUST be ONLY the single, lowercase word for the determined type: `compound`, `protein`, or `pathway`. Do not provide any explanation or additional text.

        ---
        **Examples:**

        Input: "aspirin"
        Output: "compound"

        Input: "MTOR1"
        Output: "protein"

        Input: "MTOR signaling"
        Output: "pathway"

        Input: "Krebs cycle"
        Output: "pathway"

        Input: "BRCA2"
        Output: "protein"
        ---

    """

    input_text = f"""

        **Perform the classification for the following input:**

        Input: "{input}"

    """

    content = llm_instruction + input_text

    res = client.generate([{"role": "user", "content": content}])

    return res.content


def compare_phenotypes(phenotype_1: str, phenotype_2: str, client: Optional[LLMClient] = None) -> Union[bool, str]:
    """
    Compare two phenotype descriptions and determine if they describe the same cellular phenotype.

    Args:
        phenotype_1 (str): First phenotype description.
        phenotype_2 (str): Second phenotype description.
        client (optional): LLM client instance.

    Returns:
        bool or str: True if phenotypes are equivalent, False if not, or LLM response otherwise.
    """
    if not client:
        client = create_client(provider="litellm", model="gemini-2.5-flash")

    llm_instruction = """
        Task: Decide whether the following two texts describe the same cellular phenotype, 
        even if they use different wording or synonyms. 
        Consider them the same if their meanings are equivalent. 
        Only return true or false without explanation.
    
    Instructions:

        Carefully read Text 1 and Text 2.

        Identify all terms related to cellular phenotypes, such as cell morphology, function, proliferation, death, or signaling.

        Compare the terms identified in both texts.

        Return true if the terms are synonymous or describe a cohesive set of features that represent the same cellular state or process.

        Return false if the terms describe distinct or unrelated cellular phenotypes.

        Examples: 

        Example 1: True

        Text 1: "Analysis revealed a significant increase in programmed cell death, leading to a reduction in cell count."

        Text 2: "The biopsy showed widespread apoptosis throughout the tissue."

        Output: true

        Example 2: False

        Text 1: "The cells displayed a highly disorganized cytoskeleton and impaired motility."

        Text 2: "Mitochondrial swelling and reduced ATP production were observed."

        Output: false
    """

    input_text = f"""

    **Perform the comparison for the following inputs:**

    Text 1: "{phenotype_1}"
    
    Text 2: "{phenotype_2}"

    """
    
    content = llm_instruction + input_text

    res = client.generate([{"role": "user", "content": content}])

    if isinstance(res, str):
        if res.strip().lower() == "false":
            return False
        elif res.strip().lower() == "true":
            return True
        else:
            return res
    elif hasattr(res, "content"):
        val = res.content.strip().lower()
        if val == "false":
            return False
        elif val == "true":
            return True
        else:
            return res.content
    else:
        return res
