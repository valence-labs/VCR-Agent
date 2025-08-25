import time

import requests
from loguru import logger
from pydantic import BaseModel

from explain.utils.bq import retrieve_from_bigquery
from explain.utils.chem.chembl import ChEMBLClient
from explain.utils.chem.mol_utils import standardize_smiles, to_inchikey
from explain.utils.chem.pubchem import PubChemClient

class BaseEntity(BaseModel):
    """Base class for biological entities."""

    id: str | None = None

    def retrieve_identifiers(self, **kwargs):
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
    id: str # ensembl id of the gene
    name: str | None = None
    ChEMBL: str | None = None
    Ensembl: str | None = None
    HGNC: str | None = None
    RefSeq: str | None = None
    HPA: str | None = None
    GeneID: str | None = None  # ncbi
    GeneSymbol: str | None = None  # ncbi
    UniprotAC: str | None = None
    UniprotID: str | None = None

    def _get_entity(self, ner_model):
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

    def retrieve_identifiers(self, ner_model: NERModel | None = None) -> "GeneEntity":
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
PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
SELECT DISTINCT ?compound_uri ?inchikey ?chembl_id ?pubchem_id 
       (SAMPLE(?smiles_) as ?smiles) 
       ?primary_name ?synonym ?formula ?molecular_weight ?max_phase
WHERE {
    {
        BIND(IRI(CONCAT(str(cbs:), "inchikey_", ?input_id)) AS ?compound_uri)
    } UNION {
        BIND(IRI(CONCAT(str(cbs:), "compound_", ?input_id)) AS ?compound_uri)
    }

    ?compound_uri a cbs:Compound .

    # Get InChIKey
    ?compound_uri cbs:hasId ?inchikey .

    # Get ChEMBL ID - handle both URI types
    BIND(
        IF(CONTAINS(str(?compound_uri), "compound_"),
           REPLACE(str(?compound_uri), str(cbs:compound_), ""),
           ""
        ) AS ?chembl_id
    )

    # Get SMILES (using SAMPLE to avoid duplicates)
    OPTIONAL { ?compound_uri cbs:hasCanonicalSmiles ?smiles_ }

    # Get primary name and synonyms
    OPTIONAL { ?compound_uri cbs:hasPrimaryName ?primary_name }
    OPTIONAL {
        ?compound_uri cbs:hasSynonym ?syn .
        ?syn cbs:hasValue ?synonym .
    }

    # Get PubChem ID
    OPTIONAL {
        ?compound_uri cbs:identifiedBy ?pubchem_assertion .
        ?pubchem_assertion cbs:hasValue ?pubchem_id ;
                         cbs:derivedBy cbs:standard_pubchem .
    }

    # Get additional properties
    OPTIONAL { ?compound_uri cbs:hasFormula ?formula }
    OPTIONAL { ?compound_uri cbs:hasMolecularWeight ?molecular_weight }
    OPTIONAL { ?compound_uri cbs:hasMaxClinicalTrialPhase ?max_phase }
}
GROUP BY ?compound_uri ?inchikey ?chembl_id ?pubchem_id ?primary_name 
         ?synonym ?formula ?molecular_weight ?max_phase
    dose: float | None = None

    def retrieve_identifiers(self) -> "CompoundEntity":
        """
        Retrieve all identifiers for the compound entity.

        Args:
            ner_model (NERModel, optional): Named entity recognition model.

        Returns:
            self
        """
        pubchem_client = PubChemClient()
        chembl_client = ChEMBLClient()
        
        if self.smiles:
            self.std_smiles = standardize_smiles(self.smiles)
            self.inchi_key = to_inchikey(self.std_smiles)
            iupac_result = pubchem_client.get_iupac_name(self.inchi_key)
            if isinstance(iupac_result, str):
                self.iupac_name = iupac_result
        elif self.name:
            response = pubchem_client.get_compound_info_by_synonym(self.name)
            if "error" in response and ner_model:
                response = self._get_entity(ner_model)
            if "error" not in response:
                self.smiles = response.get("SMILES")
                self.inchi_key = response.get("InChIKey")
                self.iupac_name = response.get("IUPACName")

        if self.inchi_key:
            # Get PubChem info
            pubchem_response = pubchem_client.get_compound_info_by_inchikey(self.inchi_key)
            if "error" not in pubchem_response:
                self.PubChem = pubchem_response.get("CID")
                chembl_id = pubchem_response.get("ChEMBL")
                if chembl_id:
                    self.ChEMBL = chembl_id[0] if isinstance(chembl_id, list) else chembl_id
            
            # Try to get ChEMBL info if not already found
            if not self.ChEMBL:
                chembl_response = chembl_client.get_compound_info(inchikey=self.inchi_key)
                if "error" not in chembl_response:
                    self.ChEMBL = chembl_response.get("chembl_id")
            
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
        pubchem_client = PubChemClient()
        return pubchem_client.get_compound_info_by_synonym(ner_name)

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
    rxrx_name: str | None = None
    std_name: str | None = None

    def _get_entity(self):
        """
        Placeholder for cell line mapping logic.

        Returns:
            str: Standardized cell type name.
        """
        # todo: add cell line mapping table.
        return self.name


class PerturbationEntity(BaseEntity):
    name: str | None = ""
    gkos: list[GeneEntity] | None = None
    compounds: list[CompoundEntity] | None = None
    cell_type: CellTypeEntity | None = None
    reference: str | None = None
    perturbation: str | None = None

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
