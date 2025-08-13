import re
import time
from abc import abstractmethod
from typing import Any, Optional

import requests
from flair.data import Sentence
from flair.models import EntityMentionLinker
from flair.nn import Classifier
from loguru import logger
from pydantic import BaseModel


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


class CompoundEntity(BaseEntity):
    name: str = None
    ChEMBL: str = None
    REC_ID: str = None
    inchi_key: str = None
    smiles: str = None

    def _get_entity(self):
        return self.smiles


class CellTypeEntity(BaseEntity):
    name: str
    rxrx_name: str = None
    std_name: str = None

    def _get_entity(self):
        # todo: add cell line mapping table.
        return self.name


class PerturbationEntity(BaseEntity):
    name: Optional[str] = ''
    gkos: Optional[list[GeneEntity]] = None
    compounds: Optional[list[CompoundEntity]] = None
    cell_type: Optional[CellTypeEntity] = None
    dose: Optional[float] = None

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
    