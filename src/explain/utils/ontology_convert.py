from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DATASETS = [
    {"id": "affy_probeset", "name": "Affymetrix probeset", "category": "Probe"},
    {"id": "assembly_insdc", "name": "Assembly INSDC", "category": "Genome"},
    {"id": "assembly_refseq", "name": "Assembly RefSeq", "category": "Genome"},
    {"id": "atc", "name": "ATC classification", "category": "Drug"},
    {"id": "bioproject", "name": "BioProject", "category": "Project"},
    {"id": "bioproject_umbrella", "name": "BioProject umbrella", "category": "Project"},
    {"id": "biosample", "name": "BioSample", "category": "Sample"},
    {"id": "ccds", "name": "Consensus CDS", "category": "Gene"},
    {"id": "cellosaurus", "name": "Cellosaurus", "category": "CellLine"},
    {"id": "chebi", "name": "ChEBI compound", "category": "Compound"},
    {"id": "chembl_compound", "name": "ChEMBL compound", "category": "Compound"},
    {"id": "chembl_target", "name": "ChEMBL target", "category": "Protein"},
    {"id": "clinvar", "name": "ClinVar variant", "category": "Variant"},
    {"id": "cog", "name": "COG", "category": "Ortholog"},
    {"id": "dbsnp", "name": "dbSNP", "category": "Variant"},
    {"id": "doid", "name": "Disease ontology", "category": "Phenotype"},
    {"id": "drugbank", "name": "DrugBank", "category": "Drug"},
    {"id": "ec", "name": "Enzyme nomenclature", "category": "Function"},
    {"id": "efo_disease", "name": "EFO disease", "category": "Phenotype"},
    {"id": "ensembl_gene", "name": "Ensembl gene", "category": "Gene"},
    {"id": "ensembl_protein", "name": "Ensembl protein", "category": "Protein"},
    {"id": "ensembl_transcript", "name": "Ensembl transcript", "category": "Transcript"},
    {"id": "gea", "name": "GEA", "category": "Experiment"},
    {"id": "geo_sample", "name": "GEO sample", "category": "Sample"},
    {"id": "geo_series", "name": "GEO series", "category": "Project"},
    {"id": "glycomotif", "name": "GlycoMotif", "category": "Glycan"},
    {"id": "glytoucan", "name": "GlyTouCan", "category": "Glycan"},
    {"id": "go", "name": "Gene ontology", "category": "Function"},
    {"id": "hgnc", "name": "HGNC", "category": "Gene"},
    {"id": "hgnc_symbol", "name": "HGNC gene symbol", "category": "Gene, Synonym"},
    {"id": "hmdb", "name": "HMDB", "category": "Compound"},
    {"id": "homologene", "name": "HomoloGene", "category": "Ortholog"},
    {"id": "hp_inheritance", "name": "HP Inheritance", "category": "Classification"},
    {"id": "hp_phenotype", "name": "HP Phenotype", "category": "Phenotype"},
    {"id": "inchi_key", "name": "InChIKey", "category": "Compound"},
    {"id": "insdc", "name": "GenBank/ENA/DDBJ", "category": "Gene"},
    {"id": "insdc_master", "name": "INSDC master", "category": "Project"},
    {"id": "intact", "name": "IntAct", "category": "Interaction"},
    {"id": "interpro", "name": "InterPro", "category": "Domain"},
    {"id": "iuphar_ligand", "name": "IUPHAR ligand", "category": "Compound"},
    {"id": "jga_dataset", "name": "JGA Dataset", "category": "Project"},
    {"id": "jga_study", "name": "JGA Study", "category": "Project"},
    {"id": "lipidmaps", "name": "LIPID MAPS", "category": "Lipid"},
    {"id": "lncbook_gene", "name": "LncBook gene", "category": "Gene"},
    {"id": "lrg", "name": "LRG", "category": "Gene"},
    {"id": "mbgd_gene", "name": "MBGD gene", "category": "Gene"},
    {"id": "mbgd_organism", "name": "MBGD organism", "category": "Organism"},
    {"id": "meddra", "name": "MedDRA", "category": "Phenotype"},
    {"id": "medgen", "name": "MedGen", "category": "Phenotype"},
    {"id": "mesh", "name": "MeSH", "category": "Phenotype"},
    {"id": "mgi_allele", "name": "MGI allele", "category": "Variant"},
    {"id": "mgi_gene", "name": "MGI gene", "category": "Gene"},
    {"id": "mgi_genotype", "name": "MGI genotype", "category": "Genotype"},
    {"id": "mirbase", "name": "miRBase", "category": "Transcript"},
    {"id": "mirbase_mature", "name": "miRBase mature", "category": "Transcript"},
    {"id": "mondo", "name": "MONDO", "category": "Phenotype"},
    {"id": "mp", "name": "Mammalian Phenotype Ontology", "category": "Phenotype"},
    {"id": "nando", "name": "NANDO", "category": "Phenotype"},
    {"id": "nbdc_human_db", "name": "NBDC Human Database", "category": "Project"},
    {"id": "ncbigene", "name": "NCBI Gene", "category": "Gene"},
    {"id": "ncit_disease", "name": "NCIt disease", "category": "Phenotype"},
    {"id": "ncit_tissue", "name": "NCIt tissue", "category": "Anatomy"},
    {"id": "oma_group", "name": "OMA group", "category": "Ortholog"},
    {"id": "oma_protein", "name": "OMA protein", "category": "Protein"},
    {"id": "omim_gene", "name": "OMIM gene", "category": "Gene"},
    {"id": "omim_phenotype", "name": "OMIM phenotype", "category": "Phenotype"},
    {"id": "orphanet_gene", "name": "Orphanet gene", "category": "Gene"},
    {"id": "orphanet_phenotype", "name": "Orphanet phenotype", "category": "Phenotype"},
    {"id": "pathbank", "name": "Pathbank", "category": "Pathway"},
    {"id": "pdb", "name": "PDB", "category": "Structure"},
    {"id": "pdb_ccd", "name": "PDB CCD", "category": "Compound"},
    {"id": "pfam", "name": "Pfam", "category": "Domain"},
    {"id": "pmc", "name": "PMC", "category": "Literature"},
    {"id": "prosite", "name": "PROSITE", "category": "Domain"},
    {"id": "prosite_prorule", "name": "PROSITE ProRule", "category": "AnnotationRule"},
    {"id": "pubchem_compound", "name": "PubChem compound", "category": "Compound"},
    {"id": "pubchem_pathway", "name": "PubChem pathway", "category": "Pathway"},
    {"id": "pubchem_substance", "name": "PubChem substance", "category": "Compound"},
    {"id": "pubmed", "name": "PubMed", "category": "Literature"},
    {"id": "reactome_pathway", "name": "Reactome pathway", "category": "Pathway"},
    {"id": "reactome_reaction", "name": "Reactome reaction", "category": "Reaction"},
    {"id": "refseq_genomic", "name": "RefSeq genomic", "category": "Gene"},
    {"id": "refseq_protein", "name": "RefSeq protein", "category": "Protein"},
    {"id": "refseq_rna", "name": "RefSeq RNA", "category": "Transcript"},
    {"id": "rfam", "name": "Rfam", "category": "Transcript"},
    {"id": "rgd", "name": "RGD", "category": "Gene"},
    {"id": "rhea", "name": "Rhea", "category": "Reaction"},
    {"id": "rnacentral", "name": "RNAcentral", "category": "Transcript"},
    {"id": "sgd", "name": "SGD", "category": "Gene"},
    {"id": "sio", "name": "SIO", "category": "Classification"},
    {"id": "smart", "name": "SMART", "category": "Domain"},
    {"id": "sra_accession", "name": "SRA accession", "category": "Submission"},
    {"id": "sra_analysis", "name": "SRA analysis", "category": "Analysis"},
    {"id": "sra_experiment", "name": "SRA experiment", "category": "Experiment"},
    {"id": "sra_project", "name": "SRA project", "category": "Project"},
    {"id": "sra_run", "name": "SRA run", "category": "SequenceRun"},
    {"id": "sra_sample", "name": "SRA sample", "category": "Sample"},
    {"id": "swisslipids", "name": "SwissLipids", "category": "Lipid"},
    {"id": "tair", "name": "TAIR", "category": "Gene"},
    {"id": "taxonomy", "name": "Taxonomy", "category": "Organism"},
    {"id": "togovar", "name": "TogoVar variant", "category": "Variant"},
    {"id": "uberon", "name": "UBERON", "category": "Anatomy"},
    {"id": "uniprot", "name": "UniProt", "category": "Protein"},
    {"id": "uniprot_mnemonic", "name": "UniProt mnemonic", "category": "Protein, Synonym"},
    {"id": "uniprot_proteome", "name": "UniProt proteome", "category": "Proteome"},
    {"id": "vgnc", "name": "VGNC", "category": "Gene"},
    {"id": "wikipathways", "name": "WikiPathways", "category": "Pathway"},
]


class TogoIDError(RuntimeError):
    """Raised when TogoID returns an error or an unexpected payload."""


@dataclass
class TogoIDResponse:
    ids: list[str]  # input ids echoed back
    route: list[str]  # normalized route actually executed
    result: list[Any]  # output (format depends on 'report')


class TogoIDClient:
    """
    Minimal client for TogoID's /convert endpoint.

    GET /convert
      - ids    : comma-separated source IDs
      - route  : comma-separated datasets (src..dst), e.g. "mondo,mesh"
      - report : 'target' | 'pair' | 'all' | 'full'  (default 'target')
      - format : 'json' (this client only supports JSON)
      - limit/offset supported by API; for large lists use convert_batched()
    """

    BASE_URL = "https://api.togoid.dbcls.jp/convert"

    VALID_REPORTS: set[str] = {"pair", "full"}

    def __init__(
        self,
        timeout: int = 15,
        max_per_call: int = 10_000,
        session: requests.Session | None = None,
    ):
        self.base_url = self.BASE_URL.rstrip("/")
        self.timeout = timeout
        self.max_per_call = max_per_call
        self.session = session or requests.Session()
        self.known_datasets = pd.DataFrame(DATASETS)
        self._known_dataset_ids = set(self.known_datasets["id"])

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "TogoIDClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _normalize_route(self, route: list[str] | str) -> tuple[list[str], list[str]]:
        """
        Normalize a route spec into a list of known dataset ids and collect unknowns.

        Returns (normalized_route, unknown_datasets)
        """
        if isinstance(route, str):
            route_list = [s.strip() for s in route.split(",") if s.strip()]
        else:
            route_list = [str(s).strip() for s in route if str(s).strip()]

        route_norm = [r for r in route_list if r in self._known_dataset_ids]
        unknown = [r for r in route_list if r not in self._known_dataset_ids]
        return route_norm, unknown

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.8, min=0.8, max=6.0),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _get_json(self, params: dict[str, str | int]) -> dict[str, Any]:
        r = self.session.get(self.base_url, params=params, timeout=self.timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # Surface server/client message
            msg = r.text
            raise TogoIDError(f"TogoID HTTP {r.status_code}: {msg}") from e

        try:
            return r.json()
        except ValueError as e:
            raise TogoIDError("TogoID returned non-JSON response") from e

    def convert(
        self,
        ids: list[str],
        route: list[str] | str,
        *,
        report: str = "pair",  # or 'pair' | 'full'
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        """Convert a list of IDs along a dataset route."""

        # Normalize route
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(route, str):
            route = [route]
        route_norm, _ = self._normalize_route(route)
        if _:
            logger.warning(f"Unknown datasets in route: {_}")
        # Validate report
        if report not in self.VALID_REPORTS:
            raise ValueError(f"Invalid report '{report}'. Must be one of {sorted(self.VALID_REPORTS)}")

        norm_ids = [str(i).strip() for i in ids if i is not None and str(i).strip()]
        if not norm_ids:
            return None

        params: dict[str, str | int] = {
            "ids": ",".join(norm_ids),
            "route": ",".join(route_norm),
            "report": report,
            "format": "json",
        }
        if limit is not None:
            params["limit"] = int(limit)
        if offset is not None:
            params["offset"] = int(offset)

        data = self._get_json(params)
        columns = [data["route"][0], data["route"][1]] if report == "pair" else data["route"]
        return pd.DataFrame(data["results"], columns=columns)

    def convert_batched(
        self,
        ids: list[str],
        route: list[str] | str,
        *,
        report: str = "target",
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Split large ID lists into chunks (respecting API max_per_call).
        Returns a list of TogoIDResponse, one per batch.
        """
        norm_ids = [str(i).strip() for i in ids if i is not None and str(i).strip()]
        if not norm_ids:
            return None

        # Determine chunk size honoring both provided batch_size and client max_per_call
        effective_chunk_size = self.max_per_call if batch_size is None else min(self.max_per_call, int(batch_size))
        if effective_chunk_size < 1:
            raise ValueError("batch_size must be >= 1")

        n_batches = len(norm_ids) // effective_chunk_size
        out: list[pd.DataFrame] = []
        for b in range(n_batches):
            start = b * effective_chunk_size
            end = (b + 1) * effective_chunk_size
            chunk = norm_ids[start:end]
            out.append(self.convert(chunk, route, report=report))
        return pd.concat(out)
