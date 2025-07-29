from io import StringIO
from typing import List, Union

import pandas as pd
import requests


class GraphQueries:
    """
    Responsible for sending SPARQL queries to the graph database through
    the REST API.

    Attributes:
        URI (str): The base URI for the API endpoints.
        HEADERS (dict): Default headers included in API requests.
    """

    URL = "https://graphdb-ingress-test.centaur-platform-dev.com/" "repositories/cb-head"
    HEADERS = {
        "Content-Type": "application/sparql-query",
        "Accept": "text/tab-separated-values",
    }

    @staticmethod
    def post_request(payload: str) -> pd.DataFrame:
        """
        Makes a POST request to the graph database with the given SPARQL query.

        Args:
            payload (str): The SPARQL query to be sent to the graph database.

        Returns:
            pd.DataFrame: The response from the graph database as a DataFrame.
        """
        response = requests.post(GraphQueries.URL, data=payload, headers=GraphQueries.HEADERS, timeout=600)
        return pd.read_csv(StringIO(response.text), sep="\t")

    @staticmethod
    def query_gene_pathway(gene_list: list[str], exclude_gene_symbol: Union[str, None] = None) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find pathways for a given list of genes
        and other genes within those pathways, with optional exclusion.

        Args:
            gene_list: A list of gene symbols to query for.
            exclude_gene_symbol: An optional gene symbol to exclude from the
                                ?other_gene_symbol results, in addition to ?input_gene_symbol.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the query,
                          with columns for input gene symbol, pathway name, and other gene symbols.
        """
        if not gene_list:
            raise ValueError("The 'gene_list' cannot be empty.")

        # Format the gene list for the VALUES clause
        # Each gene symbol needs to be double-quoted and space-separated
        genes_for_values = " ".join(f'"{gene}"' for gene in gene_list)

        additional_filter = ""
        if exclude_gene_symbol:
            # Ensure the exclude_gene_symbol is properly quoted
            additional_filter = f'FILTER(?other_gene_symbol != "{exclude_gene_symbol}")'

        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?input_gene_symbol ?pathway_name ?other_gene_symbol
            WHERE {{
                # Use VALUES for input gene list
                VALUES ?input_gene_symbol {{ {genes_for_values} }}

                # Start with input gene and get its protein
                ?gene cbs:hasApprovedSymbol ?input_gene_symbol ;
                    cbs:encodesProtein ?protein .

                # Find pathways containing this protein
                ?pathway cbs:containsProtein ?protein ;
                        cbs:hasName ?pathway_name .

                # Find other proteins in same pathways
                ?pathway cbs:containsProtein ?other_protein .

                # Get genes encoding those other proteins
                ?other_gene cbs:encodesProtein ?other_protein ;
                            cbs:hasApprovedSymbol ?other_gene_symbol .

                # Exclude the original input gene
                FILTER(?other_gene_symbol != ?input_gene_symbol)

                # Add an additional exclusion if specified
                {additional_filter}
            }}
            """

        return GraphQueries.post_request(query)





New
1:21

    @staticmethod
    def query_pathways_location_processes_for_gene(
        gene_symbols: List[str],
    ) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find all pathways, biological processes and subcellular
        location for a given gene symbol. It will cocatenate all the pathway names together,
        similarly for biological processes and subcellular locations to give a single row
        for each gene symbol.

        Args:
            gene_symbol: The gene symbol to query for.

        Returns:
            pd.DataFrame: A DataFrame containing with columns for input gene symbol,
            pathway name, and other gene symbols.
        """
        genes = " ".join(f'"{gene}"' for gene in gene_symbols)
        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#> # Added SKOS prefix
            SELECT ?symbol
                (GROUP_CONCAT(DISTINCT ?subcellular_location_name; separator="; ") as ?locations)
                (GROUP_CONCAT(DISTINCT ?pathway_name; separator="; ") as ?pathways)
                (GROUP_CONCAT(DISTINCT ?process_name; separator="; ") as ?processes)
                (GROUP_CONCAT(DISTINCT ?pathway_id; separator="; ") as ?pathway_ids)
                # Added parent_pathways
                (GROUP_CONCAT(DISTINCT ?parent_pathway_name; separator="; ") AS ?parent_pathways)
            WHERE {{
                VALUES ?symbol {{ {genes} }}
                ?gene a cbs:Gene ;
                    cbs:hasApprovedSymbol ?symbol ;
                    cbs:encodesProtein ?protein .

                OPTIONAL {{
                    ?protein cbs:proteinLocalizedTo ?location .
                    ?location cbs:identifiedBy ?location_identifier .
                    ?location_identifier cbs:hasValue ?subcellular_location .
                    BIND(URI(CONCAT("http://purl.obolibrary.org/obo/",?subcellular_location)) AS ?go_subcellular_location)
                    ?go_subcellular_location rdfs:label ?subcellular_location_name
                }}

                OPTIONAL {{
                    ?pathway cbs:containsProtein ?protein ;
                            cbs:hasName ?pathway_name .
                    ?pathway cbs:identifiedBy ?id .
                    ?id cbs:hasValue ?pathway_id .

                    # Traversal to find parent pathways
                    OPTIONAL {{ # Use OPTIONAL to ensure results even if no parent pathway
                        ?pathway ^skos:narrower* ?parent_pathway .
                        ?parent_pathway a cbs:Pathway ;
                                        cbs:hasName ?parent_pathway_name .
                        FILTER NOT EXISTS {{
                            ?grand_parent_pathway skos:narrower ?parent_pathway
                        }}
                    }}
                }}

                OPTIONAL {{
                    ?protein cbs:hasFunction ?process .
                    ?process rdfs:label ?process_name .
                }}
            }}
            GROUP BY ?symbol
            """
        return GraphQueries.post_request(query)
1:21
    @staticmethod
    def query_clinical_trials_for_gene(
        gene_symbols: List[str],
        disease_mesh_id: str,
        disease_mondo_id: str,
    ) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find clinical trials associated with a given gene symbol and disease.
        Args:
            gene_symbols: A list of gene symbols to query for.
            disease_mesh_id: The MeSH ID of the disease.
            disease_mondo_id: The MONDO ID of the disease.
        Returns:
            pd.DataFrame: A DataFrame containing clinical trial information for the specified genes.
        """
        genes = " ".join(f'"{gene}"' for gene in gene_symbols)
        disease_filter_values = f"{disease_mesh_id}, {disease_mondo_id}"
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?symbol (MAX(?phase) AS ?maxPhase)
            {{
                # Get the drug information for a gene and disease
                ?drug rdf:type cbs:Drug .
                ?drug cbs:hasPrimaryName ?drugName .
                ?drug cbs:targets ?gene .
                ?gene cbs:encodesProtein ?protein .
                ?gene cbs:hasApprovedSymbol ?symbol .
                ?disease rdf:type cbs:Disease .
                ?drug cbs:participatesIn ?drugTrial .
                # Filter out the most broad disease and one level directly below it
                FILTER(?disease IN ( {disease_filter_values} )) .
                VALUES ?symbol {{ {genes} }}
                # Filter out neoplasm and everything above it
                {{
                    # Get clinical trial data
                    ?clinicalTrial rdf:type cbs:ClinicalTrial .
                    ?clinicalTrial cbs:contains ?drugTrial .
                    ?clinicalTrial cbs:hasTrialPhase ?rawTrialPhase .
                    BIND (
                        IF(?rawTrialPhase>=1 && ?rawTrialPhase<2, 1,
                            IF(?rawTrialPhase>=2 && ?rawTrialPhase<3, 2,
                                IF(?rawTrialPhase>=3 && ?rawTrialPhase<4, 3,
                                    IF(?rawTrialPhase>=4 , 4, -2)
                                )
                            )
                        ) AS ?phase
                    )
                    ?clinicalTrial cbs:investigates ?disease .
                }} UNION {{
                    # Get drug indication
                    ?drug cbs:indicates ?drugIndication .
                    ?drugIndication cbs:targets ?disease .
                    ?drugIndication cbs:hasStatus ?status .
                    BIND(
                        IF(?status='Phase I Clinical Trial', 1,
                            IF(?status='Phase II Clinical Trial', 2,
                                IF(?status='Phase III Clinical Trial', 3,
                                    IF(?status IN ('Pre-registration', 'Registered', 'Launched'), 4, -2)
                                )
                            )
                        ) AS ?phase
                    )
                }}
                BIND("informa" AS ?type)
            }} GROUP BY  ?symbol
            """
        return GraphQueries.post_request(query)

    @staticmethod
    def query_pockets_for_gene(
        gene_symbols: List[str],
    ) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find pxC50 values for a given list of genes.
        Args:
            gene_symbols: A list of gene symbols to query for.
        Returns:
            pd.DataFrame: A DataFrame containing the pxC50 values for the specified genes.
        """
        genes = " ".join(f'"{gene}"' for gene in gene_symbols)
        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?symbol (COUNT(DISTINCT?pocket) AS ?pocketCount)
            WHERE {{
                VALUES ?symbol {{ {genes} }}
                OPTIONAL {{
                    ?gene cbs:hasApprovedSymbol ?symbol ;
                        cbs:encodesProtein ?protein .
                    ?protein cbs:containsPocket ?pocket .
                }}
            }}
            GROUP BY ?symbol
        """
        return GraphQueries.post_request(query)
1:21
    @staticmethod
    def query_gene_disease_cooccurrence(
        gene_symbols: List[str],
        disease_mesh_id: str,
        disease_mondo_id: str,
    ) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find co-occurrence of genes and diseases in literature.
        Args:
            gene_symbols: A list of gene symbols to query for.
            disease_mondo_id: The MONDO ID of the disease.
        Returns:
            pd.DataFrame: A DataFrame containing the co-occurrence information.
        """
        genes = " ".join(f'"{gene}"' for gene in gene_symbols)
        disease_filter_values = f"{disease_mesh_id}, {disease_mondo_id}"
        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?symbol ?count
            WHERE {{
            VALUES ?symbol {{ {genes} }}
                ?cooccurrence a cbs:MedlineCooccuranceCount ;
                            cbs:cooccuranceRefersToGene ?gene ;
                            cbs:cooccuranceRefersToDisease ?disease ;
                            cbs:hasValue ?count .

                ?gene cbs:hasApprovedSymbol ?symbol .
                FILTER(?disease IN ( {disease_filter_values} )) .
            }}

            ORDER BY DESC(?count)
            """
        return GraphQueries.post_request(query)

    @staticmethod
    def query_chembl_compound_name(chembl_id: str, smiles: str) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find ChEMBL compounds by ChEMBL ID or SMILES.
        Args:
            chembl_id: The ChEMBL ID of the compound.
            smiles: The SMILES representation of the compound.
        Returns:
            pd.DataFrame: A DataFrame containing the ChEMBL compounds with its name
            and synonyms.
        """
        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?name WHERE {{
                {{
                    ?compound a cbs:Compound ;
                    cbs:hasCanonicalSmiles ?smiles ;
                    cbs:hasName ?name ;
                    cbs:hasId ?id .
                    VALUES ?id {{"{chembl_id}"}}
                    OPTIONAL {{
                        VALUES ?smiles {{"{smiles}"}}
                    }}
                }}
            }}
        """
        return GraphQueries.post_request(query)

    @staticmethod
    def query_compound_name_by_smiles(smiles: str) -> pd.DataFrame:
        """
        Constructs a SPARQL query to find compounds by SMILES.
        Args:
            smiles: The SMILES representation of the compound.
        Returns:
            pd.DataFrame: A DataFrame containing the ChEMBL compounds with its name
            and synonyms.
        """
        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?name WHERE {{
                {{
                    ?compound a cbs:Compound ;
                    cbs:hasCanonicalSmiles ?smiles ;
                    cbs:hasName ?name ;
                    cbs:hasId ?id .
                    VALUES ?smiles {{"{smiles}"}}
                }}
            }}
        """
        return GraphQueries.post_request(query)