from typing import Any, cast, Optional, Union, List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
import os
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, List
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import faiss
import re

graph_type_textual_descriptions = {
    "go": "Gene Ontology",
    "string": "STRINGdb",
    "pxmap": "phenomics embedding similarity",
    "txmap": "transcriptomic embedding similarity",
    "comp_moa": "compound-gene Mechanism of Action connectivity",
    "comp_sphere": "compound-gene phenomics similarity or dissimilarity",
    "comp_pxmap": "compound-gene phenomics similarity",
}

class KnowledgeGraph:
    """Class to create and manage knowledge graphs from various biological data sources.

    This class supports multiple graph types for hybrid GNN implementations, including
    GO, STRING, PXMAP, TXMAP, and other network types. It handles graph creation,
    processing, and caching.

    Args:
        graph_cfg: The type of graph including optional arguments.
            - str: Provide graph type (e.g. "go", "string", "pxmap", "txmap", "dense")
            - List: Provide a list of graph types, e.g., ["go", "string"]
            - Dict: Provide a dictionary with graph identifiers as keys and optional
              arguments as values
        cache_dir: The directory to store the graphs
    """

    def __init__(
        self,
        graph_cfg: str | list[str] | dict[str, dict[str, Any]] = "go",
        cache_dir: str = "/rxrx/data/valence/pathfinder/modules/gspp/graphs",
    ) -> None:
        """Initialize the KnowledgeGraph.

        Args:
            graph_cfg: Configuration for graph types and their arguments
            cache_dir: Directory path for caching graphs
        """
        # Convert various accepted input formats into a DictConfig once so that the
        # attribute does not change type later (avoids mypy complaints).
        if isinstance(graph_cfg, str):
            graph_cfg_dict: dict[str, dict[str, Any]] = {
                "graph0": {"graph_type": graph_cfg}
            }
        elif isinstance(graph_cfg, list):
            graph_cfg_dict = {
                f"graph{i}": {"graph_type": g_type}
                for i, g_type in enumerate(graph_cfg)
            }
        else:
            # Already a mapping
            graph_cfg_dict = graph_cfg

        # Store as DictConfig for homogeneous typing downstream
        self.graph_cfg: DictConfig = OmegaConf.create(graph_cfg_dict)
        self.cache_dir = cache_dir
        
        self.edge_index = []
        self.edge_textual_descriptions = []
        self.edge_attr = None

        # Load gene sets and create mappings
        gears_genes = pd.read_csv(f"{cache_dir}/gears_gene_set.csv")
        master_genes = pd.read_csv(f"{cache_dir}/master_gene_set_sorted.csv")
        self.has_compound = False
        self.comp2id: dict[str, int] = {}

        # Create pert2id mapping from gears gene set
        self.pert2id = {row.iloc[1]: row.iloc[0] for _, row in gears_genes.iterrows()}

        # Create gene2id mapping from master gene set
        self.gene2id = {
            row["gene_name"]: row["gene_id"] for _, row in master_genes.iterrows()
        }

        self.graph_dict = {}
        graph_cfg_dict = cast(dict[str, dict[str, Any]], self.graph_cfg)
        for graph_name, graph_args_raw in graph_cfg_dict.items():
            graph_args: Any = graph_args_raw
            if isinstance(graph_args, dict) and not OmegaConf.is_config(graph_args):
                graph_args = OmegaConf.create(graph_args)

            graph_type = graph_args.pop("graph_type", "string")
            graph = self.create_graph(graph_type, cast(dict[str, Any], graph_args))
            self.graph_dict[graph_name] = graph
            
        self.language_model: Optional[SentenceTransformer] = None
        
        self.nodes_textual_descriptions = {}
        self.name_to_id = {}
        for gene_name, id in self.pert2id.items():
            name = f"gene {gene_name}"
            self.nodes_textual_descriptions[id] = name
            self.name_to_id[gene_name.lower().strip()] = id
            # TODO: add other aliases for gene names
        for smiles, id in self.comp2id.items():
            name = f"compound {smiles}"
            self.nodes_textual_descriptions[id] = name
            self.name_to_id[smiles] = id
            # TODO: add inchikeys and other aliases to the name_to_id
            
        self.edge_index = torch.cat(self.edge_index, dim=1)
        self.node_embeddings, self.edge_attr = self.create_node_and_edge_embeddings()
        
        self.pyg_graph = Data(x=self.node_embeddings, edge_index=self.edge_index, edge_attr=self.edge_attr)
            
    def create_node_and_edge_embeddings(self) -> torch.Tensor:
        """Create node and edge embeddings from textual descriptions using the language model.
        
        Returns:
            torch.Tensor: Node embedding matrix where row i contains the embedding for node ID i,
                         or zero vector if ID i has no description
            torch.Tensor: Edge embedding matrix where row i contains the embedding for edge i
        """
        if self.language_model is None:
            self.load_language_model()
            
        # Get max node ID to determine embedding matrix size
        max_id = max(self.nodes_textual_descriptions.keys())
        
        # Get embeddings for nodes that have descriptions
        node_ids = list(self.nodes_textual_descriptions.keys())
        descriptions = [self.nodes_textual_descriptions[id] for id in node_ids]
        embeddings = self.text_to_embedding(descriptions)
        
        # Create zero tensor of shape [max_id + 1, embedding_dim]
        embedding_dim = embeddings.shape[1]
        x = torch.zeros((max_id + 1, embedding_dim), device=embeddings.device)
        
        # Fill in embeddings for nodes that have descriptions
        for idx, node_id in enumerate(node_ids):
            x[node_id] = embeddings[idx]
            
        edge_embeddings = self.text_to_embedding(self.edge_textual_descriptions)
        return x, edge_embeddings

    def load_language_model(self, model_name: str = "pritamdeka/S-PubMedBERT-MS-MARCO") -> None:
        """Load a lightweight language model for creating text embeddings.
        
        Args:
            model_name: Name of the sentence-transformer model to load. Default is 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
                       which is a biomedical language model trained on PubMed data. Other options include:
                       - 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' (default, best for biomedical concepts)
                       - 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb' (BioBERT fine-tuned on multiple tasks)
                       - 'pritamdeka/S-PubMedBERT-MS-MARCO' (PubMedBERT fine-tuned for biomedical search)
                       - 'michiyasunaga/BioLinkBERT-base' (BioLinkBERT for biomedical entity linking)
        """
        self.language_model = SentenceTransformer(model_name)
        
    def text_to_embedding(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Convert text strings into numerical embeddings using the loaded language model.
        
        Args:
            texts: Single text string or list of textual descriptions for nodes
            
        Returns:
            torch.Tensor: Embedding vector for single text, or matrix of embeddings (one row per text) for multiple texts
            
        Raises:
            RuntimeError: If the language model hasn't been loaded
        """
        if self.language_model is None:
            raise RuntimeError("Language model not loaded. Call load_language_model() first.")
            
        embeddings = self.language_model.encode(texts, convert_to_tensor=True)
        return embeddings

    def create_graph(
        self, graph_type: str, graph_args: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load and create a graph based on the specified type.

        Args:
            graph_type: Type of graph to create (e.g., "go", "string", "pxmap")
            graph_args: Optional arguments for graph creation

        Returns:
            Tuple containing:
                - Edge index tensor
                - Edge weight tensor
                - Number of nodes in the graph

        Raises:
            ValueError: If an invalid graph type is provided
        """
        if graph_args is None:
            graph_args = {}

        if graph_type == "go-gears":
            edge_index = joblib.load(f"{self.cache_dir}/go-grears/gears-go-edges.pkl")
            edge_weight = joblib.load(
                f"{self.cache_dir}/go-grears/gears-go-weights.pkl"
            )
            graph = edge_index, edge_weight, len(self.pert2id)

        elif graph_type == "go":
            network = pd.read_csv(f"{self.cache_dir}/go/go_essential_all.csv")
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "reactome":
            network = pd.read_csv(f"{self.cache_dir}/reactome/gsp_reactome.csv")
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "string":
            network = pd.read_parquet(f"{self.cache_dir}/string/v11.5.parquet")
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "pxmap":
            raise ValueError(
                "pxmap is a large file and is not supported yet. You can use pxmap-top1 instead."
            )

        elif graph_type == "txmap":
            raise ValueError(
                "txmap is a large file and is not supported yet. You can use txmap-top1 instead."
            )

        elif graph_type == "pxmap-top1":
            network = pd.read_pickle(
                f"{self.cache_dir}/pxmap/top1/top_phenomics_similarities.pickle"
            )
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "txmap-top1":
            network = pd.read_pickle(
                f"{self.cache_dir}/txmap/top1/top_transcriptomic_similarities.pickle"
            )
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "afm-conf":
            network = pd.read_pickle(f"{self.cache_dir}/afm/AFM_screen_WS3.pkl")
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "afm-full":
            network = pd.read_pickle(f"{self.cache_dir}/afm/AFM_screen_WS3_full.pkl")
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "depmap":
            raise ValueError(
                "depmap is a large file and is not supported yet. You can use depmap-top1 instead."
            )

        elif graph_type == "depmap-top1":
            network = pd.read_pickle(f"{self.cache_dir}/depmap/DepMap_WS3_top1.pkl")
            graph = self.process_gene_gene_graph(network, **graph_args)

        elif graph_type == "dense":
            num_nodes = len(self.pert2id)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
            graph = edge_index, edge_weight, len(self.pert2id)

        # adding three compound-gene graphs: moa, sphere, pxmap
        elif graph_type.startswith("comp"):
            self.has_compound = True
            graph = self.process_compound_gene_graph(
                graph_type, f"{self.cache_dir}/comp-gene-graphs"
            )

        else:
            raise ValueError(f"Invalid graph type: {graph_type}")
        
        edge_index, edge_weight, num_nodes = graph
        self.edge_index.append(edge_index)
        
        edge_description_base = graph_type_textual_descriptions[graph_type] if graph_type in graph_type_textual_descriptions else graph_type
        for i in range(edge_index.size(1)):
            weight = edge_weight[i] if edge_weight is not None else 1.0
            self.edge_textual_descriptions.append(f"{edge_description_base} with weight {weight}")

        return graph

    def process_gene_gene_graph(
        self,
        network: pd.DataFrame,
        reduce2perts: bool = True,
        reduce2positive: bool = False,
        norm_weights: bool = False,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process the network and convert it to PyG format.

        Args:
            network: The network to process
            reduce2perts: Whether to reduce the network to only perturbation genes
            reduce2positive: Whether to reduce the network to only positive weights
            norm_weights: Whether to normalize the weights to [0,1]
            mode: The mode of edge selection. Supports:
                - "top_n": Keep n edges per target
                - "quantile_q": Keep edges in specified quantile
                - "threshold": Keep edges above weight threshold
                - "cossim_thresh": Signal cut-off for Ph and Tx
                - "top_k": Keep absolute top signals (must be <5)

        Returns:
            Tuple containing:
                - Edge index tensor
                - Edge weight tensor
                - Number of nodes in the graph

        Raises:
            ValueError: If network columns are invalid or cosine similarity threshold is invalid
        """
        # Create a copy of the input DataFrame to avoid SettingWithCopyWarning
        network = network.copy()

        # Check which column naming convention is used
        if "regulator" in network.columns and "target" in network.columns:
            source_col = "regulator"
            target_col = "target"
        elif "source" in network.columns and "target" in network.columns:
            source_col = "source"
            target_col = "target"
        elif "gene1" in network.columns and "gene2" in network.columns:
            source_col = "gene1"
            target_col = "gene2"
        else:
            raise ValueError(
                "Network must have either regulator/target or gene1/gene2 columns"
            )

        # Standardize column names
        column_mappings = {
            "similarity": "weight",
            "importance": "weight",
            "normalized_iptm_conf": "weight",
            "iptm": "weight",
        }
        for old_col, new_col in column_mappings.items():
            if old_col in network.columns:
                network = network.rename(columns={old_col: new_col})

        # Rename columns to standard format
        if source_col != "regulator":
            network = network.rename(
                columns={source_col: "regulator", target_col: "target"}
            )

        # Reduce to edges between perturbation genes
        if reduce2perts:
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)      
        else:
            network["regulator"] = network["regulator"].map(self.gene2id)
            network["target"] = network["target"].map(self.gene2id)
            network = network.dropna()
            num_nodes = len(self.gene2id)

        # Reduce to positive weights only
        if reduce2positive:
            network = network[network["weight"] > 0]
            network = network.reset_index(drop=True)

        # Normalize the weights to [0,1]
        if norm_weights:
            # Ensure weight column is float type
            network["weight"] = network["weight"].astype(float)
            # Calculate max weight for normalization
            max_weight = network["weight"].max()
            # Use loc to avoid SettingWithCopyWarning
            network.loc[:, "weight"] = abs(network["weight"] / max_weight)

        # Process based on mode
        if mode is not None:
            mode, arg_str = mode.split("_")
            arg = int(arg_str)

            if mode == "top":
                network = (
                    network.groupby("target")
                    .apply(lambda x: x.nlargest(arg, ["weight"]))
                    .reset_index(drop=True)
                )
            elif mode == "threshold":
                network = (
                    network.groupby("target")
                    .apply(lambda x: x[x["weight"] >= arg])
                    .reset_index(drop=True)
                )
            elif "cossim-threshold" in mode:
                threshold = arg / 100

                if not 0 <= threshold <= 1:
                    raise ValueError(
                        "Cosine similarity threshold must be between 0 and 1"
                    )
                if threshold < 0.1:
                    raise ValueError("The threshold is too low for cosine similarity.")

                if "abs" in mode:
                    network = network[network["weight"].abs() >= threshold]
                else:
                    network = network[network["weight"] >= threshold]

                network = network.reset_index(drop=True)
            elif "percentile" in mode:
                if "abs" in mode:
                    threshold = int(np.percentile(network["weight"].abs(), arg))
                    network = network[network["weight"].abs() >= threshold]
                else:
                    threshold = int(np.percentile(network["weight"], arg))
                    network = network[network["weight"] >= threshold]

                network = network.reset_index(drop=True)

        # Package the graph into PyG format
        edge_index = torch.tensor(
            np.array([network["regulator"].to_numpy(), network["target"].to_numpy()]),
            dtype=torch.long,
        )
        edge_weight = torch.tensor(network["weight"].to_numpy(), dtype=torch.float)

        return edge_index, edge_weight, num_nodes

    def process_compound_gene_graph(
        self, graph_type: str, base_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process compound-gene graph data.

        Args:
            graph_type: Type of compound graph to process (e.g., "comp_pxmap", "comp_moa")

        Returns:
            Tuple containing:
                - Edge index tensor of shape (2, num_edges)
                - Edge weight tensor of shape (num_edges,)
                - Number of nodes in the graph (including compounds)

        Raises:
            ValueError: If graph_type is not in expected format or if required files are missing
            FileNotFoundError: If required data files cannot be found
        """
        try:
            # Extract graph category from type
            if not graph_type.startswith("comp_"):
                raise ValueError(f"Invalid compound graph type: {graph_type}")
            graph_category = graph_type.split("_")[1]

            # Load compound-gene relations
            file_path = f"{base_path}/{graph_category}_comp_gene_relations.csv"
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Compound-gene relations file not found: {file_path}"
                )
            df = pd.read_csv(file_path)

            # Load SMILES to InChIKey mapping
            smiles_inchikeys_path = f"{base_path}/smiles_inchikeys.csv"
            if not os.path.exists(smiles_inchikeys_path):
                raise FileNotFoundError(
                    f"SMILES-InChIKey mapping file not found: {smiles_inchikeys_path}"
                )

            s2i = self._load_smiles_inchikey_mapping(smiles_inchikeys_path)

            # Process graph based on category
            if graph_category == "pxmap":
                df = self._process_pxmap_data(df)

            # Create compound-gene mappings
            inchi_to_genes, inchi_to_weights = self._create_compound_gene_mappings(df)

            # Build edge index and weights
            edge_index, edge_weights = self._build_compound_graph_edges(
                s2i, inchi_to_genes, inchi_to_weights
            )

            return edge_index, edge_weights, len(self.comp2id)

        except Exception as e:
            raise RuntimeError(f"Error processing compound graph: {str(e)}") from e

    def _load_smiles_inchikey_mapping(self, file_path: str) -> dict[str, str]:
        """Load SMILES to InChIKey mapping from file.

        Args:
            file_path: Path to the SMILES-InChIKey mapping file

        Returns:
            Dictionary mapping SMILES to InChIKeys

        Raises:
            ValueError: If file format is invalid or duplicate SMILES found
        """
        s2i = {}
        with open(file_path) as f:
            header = f.readline().strip()
            if header != "SMILES,InChIKey":
                raise ValueError(
                    "Invalid file format: expected 'SMILES,InChIKey' header"
                )

            for line in f:
                smiles, inchikey = line.strip().split(",")
                if smiles in s2i:
                    raise ValueError(f"Duplicate SMILES found: {smiles}")
                s2i[smiles] = inchikey
        return s2i

    def _process_pxmap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process PxMap data by sorting and selecting top connections.

        Args:
            df: Input DataFrame with PXMAP data

        Returns:
            Processed DataFrame with top connections per compound
        """
        return (
            df.sort_values(by=["inchi_key", "weight"], ascending=[True, False])
            .groupby("inchi_key")
            .head(20)
            .reset_index(drop=True)
        )

    def _create_compound_gene_mappings(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
        """Create mappings from InChIKey to gene IDs and weights.

        Args:
            df: DataFrame containing compound-gene relations

        Returns:
            Tuple of dictionaries mapping InChIKey to gene IDs and weights
        """
        # Select only the required columns before groupby to avoid deprecation warning
        required_cols = ["gene"]
        if "weight" in df.columns:
            required_cols.append("weight")

        inchi_to_genes_weights = (
            df.groupby("inchi_key")[required_cols]
            .apply(
                lambda x: list(
                    zip(x["gene"], x["weight"] if "weight" in x else [1.0] * len(x))
                )
            )
            .to_dict()
        )

        inchi_to_genes: dict[str, list[int]] = {}
        inchi_to_weights: dict[str, list[float]] = {}

        for inchi_key, gene_weights in inchi_to_genes_weights.items():
            gene_set: dict[str, float] = {}
            for gene, weight in gene_weights:
                if gene not in gene_set or abs(weight) > abs(gene_set[gene]):
                    gene_set[gene] = weight

            intersected_genes = set(gene_set.keys()).intersection(self.pert2id.keys())
            inchi_to_genes[inchi_key] = [
                self.pert2id[gene] for gene in intersected_genes
            ]
            inchi_to_weights[inchi_key] = [gene_set[gene] for gene in intersected_genes]

        return inchi_to_genes, inchi_to_weights

    def _build_compound_graph_edges(
        self,
        s2i: dict[str, str],
        inchi_to_genes: dict[str, list[int]],
        inchi_to_weights: dict[str, list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build edge index and weight tensors for the compound graph.

        Args:
            s2i: Dictionary mapping SMILES to InChIKeys
            inchi_to_genes: Dictionary mapping InChIKey to gene IDs
            inchi_to_weights: Dictionary mapping InChIKey to weights

        Returns:
            Tuple of edge index and weight tensors
        """
        edge_index = []
        edge_weights = []

        # Add control node
        self.comp2id["ctrl"] = len(self.pert2id)

        # Process each compound
        for smiles in s2i.keys():
            if smiles not in self.comp2id:
                self.comp2id[smiles] = len(self.comp2id) + len(self.pert2id)

            cid = self.comp2id[smiles]
            inchikey = s2i[smiles]

            if inchikey in inchi_to_genes:
                for idx, gid in enumerate(inchi_to_genes[inchikey]):
                    edge_index.append((gid, cid))
                    edge_weights.append(inchi_to_weights[inchikey][idx])

        return (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            torch.tensor(edge_weights, dtype=torch.float),
        )

    def find_similar_nodes(
        self, 
        query: str, 
        k: int = 10, 
        similarity_metric: str = "cosine",
        use_precomputed_index: bool = True
    ) -> Dict[int, float]:
        """Find the top-k most similar nodes to a given query.
        
        Args:
            query: Text query to find similar nodes for
            k: Number of top similar nodes to return
            similarity_metric: Similarity metric to use ("cosine", "euclidean", "dot")
            use_precomputed_index: Whether to use precomputed FAISS index for faster search
            
        Returns:
            Dictionary mapping node IDs to similarity scores (higher is more similar)
            
        Raises:
            RuntimeError: If language model is not loaded
            ValueError: If similarity metric is not supported
        """
        if self.language_model is None:
            raise RuntimeError("Language model not loaded. Call load_language_model() first.")
            
        # Get query embedding
        query_embedding = self.text_to_embedding(query)
        
        if use_precomputed_index and hasattr(self, '_faiss_index'):
            return self._search_with_faiss(query_embedding, k)
        else:
            return self._search_with_brute_force(query_embedding, k, similarity_metric)
    
    def _search_with_brute_force(
        self, 
        query_embedding: torch.Tensor, 
        k: int, 
        similarity_metric: str
    ) -> Dict[int, float]:
        """Search for similar nodes using brute force computation.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top similar nodes to return
            similarity_metric: Similarity metric to use
            
        Returns:
            Dictionary mapping node IDs to similarity scores
        """
        # Get node embeddings for nodes that have descriptions
        node_ids = list(self.nodes_textual_descriptions.keys())
        node_embeddings = self.node_embeddings[node_ids]
        
        # Convert to numpy for sklearn compatibility
        query_np = query_embedding.cpu().numpy().reshape(1, -1)
        node_embeddings_np = node_embeddings.cpu().numpy()
        
        # Compute similarities
        if similarity_metric == "cosine":
            similarities = cosine_similarity(query_np, node_embeddings_np)[0]
        elif similarity_metric == "euclidean":
            # Convert euclidean distance to similarity (1 / (1 + distance))
            distances = np.linalg.norm(node_embeddings_np - query_np, axis=1)
            similarities = 1 / (1 + distances)
        elif similarity_metric == "dot":
            similarities = np.dot(node_embeddings_np, query_np.T).flatten()
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return results as dictionary
        results = {}
        for idx in top_indices:
            node_id = node_ids[idx]
            results[node_id] = float(similarities[idx])
            
        return results
    
    def _search_with_faiss(
        self, 
        query_embedding: torch.Tensor, 
        k: int
    ) -> Dict[int, float]:
        """Search for similar nodes using precomputed FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top similar nodes to return
            
        Returns:
            Dictionary mapping node IDs to similarity scores
        """
        # Convert query to numpy and reshape
        query_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        # Search using FAISS
        distances, indices = self._faiss_index.search(query_np, k)
        
        # Convert distances to similarities (FAISS returns distances, not similarities)
        similarities = 1 / (1 + distances[0])
        
        # Map back to original node IDs
        node_ids = list(self.nodes_textual_descriptions.keys())
        results = {}
        for i, idx in enumerate(indices[0]):
            if idx < len(node_ids):  # Safety check
                node_id = node_ids[idx]
                results[node_id] = float(similarities[i])
                
        return results
    
    def build_similarity_index(
        self, 
        index_type: str = "faiss", 
        similarity_metric: str = "cosine",
        n_neighbors: int = 100
    ) -> None:
        """Build a precomputed similarity index for faster repeated queries.
        
        Args:
            index_type: Type of index to build ("faiss", "sklearn")
            similarity_metric: Similarity metric to use
            n_neighbors: Number of neighbors for sklearn NearestNeighbors
            
        Raises:
            ValueError: If index_type is not supported
        """
        if self.language_model is None:
            raise RuntimeError("Language model not loaded. Call load_language_model() first.")
            
        # Get node embeddings for nodes that have descriptions
        node_ids = list(self.nodes_textual_descriptions.keys())
        node_embeddings = self.node_embeddings[node_ids]
        node_embeddings_np = node_embeddings.cpu().numpy()
        
        if index_type == "faiss":
            self._build_faiss_index(node_embeddings_np, similarity_metric)
        elif index_type == "sklearn":
            self._build_sklearn_index(node_embeddings_np, similarity_metric, n_neighbors)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        # Store node IDs mapping for the index
        self._index_node_ids = node_ids
        
    def _build_faiss_index(self, embeddings: np.ndarray, similarity_metric: str) -> None:
        """Build FAISS index for similarity search.
        
        Args:
            embeddings: Node embeddings matrix
            similarity_metric: Similarity metric to use
        """
        dimension = embeddings.shape[1]
        
        if similarity_metric == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self._faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        elif similarity_metric == "euclidean":
            self._faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance
        else:
            raise ValueError(f"FAISS doesn't support similarity metric: {similarity_metric}")
            
        self._faiss_index.add(embeddings.astype(np.float32))
        
    def _build_sklearn_index(
        self, 
        embeddings: np.ndarray, 
        similarity_metric: str, 
        n_neighbors: int
    ) -> None:
        """Build sklearn NearestNeighbors index for similarity search.
        
        Args:
            embeddings: Node embeddings matrix
            similarity_metric: Similarity metric to use
            n_neighbors: Number of neighbors
        """
        if similarity_metric == "cosine":
            metric = "cosine"
        elif similarity_metric == "euclidean":
            metric = "euclidean"
        else:
            raise ValueError(f"sklearn doesn't support similarity metric: {similarity_metric}")
            
        self._sklearn_index = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric=metric, 
            algorithm="auto"
        )
        self._sklearn_index.fit(embeddings)
        
    def batch_find_similar_nodes(
        self, 
        queries: List[str], 
        k: int = 10, 
        similarity_metric: str = "cosine",
        use_precomputed_index: bool = True
    ) -> List[Dict[int, float]]:
        """Find similar nodes for multiple queries efficiently.
        
        Args:
            queries: List of text queries
            k: Number of top similar nodes to return per query
            similarity_metric: Similarity metric to use
            use_precomputed_index: Whether to use precomputed index
            
        Returns:
            List of dictionaries, each mapping node IDs to similarity scores
        """
        if self.language_model is None:
            raise RuntimeError("Language model not loaded. Call load_language_model() first.")
            
        # Get embeddings for all queries
        query_embeddings = self.text_to_embedding(queries)
        
        if use_precomputed_index and hasattr(self, '_faiss_index'):
            return self._batch_search_with_faiss(query_embeddings, k)
        else:
            return self._batch_search_with_brute_force(query_embeddings, k, similarity_metric)
    
    def _batch_search_with_faiss(
        self, 
        query_embeddings: torch.Tensor, 
        k: int
    ) -> List[Dict[int, float]]:
        """Batch search using FAISS index.
        
        Args:
            query_embeddings: Query embeddings matrix
            k: Number of top similar nodes to return per query
            
        Returns:
            List of dictionaries mapping node IDs to similarity scores
        """
        query_np = query_embeddings.cpu().numpy()
        
        # Search using FAISS
        distances, indices = self._faiss_index.search(query_np, k)
        
        # Convert distances to similarities
        similarities = 1 / (1 + distances)
        
        # Map back to original node IDs
        node_ids = list(self.nodes_textual_descriptions.keys())
        results = []
        
        for i in range(len(query_np)):
            query_results = {}
            for j, idx in enumerate(indices[i]):
                if idx < len(node_ids):
                    node_id = node_ids[idx]
                    query_results[node_id] = float(similarities[i, j])
            results.append(query_results)
            
        return results
    
    def _batch_search_with_brute_force(
        self, 
        query_embeddings: torch.Tensor, 
        k: int, 
        similarity_metric: str
    ) -> List[Dict[int, float]]:
        """Batch search using brute force computation.
        
        Args:
            query_embeddings: Query embeddings matrix
            k: Number of top similar nodes to return per query
            similarity_metric: Similarity metric to use
            
        Returns:
            List of dictionaries mapping node IDs to similarity scores
        """
        node_ids = list(self.nodes_textual_descriptions.keys())
        node_embeddings = self.node_embeddings[node_ids]
        
        query_np = query_embeddings.cpu().numpy()
        node_embeddings_np = node_embeddings.cpu().numpy()
        
        # Compute similarities for all queries
        if similarity_metric == "cosine":
            similarities = cosine_similarity(query_np, node_embeddings_np)
        elif similarity_metric == "euclidean":
            # For batch euclidean, we need to compute pairwise distances
            similarities = np.zeros((len(query_np), len(node_embeddings_np)))
            for i, query in enumerate(query_np):
                distances = np.linalg.norm(node_embeddings_np - query, axis=1)
                similarities[i] = 1 / (1 + distances)
        elif similarity_metric == "dot":
            similarities = np.dot(query_np, node_embeddings_np.T)
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
        # Get top-k for each query
        results = []
        for i in range(len(query_np)):
            top_indices = np.argsort(similarities[i])[::-1][:k]
            query_results = {}
            for idx in top_indices:
                node_id = node_ids[idx]
                query_results[node_id] = float(similarities[i, idx])
            results.append(query_results)
            
        return results

    def get_k_hop_neighborhood(
        self, 
        node_ids: Union[int, List[int]], 
        k: int = 1,
        include_self: bool = True,
        max_nodes: Optional[int] = None
    ) -> Data:
        """Retrieve a k-hop neighborhood graph around specified nodes using PyG's k_hop_subgraph.
        
        Args:
            node_ids: Single node ID or list of node IDs to find neighborhoods for
            k: Number of hops to expand (default: 1)
            include_self: Whether to include the source nodes in the neighborhood
            max_nodes: Maximum number of nodes to include in the neighborhood (None for no limit)
            
        Returns:
            PyG Data object containing the k-hop neighborhood subgraph
            
        Raises:
            ValueError: If k is negative or node_ids are invalid
            RuntimeError: If PyG graph is not available
        """
        if k < 0:
            raise ValueError("k must be non-negative")
            
        if self.pyg_graph is None:
            raise RuntimeError("PyG graph not available. Ensure graph was created successfully.")
            
        # Convert single node ID to list
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            
        # Validate node IDs
        max_node_id = self.pyg_graph.x.size(0) - 1
        for node_id in node_ids:
            if not isinstance(node_id, int) or node_id < 0 or node_id > max_node_id:
                raise ValueError(f"Invalid node ID: {node_id}. Must be between 0 and {max_node_id}")
        
        # Convert to tensor
        subset = torch.tensor(node_ids, dtype=torch.long)
        
        # Use PyG's k_hop_subgraph
        from torch_geometric.utils import k_hop_subgraph
        
        subset_nodes, edge_index, mapping, edge_mask  = k_hop_subgraph(
            subset, 
            k, 
            self.pyg_graph.edge_index, 
            relabel_nodes=True,
            num_nodes=self.pyg_graph.x.size(0),
            directed=True
        )
        
        # Create subgraph
        x = self.pyg_graph.x[subset_nodes]
        edge_attr = self.pyg_graph.edge_attr[edge_mask] if self.pyg_graph.edge_attr is not None else None
        
        subgraph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Store original node IDs for reference
        subgraph.original_node_ids = subset_nodes.tolist()
        subgraph.source_mapping = mapping.tolist()  # Maps new indices to original source indices
        
        return subgraph
    
    def get_neighborhood_info(
        self, 
        node_ids: Union[int, List[int]], 
        k: int = 1,
        include_self: bool = True
    ) -> Dict[str, Any]:
        """Get detailed information about the k-hop neighborhood.
        
        Args:
            node_ids: Single node ID or list of node IDs
            k: Number of hops
            include_self: Whether to include source nodes
            
        Returns:
            Dictionary containing neighborhood information
        """
        # Convert single node ID to list
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        
        # Get the subgraph
        subgraph = self.get_k_hop_neighborhood(node_ids, k, include_self)
        
        # Get node descriptions
        node_descriptions = {}
        for idx, original_id in enumerate(subgraph.original_node_ids):
            description = self.nodes_textual_descriptions.get(original_id, f"Node {original_id}")
            is_source = original_id in node_ids
            node_descriptions[idx] = {
                'original_id': original_id,
                'description': description,
                'is_source': is_source
            }
        
        # Calculate statistics
        num_nodes = subgraph.x.size(0)
        num_edges = subgraph.edge_index.size(1) if subgraph.edge_index.size(1) > 0 else 0
        
        # Find source nodes in the subgraph
        source_nodes_in_subgraph = []
        for idx, original_id in enumerate(subgraph.original_node_ids):
            if original_id in node_ids:
                source_nodes_in_subgraph.append(idx)
        
        return {
            'subgraph': subgraph,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'k': k,
            'source_nodes': node_ids,
            'source_nodes_in_subgraph': source_nodes_in_subgraph,
            'node_descriptions': node_descriptions,
            'original_node_ids': subgraph.original_node_ids
        }
    
    def visualize_neighborhood(
        self, 
        node_ids: Union[int, List[int]], 
        k: int = 1,
        include_self: bool = True,
        max_nodes: int = 50
    ) -> str:
        """Generate a text representation of the neighborhood for visualization.
        
        Args:
            node_ids: Single node ID or list of node IDs
            k: Number of hops
            include_self: Whether to include source nodes
            max_nodes: Maximum number of nodes to include
            
        Returns:
            String representation of the neighborhood
        """
        info = self.get_neighborhood_info(node_ids, k, include_self)
        
        # Limit nodes for visualization
        if info['num_nodes'] > max_nodes:
            # Keep source nodes and sample others
            source_indices = info['source_nodes_in_subgraph']
            other_indices = [i for i in range(info['num_nodes']) if i not in source_indices]
            sample_size = max_nodes - len(source_indices)
            if sample_size > 0:
                sampled_others = other_indices[:sample_size]
                kept_indices = source_indices + sampled_others
            else:
                kept_indices = source_indices[:max_nodes]
        else:
            kept_indices = list(range(info['num_nodes']))
        
        # Build visualization string
        lines = []
        lines.append(f"K-hop Neighborhood (k={k})")
        lines.append(f"Total nodes: {info['num_nodes']}, Total edges: {info['num_edges']}")
        lines.append(f"Source nodes: {node_ids}")
        lines.append("-" * 50)
        
        # Add node information
        lines.append("Nodes in neighborhood:")
        for idx in kept_indices:
            node_info = info['node_descriptions'][idx]
            marker = "★" if node_info['is_source'] else "○"
            lines.append(f"  {marker} {idx} (ID: {node_info['original_id']}): {node_info['description']}")
        
        if info['num_nodes'] > max_nodes:
            lines.append(f"  ... and {info['num_nodes'] - max_nodes} more nodes")
        
        # Add edge information if not too many
        if info['num_edges'] > 0 and info['num_edges'] <= 100:
            lines.append("\nEdges:")
            edge_index = info['subgraph'].edge_index
            for i in range(min(20, edge_index.size(1))):  # Show first 20 edges
                src, dst = edge_index[:, i].tolist()
                src_desc = info['node_descriptions'][src]['description']
                dst_desc = info['node_descriptions'][dst]['description']
                lines.append(f"  {src} → {dst}: {src_desc} → {dst_desc}")
            
            if edge_index.size(1) > 20:
                lines.append(f"  ... and {edge_index.size(1) - 20} more edges")
        
        return "\n".join(lines)
    
    def find_nodes_from_query(self, query: str) -> Dict[str, int]:
        """Break query into words and find matching nodes in the graph.
        
        Args:
            query: Text query to search for matching nodes
            
        Returns:
            Dictionary mapping matched words to their corresponding node IDs
        """
        # Split on spaces and punctuation
        words = re.split(r'[?\s,!;.]+', query.lower())
        words = [w.strip() for w in words if w.strip()]
        
        # Find matches in name_to_id mapping
        matches = {}
        for word in words:
            if word in self.name_to_id:
                matches[word] = (self.name_to_id[word], self.nodes_textual_descriptions[self.name_to_id[word]])
                
        return matches
    
    def subgraph_to_text(
        self,
        node_ids: List[int],
        edge_ids: torch.Tensor,
        max_edges: int = 500
    ) -> str:
        """Convert a subgraph to a textual description suitable for LLMs.

        Args:
            node_ids: List of node IDs in the subgraph
            edge_ids: List of edge IDs in the subgraph
            max_edges: Maximum number of edges to include in description

        Returns:
            String containing natural language description of the subgraph
        """
        # Start with overview
        lines = []
        lines.append(f"The retrieved subgraph contains {len(node_ids)} nodes and {edge_ids.size(1)} edges.")
        
        edge_index = self.pyg_graph.edge_index[:, edge_ids]
        
        # Add node descriptions
        lines.append("\nNodes:")
        for node_id in node_ids:
            if node_id in self.nodes_textual_descriptions:
                lines.append(f"- {self.nodes_textual_descriptions[node_id]}")
            else:
                lines.append(f"- Node {node_id}")
                
        # Add edge descriptions
        lines.append("\nConnections:")
        num_edges = min(edge_index.size(1), max_edges)
        for i in range(num_edges):
            src, dst = edge_index[:, i].tolist()
            src_desc = self.nodes_textual_descriptions.get(src, f"Node {src}")
            dst_desc = self.nodes_textual_descriptions.get(dst, f"Node {dst}")
            lines.append(f"- {src_desc} is connected to {dst_desc} with edge information: {self.edge_textual_descriptions[edge_ids[i]]}")
                
        if edge_index.size(1) > max_edges:
            lines.append(f"\n... and {edge_index.size(1) - max_edges} more connections")
            
        return "\n".join(lines)