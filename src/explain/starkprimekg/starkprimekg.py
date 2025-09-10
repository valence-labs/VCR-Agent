import torch
import os
import pickle
import json
import torch_geometric
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Union, List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.utils import k_hop_subgraph
import networkx as nx

# Global DrugBankSearcher instance to be shared across KnowledgeGraph instances
_global_drugbank_searcher = None

class StarkPrimeKG:
    def __init__(self, save_address='/rxrx/data/user/hamed.shirzad/outgoing/stark_prime_kg', 
                 enc_language_model_name='pritamdeka/S-PubMedBERT-MS-MARCO', 
                 node_emb_save_address='/rxrx/data/user/hamed.shirzad/outgoing/stark_prime_kg',
                 drugbank_xml_path: str = "/rxrx/data/user/lu.zhu/outgoing/data/drugbank20231211/full_database.xml"):
        """
        Load a knowledge graph from saved files.
        
        Args:
            save_address (str): Directory path where the KG files are saved
            enc_language_model_name: Name of the sentence-transformer model to load. Default is 'pritamdeka/S-PubMedBERT-MS-MARCO'
                       which is a biomedical language model trained on PubMed data. Other options include:
                       - 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' (default, best for biomedical concepts)
                       - 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb' (BioBERT fine-tuned on multiple tasks)
                       - 'pritamdeka/S-PubMedBERT-MS-MARCO' (PubMedBERT fine-tuned for biomedical search)
                       - 'michiyasunaga/BioLinkBERT-base' (BioLinkBERT for biomedical entity linking)
                       - 'all-MiniLM-L6-v2' (default sentence transformer model)
            node_emb_save_address (str): Directory path where the node embeddings are saved
            drugbank_xml_path (str): Path to DrugBank XML file (only used when DrugBankSearcher is first accessed)
        """
        self.save_address = save_address
        
        # Load tensors
        self.edge_index = torch.load(os.path.join(save_address, 'edge_index.pt')) # shape: (2, #num_edges)
        self.edge_types = torch.load(os.path.join(save_address, 'edge_types.pt')) # shape: (#num_edges)
        self.node_types = torch.load(os.path.join(save_address, 'node_types.pt'))
        
        # Load dictionaries
        with open(os.path.join(save_address, 'edge_type_dict.pkl'), 'rb') as f:
            self.edge_type_dict = pickle.load(f)
        with open(os.path.join(save_address, 'node_attr_dict.pkl'), 'rb') as f:
            self.node_attr_dict = pickle.load(f)
        with open(os.path.join(save_address, 'node_type_dict.pkl'), 'rb') as f:
            self.node_type_dict = pickle.load(f)
        
        # Load node info
        with open(os.path.join(save_address, 'node_info.json'), 'r') as f:
            self.node_info = json.load(f)
        
        # Load doc info
        with open(os.path.join(save_address, 'doc_info_with_rel.pkl'), 'rb') as f:
            self.doc_info_with_rel = pickle.load(f)
        with open(os.path.join(save_address, 'doc_info_without_rel.pkl'), 'rb') as f:
            self.doc_info_without_rel = pickle.load(f)
        
        # self.num_candidates = len(self.doc_info_with_rel)
        
        self.num_nodes = self.node_types.shape[0]
        self.num_edges = self.edge_index.shape[1]
        
        self.node_emb_save_address = node_emb_save_address
        
        self.enc_language_model_name = enc_language_model_name
        self.enc_language_model = SentenceTransformer(self.enc_language_model_name)
        
        self.node_embeddings = self.create_node_embeddings(use_rels=True)
        self.edge_type_embeddings = self.create_edge_embeddings()
        
        self.pyg_graph = torch_geometric.data.Data(
            x=self.node_embeddings,
            edge_index=self.edge_index,
            edge_type=self.edge_types,
            node_type=self.node_types,
        )

        self.kg_node_name_dict = {node['name']: idx for idx, node in self.node_info.items()}
        
        # Store DrugBank XML path for lazy loading
        self._drugbank_xml_path = drugbank_xml_path
        self._drugbank_searcher = None

        # self.edge_map = {(int(src), int(dst)): idx for idx, (src, dst) in enumerate(zip(self.edge_index[0], self.edge_index[1]))}
        self._ensure_ner_caches()
    
    def get_drugbank_searcher(self):
        """
        Get the DrugBankSearcher instance, initializing it lazily if needed.
        
        Returns:
            DrugBankSearcher instance or None if initialization fails
        """
        global _global_drugbank_searcher
        
        if _global_drugbank_searcher is None:
            try:
                from explain.starkprimekg.drugbanksearcher import DrugBankSearcher
                _global_drugbank_searcher = DrugBankSearcher(self._drugbank_xml_path)
                print("DrugBankSearcher initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize DrugBankSearcher: {e}")
                return None
        
        return _global_drugbank_searcher
    
    @property
    def drugbank_searcher(self):
        """
        Property to access DrugBankSearcher with lazy loading.
        
        Returns:
            DrugBankSearcher instance or None if not available
        """
        return self.get_drugbank_searcher()
    
    def create_node_embeddings(self, use_rels: bool = False, batch_size: int = 128, device=None) -> torch.Tensor:
        """Create node and edge embeddings from textual descriptions using the language model.
        
        Args:
            use_rels (bool): Whether to use relationship information for node embeddings
        """
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.enc_language_model.to(self.device)
        
        address = os.path.join(self.node_emb_save_address, self.enc_language_model_name, 'node_embeddings.pt')
        
        if self.node_emb_save_address is not None and os.path.exists(address):
            print(f"Loading embeddings from {address}")
            node_embeddings = torch.load(address).to(self.device)
            return node_embeddings
        
        embeddings = []
        for i in tqdm(range(0, self.num_nodes, batch_size), desc="Creating embeddings"):
            batch_texts = self.doc_info_with_rel[i:i + batch_size] if use_rels else self.doc_info_without_rel[i:i + batch_size]
            batch_embeddings = self.enc_language_model.encode(batch_texts, 
                                               convert_to_tensor=True, 
                                               show_progress_bar=False)
            embeddings.append(batch_embeddings)

        node_embeddings = torch.cat(embeddings, dim=0)
        
        # Save embeddings
        if self.node_emb_save_address is not None:
            os.makedirs(os.path.dirname(address), exist_ok=True)
            torch.save(node_embeddings.cpu(), address)
        
        print(f"Embeddings created and saved to {address}")
        
        return node_embeddings
    
    def create_edge_embeddings(self):
        """Create edge embeddings from textual descriptions using the language model.
        edge embeddings would be inefficient to saved individually, we will keep edge embedding per edge type and when reading these embeddings can be casted
        """
        edge_type_embeddings = []
        for i in range(len(self.edge_type_dict)):
            edge_type_text = self.edge_type_dict[i]
            edge_type_embedding = self.enc_language_model.encode(edge_type_text, convert_to_tensor=True, show_progress_bar=False)
            edge_type_embeddings.append(edge_type_embedding)
        
        return torch.stack(edge_type_embeddings, dim=0)
    
    
    def subgraph_to_text(
        self,
        edge_ids: list,
        max_edges: int = 500
    ) -> str:
        """Convert a subgraph to a textual description suitable for LLMs.

        Args:
            edge_ids: List of edge IDs in the subgraph
            max_edges: Maximum number of edges to include in description

        Returns:
            String containing natural language description of the subgraph
        """
        if len(edge_ids) > max_edges:
            edge_ids = edge_ids[:max_edges]
            
        node_ids = self.pyg_graph.edge_index[:, edge_ids].flatten().unique().tolist()
        
        # Start with overview
        lines = []
        lines.append(f"The retrieved subgraph contains {len(node_ids)} nodes and {len(edge_ids)} edges.")
        
        # Add node descriptions
        lines.append("\nNodes:")
        for node_id in node_ids:
            lines.append(f"- Node ID: {node_id}, Doc Info: {self.doc_info_without_rel[node_id]}")
                
        # Add edge descriptions
        lines.append("\nConnections:")
        for edge_id in edge_ids:
            src, dst = self.edge_index[:, edge_id].tolist()
            edge_type = self.edge_type_dict[self.edge_types[edge_id].item()]
            lines.append(f"- Node ID {src} is connected to Node ID {dst} with edge type: {edge_type}")
            
        return "\n".join(lines)

    def text_to_embedding(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Convert text strings into numerical embeddings using the loaded language model.
        
        Args:
            texts: Single text string or list of textual descriptions for nodes
            
        Returns:
            torch.Tensor: Embedding vector for single text, or matrix of embeddings (one row per text) for multiple texts
            
        Raises:
            RuntimeError: If the language model hasn't been loaded
        """
        if self.enc_language_model is None:
            raise RuntimeError("Language model not loaded. Call load_language_model() first.")
            
        embeddings = self.enc_language_model.encode(texts, convert_to_tensor=True)
        return embeddings

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
        if self.enc_language_model is None:
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
        # node_ids = list(self.nodes_textual_descriptions.keys())
        node_embeddings = self.node_embeddings
        
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
            results[idx] = float(similarities[idx])
            
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

    def find_edge_index(self, target_edge: torch.Tensor) -> List[int]:
        """Find the edge index of the given edge index.

        Args:
            target_edge_index: Target edge index tensor
        """

        BASE = int(self.edge_index.max()) + 1  # must be > any value in mat
        keys = self.edge_index[0] * BASE + self.edge_index[1]  # shape (N,)

        # Sort once
        order = torch.argsort(keys)
        keys_sorted = keys[order]

        # Query
        target_key = target_edge[0] * BASE + target_edge[1]
        pos = torch.searchsorted(keys_sorted, target_key)

        if pos < self.num_edges and keys_sorted[pos] == target_key:
            idx = order[pos]   # found in original matrix
        else:
            idx = -1           # not found

        return idx.item()

    def edge_index_to_id(self, target_edge_index: torch.Tensor) -> List[int]:
        """Convert edge index to node IDs.
        
        Args:
            edge_index: Edge index tensor
        """
        # Build a dictionary mapping (src, dst) to its position in edge_index

        # TODO: solve OOM error
        edge_ids = [self.find_edge_index(edge) for edge in target_edge_index.transpose(1,0)]

        return edge_ids

            
    def get_subgraph_from_nodes(self, node_indices: List[int]) -> List:
        """Get the minimal subgraph from the given node indices.
        
        Args:
            node_indices: List of node indices to include in the subgraph
            
        Returns:
            Edge indices of the subgraph
        """

        G = to_networkx(self.pyg_graph, to_undirected=True)
        T = torch.as_tensor(node_indices, dtype=torch.long).tolist()
        T = list(dict.fromkeys(int(t) for t in T))  # unique, preserve order
        # 2) Shortest paths & distances among terminals
        paths, dists = {}, {}
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                u, v = T[i], T[j]
                p = nx.shortest_path(G, u, v)
                d = len(p) - 1
                paths[(u, v)] = p
                dists[(u, v)] = d

        # 3) Choose terminal pairs
        # if method == "all_pairs":
        #     pairs = list(dists.keys())
        MC = nx.Graph()
        MC.add_nodes_from(T)
        for (u, v), d in dists.items():
            MC.add_edge(u, v, weight=d)
        mst = nx.minimum_spanning_tree(MC, weight="weight")
        pairs = list(mst.edges())
        # else:
        #     raise ValueError("method must be 'mst' or 'all_pairs'.")

        # 4) Union the original-graph shortest paths for chosen pairs
        nodes_keep = set(T)
        for (u, v) in pairs:
            p = paths.get((u, v), paths.get((v, u)))
            nodes_keep.update(p)

        # 5) Extract induced subgraph in PyG (relabels node ids to 0..n-1)
        mask = torch.zeros(self.pyg_graph.num_nodes, dtype=torch.bool)
        mask[list(nodes_keep)] = True
        edge_index_sub, _ = pyg_subgraph(
            mask, self.pyg_graph.edge_index, getattr(self.pyg_graph, 'edge_attr', None),
            relabel_nodes=False
        )

        edge_ids = self.edge_index_to_id(edge_index_sub)

        return edge_ids

    def get_k_hop_neighbors(self, node_index: int, k: int=1) -> List[int]:
        """Get the k-hop neighbors of the given node index.
        
        Args:
            node_index: Index of the node to get neighbors for
            k: Number of hops to consider
        """
        edge_index_sub, _ = k_hop_subgraph(node_index, k, self.edge_index)
        return edge_index_sub

    def _ensure_ner_caches(self):
        """Build one-time caches on the KG instance for fast lookups."""
        if not hasattr(self, '_kg_node_name_lower'):
            self._kg_node_name_lower = {name.lower(): idx for name, idx in self.kg_node_name_dict.items()}
        if not hasattr(self, '_alias_to_node_index'):
            alias_map = {}
            for node in self.node_info.values():
                if node.get('type') != 'gene/protein':
                    continue
                aliases = node.get('details', {}).get('alias')
                if not aliases:
                    continue
                if not isinstance(aliases, list):
                    aliases = [aliases]
                node_idx = self.kg_node_name_dict.get(node['name'])
                for alias in aliases:
                    alias_map[alias] = node_idx
            self._alias_to_node_index = alias_map
        if not hasattr(self, '_drugbank_id_to_name'):
            self._drugbank_id_to_name = {node['id']: node['name'] for node in self.node_info.values() if node.get('type') == 'drug'}
        if not hasattr(self, '_pubchem_cache'):
            self._pubchem_cache = {}
        if not hasattr(self, '_ner_entity_cache'):
            self._ner_entity_cache = {}