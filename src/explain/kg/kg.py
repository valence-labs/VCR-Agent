import torch
import os
import pickle
import json
import torch_geometric
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class KnowledgeGraph:
    def __init__(self, save_address='/rxrx/data/user/hamed.shirzad/outgoing/stark_prime_kg', 
                 enc_language_model_name='pritamdeka/S-PubMedBERT-MS-MARCO', 
                 node_emb_save_address='/rxrx/data/user/hamed.shirzad/outgoing/stark_prime_kg'):
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
        """
        self.save_address = save_address
        
        # Load tensors
        self.edge_index = torch.load(os.path.join(save_address, 'edge_index.pt')) # shape: (2, #num_edges)
        self.edge_types = torch.load(os.path.join(save_address, 'edge_types.pt')) # shape: (#num_edges)
        self.node_types = torch.load(os.path.join(save_address, 'node_types.pt')) # shape: (#num_nodes)
        
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
        
        node_embeddings = self.create_node_embeddings(use_rels=True)
        self.edge_type_embeddings = self.create_edge_embeddings()
        
        self.pyg_graph = torch_geometric.data.Data(
            x=node_embeddings,
            edge_index=self.edge_index,
            edge_type=self.edge_types,
            node_type=self.node_types,
        )
    
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