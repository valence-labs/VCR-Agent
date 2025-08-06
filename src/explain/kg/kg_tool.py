from langchain.tools import BaseTool
from explain.kg.kg_utils import KnowledgeGraph

class KGNeighborTool(BaseTool):
    name: str = "knowledge_graph_neighbor"
    description: str = "A tool to query the knowledge graph for the neighborhood of a given gene"
    kg: KnowledgeGraph = KnowledgeGraph(graph_cfg={"graph0": {"graph_type": "string", "reduce2perts": True,
    "mode": "top_3", "norm_weights": True}})

    def _run(self, query: str) -> str:
        return self.kg.find_nodes_from_query(query)