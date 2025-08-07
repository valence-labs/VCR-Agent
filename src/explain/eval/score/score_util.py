import networkx as nx
import re



def dag_to_networkx_graph(dag: str) -> nx.DiGraph:
    """
    Convert a DAG string to a NetworkX graph.
    """

    edge_pattern = re.compile(r'edge\("(?P<src>n\d+)",\s*"(?P<tgt>n\d+)",\s*relation="(?P<rel>\w+)"\)')
    edges = edge_pattern.findall(dag)
    graph = nx.DiGraph()
    for src, tgt, rel in edges:
        graph.add_edge(src, tgt, relation=rel)
    return graph

def get_primitives_from_structure_hypothesis(structure_hypothesis: str) -> list[str]:
    """
    Return list of function names appearing in top-level call style:
    name( ... )  (not preceded by '.' or part of attribute chain).
    """
    pattern = r'(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*)\s*\('
    primitives = re.findall(pattern, structure_hypothesis)
    return primitives