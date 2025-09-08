import re
import networkx as nx
import numpy as np

from explain.util import load_data
from explain.eval.utils import check_answer_format, is_format_correct
from explain.eval.score.score_util import dag_to_networkx_graph, get_primitives_from_structure_hypothesis

class SyntaxEvaluator:
    """
    Checks the syntax and schema gates of the generated explanations.
    """
    
    def __init__(self,  **kwargs):
        self.allowed_primitives = self.get_allowed_primitives()



    def get_allowed_primitives(self) -> list[str]:
        """
        Get the allowed primitives.
        """
        DATA_DIR = 'data/curation_v1'
        action_primitives, _, _, _ = load_data(DATA_DIR, "")
        action_primitives = [primitive['action'] for primitive in action_primitives]
        return action_primitives

    def primitive_validity(self, structure_hypothesis: str) -> float:
        """
        Check whether the generated primitives are in the allowed set.
        """
        primitives = get_primitives_from_structure_hypothesis(str(structure_hypothesis))
        if len(primitives) == 0:
            return 0
        score = sum([primitive in self.allowed_primitives for primitive in primitives])/len(primitives)
        return score


    def schema_validity(self, raw_response: str) -> float:
        """
        All mandatory tags present & closed; JSON-parsable primitives.
        """
        parsed_sections = check_answer_format(raw_response)
        format_accuracy = is_format_correct(parsed_sections)
        return 1 if format_accuracy else 0

    def id_coherence(self, structure_hypothesis: str, dag: str) -> float:
        """
        Check whether the DAG is coherent.
        """
        ids = re.findall(r'\bid\s*=\s*[\'"](n\d+)[\'"]', structure_hypothesis)
        # for set_context, we don't need to check the id coherence (remove the first line)
        structure_hypothesis = '\n'.join(structure_hypothesis.split('\n')[1:])
        ids = re.findall(r'\bid\s*=\s*[\'"](n\d+)[\'"]', structure_hypothesis)
        # Check every primitive carries a unique id
        id_existence_in_primitive = len(ids) == len(structure_hypothesis.split('\n'))
        # Check whether all IDs referenced in <dag> exist in <explain>
        if dag is np.nan:
            return 0
        ids_in_dag = re.findall(r'\bn\d+\b', dag)
        return 1 if all(dag_id in ids for dag_id in set(ids_in_dag)) and id_existence_in_primitive else 0

    def dag_well_formed(self, dag: str) -> float:
        """
        Check whether the DAG is well-formed.
        """
        if dag is np.nan:
            return 0
        graph = dag_to_networkx_graph(dag)
        return 1 if nx.is_directed_acyclic_graph(graph) else 0