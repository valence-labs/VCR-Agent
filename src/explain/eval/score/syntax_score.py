import os
from typing import Any
import re


from explain.eval.score._base import Evaluator
from explain.util import load_data


class SyntaxEvaluator(Evaluator):
    """
    Checks the syntax and schema gates of the generated explanations.
    """
    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.allowed_primitives = self.get_allowed_primitives()

    def get_primitives_from_structure_hypothesis(self, structure_hypothesis: str) -> list[str]:
        """
        Return list of function names appearing in top-level call style:
        name( ... )  (not preceded by '.' or part of attribute chain).
        """
        pattern = r'(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*)\s*\('
        primitives = re.findall(pattern, structure_hypothesis)
        return primitives

    def get_allowed_primitives(self) -> list[str]:
        """
        Get the allowed primitives.
        """
        DATA_DIR = os.getenv("DATA_DIR")
        DATA_DIR = '../../../emmanuel.noutahi/project/outgoing/hooke/hooke-explain/'
        action_primitives, _, _, _ = load_data(DATA_DIR)
        action_primitives = [primitive['action'] for primitive in action_primitives]
        return action_primitives

    def primitive_validity(self, structure_hypothesis: str) -> tuple[float, dict[str, Any]]:
        """
        Check whether the generated primitives are in the allowed set.
        """
        primitives = self.get_primitives_from_structure_hypothesis(structure_hypothesis)
        score = sum([primitive in self.allowed_primitives for primitive in primitives])/len(primitives)
        return score

