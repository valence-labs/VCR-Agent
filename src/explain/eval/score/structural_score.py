from explain.eval.score.score_util import dag_to_networkx_graph
import numpy as np
class StructuralEvaluator:
    def __init__(self,  **kwargs):
        pass


    def edge_type_accuracy(self, gt_dag: str, gen_dag: str) -> float:
        """
        Check whether the edge types are correct.
        """
        gt_graph = dag_to_networkx_graph(gt_dag)
        if gen_dag is np.nan:
            return 0
        gen_graph = dag_to_networkx_graph(gen_dag)


        gt_edges = {(src, tgt, rel) for src, tgt, rel in gt_graph.edges(data='relation')}
        gen_edges = {(src, tgt, rel) for src, tgt, rel in gen_graph.edges(data='relation')}

        correct_edges = gt_edges & gen_edges
        
        # current version returns f1 score but we can add recall and precision later
        precision = len(correct_edges) / len(gen_edges) if len(gen_edges) > 0 else 0
        recall = len(correct_edges) / len(gt_edges) if len(gt_edges) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return round(f1, 4)