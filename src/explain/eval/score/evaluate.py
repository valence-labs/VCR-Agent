import json

import numpy as np
import wandb
from tqdm import tqdm

from explain.eval.rubrics.correctness import CorrectnessRubric
from explain.eval.rubrics.plausibility import PlausibilityRubric
from explain.eval.score.accuracy_score import AccuracyEvaluator
from explain.eval.score.structural_score import StructuralEvaluator
from explain.eval.score.structure_explain import StructureExplain
from explain.eval.score.syntax_score import SyntaxEvaluator

# structure explain object


def evaluate(gen_data, gt_data, score_file_name, metrics=['syntax', 'token_accuracy', 'schema_validity', 'id_coherence', 'dag_well_formed', 'edge_type_accuracy', 'primitive_level_f1', 'argument_similarity',
'llm/correctness', 'llm/plausibility']):

    # load evaluators
    structure_explain = StructureExplain(gen_data)
    gt_structure_explain = StructureExplain(gt_data)
    syntax_evaluator = SyntaxEvaluator()
    accuracy_evaluator = AccuracyEvaluator()
    structural_evaluator = StructuralEvaluator()
    correctness_rubric = CorrectnessRubric()
    plausibility_rubric = PlausibilityRubric()

    scores = []

    for i, (gt, gen) in enumerate(tqdm(zip(gt_structure_explain, structure_explain, strict=False), total=len(gt_structure_explain))):

        if str(gt['index']) != str(gen['index']):
            raise ValueError(f'Index mismatch: {gt["index"]} != {gen["index"]}')

        cur_score_dict = {'index': i}


        for key, value in gen.items():
            gen[key] = str(value)


        if 'syntax' in metrics:
            score = syntax_evaluator.primitive_validity(gen['explain'])
            cur_score_dict['syntax'] = score
        # token accuracy for structure hypothesis
        if 'token_accuracy' in metrics:
            score = accuracy_evaluator.token_based_accuracy(gt['explain'], gen['explain'], 'structure_hypothesis')
            cur_score_dict['token_accuracy/structure_hypothesis'] = score
            # token accuracy for paragraph
            score = accuracy_evaluator.token_based_accuracy(gt['answer'], gen['answer'], 'paragraph')
            cur_score_dict['token_accuracy/paragraph'] = score
        if 'schema_validity' in metrics:
            # schema validity
            score = syntax_evaluator.schema_validity(gen['raw_response'])
            cur_score_dict['schema_validity'] = score
        # id coherence
        if 'id_coherence' in metrics:
            score = syntax_evaluator.id_coherence(gen['explain'], gen['dag'])
            cur_score_dict['id_coherence'] = score
        # dag well formed
        if 'dag_well_formed' in metrics:
            score = syntax_evaluator.dag_well_formed(gen['dag'])
            cur_score_dict['dag_well_formed'] = score
        # edge type accuracy
        if 'edge_type_accuracy' in metrics:
            score = structural_evaluator.edge_type_accuracy(gt['dag'], gen['dag'])
            cur_score_dict['edge_type_accuracy'] = score
        # primitive level f1
        if 'primitive_level_f1' in metrics:
            score = accuracy_evaluator.primitive_level_f1(gt['explain'], gen['explain'])
            cur_score_dict['primitive_level_f1'] = score
        if 'argument_similarity' in metrics:
            for mode in ['BERTScore', 'SentenceTransformers', 'openai_embeddings']:
                # argument similarity for paragraph
                score = accuracy_evaluator.argument_similarity(gt['answer'], gen['answer'], mode)
                cur_score_dict[f'argument_similarity/paragraph/{mode}'] = score
                # argument similarity for structure hypothesis
                score = accuracy_evaluator.argument_similarity(gt['explain'], gen['explain'], mode)
                cur_score_dict[f'argument_similarity/structure_hypothesis/{mode}'] = score
        if 'llm/correctness' in metrics:
            llm_state = {}

            score = correctness_rubric.judge(prompt=gt['question'], answer=gt['explain'], completion=gen['explain'], state=llm_state)
            cur_score_dict['llm/correctness'] = score['correctness']
            cur_score_dict['llm/correctness/full_response'] = score['full_response']
        if 'llm/plausibility' in metrics:
            llm_state = {}

            score = plausibility_rubric.judge(prompt=gt['question'], completion=gen['explain'], answer='', state=llm_state)
            for key in score.keys():
                cur_score_dict[f'llm/plausibility/{key}'] = score[key]
        scores.append(cur_score_dict)
        if i % 10 == 0:
            with open(score_file_name, 'w') as f:
                json.dump(scores, f, indent=4)

    for key in scores[0].keys():
        if 'token_accuracy' in key:
            for k in scores[0][key].keys():
                score = round(np.mean([score[key][k] for score in scores]), 4)
                wandb.log({f"{key}/{k}": score})
        elif 'full_response' in key:
            pass
        else:
            score = round(np.mean([score[key] for score in scores]), 4)
            wandb.log({key: score})

    wandb.finish()
