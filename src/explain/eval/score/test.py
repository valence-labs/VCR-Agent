import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from explain.eval.score.syntax_score import SyntaxEvaluator
from explain.eval.score.accuracy_score import AccuracyEvaluator
from explain.eval.score.structure_explain import StructureExplain


print(os.getcwd())
# load data
with open(os.path.join('output/structure_explain/vanilla' , "vanilla_[].json"), "r") as f:
    gen_data = json.load(f)

gt_data = pd.read_csv('/mnt/ps/home/CORP/emmanuel.noutahi/project/outgoing/hooke/hooke-explain/data/structure-explain-results-v1.csv')
gt_data = gt_data.to_dict(orient='records')
# structure explain object
structure_explain = StructureExplain(gen_data)
gt_structure_explain = StructureExplain(gt_data)
syntax_evaluator = SyntaxEvaluator()
accuracy_evaluator = AccuracyEvaluator()

scores = {}
scores['syntax'] = []
scores['token_accuracy'] = []

for gt_structure_hypothesis, gen_structure_hypothesis in zip(tqdm(gt_structure_explain.structure_hypothesis), structure_explain.structure_hypothesis):
    # syntax score
    score = syntax_evaluator.primitive_validity(gen_structure_hypothesis) 
    scores['syntax'].append(score)
    # token accuracy for structure hypothesis
    score = accuracy_evaluator.token_based_accuracy(gt_structure_hypothesis, gen_structure_hypothesis, 'structure_hypothesis')
    scores['token_accuracy/structure_hypothesis'].append(score)

# token accuracy for paragraph
for gt_paragraph, gen_paragraph in zip(tqdm(gt_structure_explain.paragraph), structure_explain.paragraph):
    score = accuracy_evaluator.token_based_accuracy(gt_paragraph, gen_paragraph, 'paragraph')
    scores['token_accuracy/paragraph'].append(score)

print('syntax', np.mean(scores['syntax']))
for key in scores['token_accuracy/structure_hypothesis'][0].keys():
    print(key, round(np.mean([score[key] for score in scores['token_accuracy/structure_hypothesis']]), 4))
for key in scores['token_accuracy/paragraph'][0].keys():
    print(key, round(np.mean([score[key] for score in scores['token_accuracy/paragraph']]), 4))




