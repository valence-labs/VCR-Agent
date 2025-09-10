import os
import json
from flair.data import Sentence
from flair.models.prefixed_tagger import PrefixedSequenceTagger
from tqdm import tqdm
import ast
import pandas as pd

global tagger

tagger = PrefixedSequenceTagger.load("hunflair/hunflair2-ner")


def set_perturbation_ner_mapping(mapping, indices):
    global perturbation_ner_mapping
    global perturbation_ner_mapping_data_indices
    perturbation_ner_mapping = mapping
    perturbation_ner_mapping_data_indices = indices

def load_data(data_dir, pert_path):
    # load the action primitives
    with open(os.path.join(data_dir, "action_primitives.json")) as f:
        action_primitives = json.load(f)
    if len(pert_path) > 0:
        with open(pert_path) as f:
            perturbation_cell_context = json.load(f)
    else:
        perturbation_cell_context = []
    report_template = open(os.path.join(data_dir, "templates/generate-report.txt")).read()
    structre_explain_template = open(os.path.join(data_dir, "templates/structure-explain.txt")).read()
    return action_primitives, perturbation_cell_context, report_template, structre_explain_template

def extract_entity_from_text(text, pert_idx):
    """
    Extract the entity from the perturbation.
    """
    
    if perturbation_ner_mapping is not None and pert_idx in perturbation_ner_mapping_data_indices:
        index = perturbation_ner_mapping_data_indices.index(pert_idx)
        result = {**perturbation_ner_mapping[index]['perturbation_entity'], **perturbation_ner_mapping[index]['context_entity']}
    else:
        result = {}
        sentence = Sentence(text)
        tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        for entity in entities:
            result[entity.text] = entity.tag

    return result

def map_perturbation_to_ner(perturbations, file_name):
    """
    Generates the perturbation-NER mapping.
    """
    if os.path.exists(file_name):
        print(f'{file_name} already exists')
        perturbation_ner_dict = json.load(open(file_name, 'r'))
    else:
        perturbation_ner_dict = []
    index_set = set([pert_dict['index'] for pert_dict in perturbation_ner_dict])
    set_perturbation_ner_mapping(perturbation_ner_dict, index_set)
    for pert_idx, perturbation in enumerate(tqdm(perturbations)):
        if perturbation is None or pert_idx in index_set:
            continue
        cur_dict = {}
        cur_dict['index'] = pert_idx
        if isinstance(perturbation, str):
            perturbation = ast.literal_eval(perturbation)
        cur_dict['perturbation'] = perturbation
        if 'tahoe' in file_name or 'rxrx' in file_name:
            perturbation_partial_text = json.dumps(perturbation['perturbations'], indent=4)
        else:
            perturbation_partial_text = json.dumps(perturbation['perturbation'], indent=4)
        perturbation_entity = extract_entity_from_text(perturbation_partial_text, pert_idx)
        context_text = json.dumps(perturbation['context'], indent=4)
        context_entity = extract_entity_from_text(context_text, pert_idx)
        cur_dict['perturbation_entity'] = perturbation_entity
        cur_dict['context_entity'] = context_entity
        perturbation_ner_dict.append(cur_dict)

        if pert_idx % 100 == 0 or pert_idx == len(perturbations) - 1:
            perturbation_ner_dict = sorted(perturbation_ner_dict, key=lambda x: x['index'])
            with open(file_name, 'w') as f:
                json.dump(perturbation_ner_dict, f)
    perturbation_ner_dict = sorted(perturbation_ner_dict, key=lambda x: x['index'])
    with open(file_name, 'w') as f:
        json.dump(perturbation_ner_dict, f)

def json_to_csv(file_name):
    data = json.load(open(file_name, 'r'))
    df = pd.DataFrame(data)
    df.to_csv(file_name.replace('.json', '.csv'), index=False)


