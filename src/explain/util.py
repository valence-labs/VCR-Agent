import os
import json
from flair.data import Sentence
from flair.models.prefixed_tagger import PrefixedSequenceTagger
from tqdm import tqdm
from explain.kg.kg import KnowledgeGraph
from collections import Counter
import ast
import pandas as pd

global perturbation_ner_mapping
global tagger

tagger = PrefixedSequenceTagger.load("hunflair/hunflair2-ner")
# TODO: need to be fixed for the entire dataset (fix perturbation index also)
perturbation_ner_mapping = json.load(open('data/perturbation_ner_mapping.json', 'r'))
perturbation_ner_mapping_data_indices = [item['index'] for item in perturbation_ner_mapping]

def load_data(data_dir):
    # load the action primitives
    with open(os.path.join(data_dir, "action_primitives.json")) as f:
        action_primitives = json.load(f)
    pert_path = os.path.join('../../../emmanuel.noutahi/project/outgoing/hooke/hooke-explain/', "data", "perturbations_old.json")
    with open(pert_path) as f:
        perturbation_cell_context = json.load(f)
    report_template = open(os.path.join(data_dir, "templates/generate-report.txt")).read()
    structre_explain_template = open(os.path.join(data_dir, "templates/structure-explain.txt")).read()
    return action_primitives, perturbation_cell_context, report_template, structre_explain_template

def extract_entity_from_text(text, pert_idx):
    """
    Extract the entity from the perturbation.
    """
    if pert_idx in perturbation_ner_mapping_data_indices:
        result = {**perturbation_ner_mapping[pert_idx]['perturbation_entity'], **perturbation_ner_mapping[pert_idx]['context_entity']}
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


def map_perturbation_to_kg_entity(perturbations):
    """
    Generates the perturbation-KG entity mapping.
    """
    kg = KnowledgeGraph()

    perturbation_dict = {}

    kg_node_info = kg.node_info
    kg_node_name_dict = {node['name']: idx for idx, node in kg_node_info.items()}
    for pert_idx, pert in enumerate(tqdm(perturbations)):
        perturbation_dict[pert_idx] = {}
        context = pert['context']
        perturbation = pert['perturbation']
        perturbation_dict[pert_idx]['kg_info'] = []
        perturbation_dict[pert_idx]['org_data'] = []
        
        for key, value in context.items():
            if value in kg_node_name_dict.values():
                node_index = kg_node_name_dict[value]
                kg_info = kg_node_info[node_index]
                kg_info['node_index'] = node_index
                perturbation_dict[pert_idx]['kg_info'].append(kg_node_info[node_index])
                perturbation_dict[pert_idx]['org_data'].append({'type': 'context', 'key': key, 'value': value})
                print(key, value, node_index)
        
        for key, value in perturbation.items():
            if isinstance(value, list):
                value = value[0]
            if value in kg_node_name_dict.keys():
                node_index = kg_node_name_dict[value]
                kg_info = kg_node_info[node_index]
                kg_info['node_index'] = node_index
                perturbation_dict[pert_idx]['kg_info'].append(kg_info)
                perturbation_dict[pert_idx]['org_data'].append({'type': 'perturbation', 'key': key, 'value': value})
                print(key, value, node_index)


    json.dump(perturbation_dict, open('data/starkprimeKG/perturbation_kg_mapping.json', 'w'))

def json_to_csv(file_name):
    data = json.load(open(file_name, 'r'))
    df = pd.DataFrame(data)
    df.to_csv(file_name.replace('.json', '.csv'), index=False)


json_to_csv('output/structure_explain/baseline/baseline_two_step_claude_[]_1.json')
json_to_csv('output/structure_explain/baseline/baseline_two_step_gemini_[]_1.json')
json_to_csv("output/structure_explain/multi_tool/multi_tool_['kg-ner', 'harmonizome', 'wikipedia', 'paperqa_list']_1.json")

