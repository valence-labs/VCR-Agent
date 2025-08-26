import os
import json
from flair.data import Sentence
from flair.models.prefixed_tagger import PrefixedSequenceTagger
from tqdm import tqdm
from explain.kg.kg import KnowledgeGraph
from collections import Counter
import ast

tagger = PrefixedSequenceTagger.load("hunflair/hunflair2-ner")

def load_data(data_dir):
    # load the action primitives
    with open(os.path.join(data_dir, "action_primitives.json")) as f:
        action_primitives = json.load(f)
    pert_path = os.path.join('../../../emmanuel.noutahi/project/outgoing/hooke/hooke-explain/', "data", "perturbations.json")
    with open(pert_path) as f:
        perturbation_cell_context = json.load(f)
    report_template = open(os.path.join(data_dir, "templates/generate-report.txt")).read()
    structre_explain_template = open(os.path.join(data_dir, "templates/structure-explain.txt")).read()
    return action_primitives, perturbation_cell_context, report_template, structre_explain_template

def extract_entity_from_text(text):
    """
    Extract the entity from the perturbation.
    """

    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = sentence.get_spans('ner')
    result = {}
    for entity in entities:
        result[entity.text] = entity.tag

    return result

def map_perturbation_to_ner(perturbations):
    """
    Generates the perturbation-NER mapping.
    """

    perturbation_ner_dict = []
    for pert_idx, perturbation in enumerate(tqdm(perturbations)):
        cur_dict = {}
        cur_dict['index'] = pert_idx
        perturbation = ast.literal_eval(perturbation)
        cur_dict['perturbation'] = perturbation
        perturbation_partial_text = json.dumps(perturbation['perturbation'], indent=4)
        perturbation_entity = extract_entity_from_text(perturbation_partial_text)
        context_text = json.dumps(perturbation['context'], indent=4)
        context_entity = extract_entity_from_text(context_text)
        cur_dict['perturbation_entity'] = perturbation_entity
        cur_dict['context_entity'] = context_entity
        perturbation_ner_dict.append(cur_dict)


    with open('data/perturbation_ner_mapping.json', 'w') as f:
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