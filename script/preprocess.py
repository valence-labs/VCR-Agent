import json
import os
import re
from copy import deepcopy
import argparse
from explain.util import map_perturbation_to_ner
from tqdm import tqdm
import pandas as pd
from explain.utils.chem.pubchem import PubChemClient


def natural_key(s):
    # split into digit chunks and non-digit chunks
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def preprocess(input_path, preprocessed_path, mode='ner-pubchem'):
    '''
    input_path: input perturbation path
    preprocessed_file_path: preprocessed perturbation path
    '''
    # NER mapping

    if 'ner' in mode:
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
            perturbations = df.to_dict(orient='records')
        else:
            perturbations = json.load(open(os.path.join(input_path), 'r'))
        map_perturbation_to_ner(perturbations, preprocessed_path)
    # PubChem info mapping
    if 'pubchem' in mode:
        processed_perturbations = json.load(open(preprocessed_path, 'r'))
        map_pubchem_info(processed_perturbations, preprocessed_path)

def map_pubchem_info(perturbations, preprocessed_path):
    pubchem_client = PubChemClient()
    for i, perturbation in enumerate(tqdm(perturbations)):
        for pert in perturbation['perturbation']['perturbations']:
            if pert['perturbation type'] == 'chemical':
                if 'pubchem_info' not in pert.keys():
                    pubchem_info = pubchem_client.get_compound_info(smiles=pert['smiles'])
                    pert['pubchem_info'] = pubchem_info
                if 'pubchem_info' in pert.keys():
                    pubchem_info = pert['pubchem_info']
                    if 'name' in pubchem_info.keys():
                        # Treat the first synonym without number as the drug name (normally USAN/INN)
                        synonym = [s for s in pubchem_info['synonyms'] if not any(c.isdigit() for c in s)]
                        if len(synonym) > 0:
                            synonym = synonym[0]
                            if synonym not in perturbation['perturbation_entity']:
                                perturbation['perturbation_entity'][synonym] = 'Chemical'

        if i % 50 == 0 or i == len(perturbations) - 1:
            with open(preprocessed_path, 'w') as f:
                json.dump(perturbations, f)

def preprocess_filter_with_reservation(input_path='/rxrx/data/user/yunhui.jang/outgoing/hunflair'):
    '''
    Filter with reserved bc experiment or reserved bc compound
    '''
    reserved_bc_expt_count = 0
    reserved_bc_comp_count = 0
    reserved_all_count = 0
    total_count = 0
    for file in tqdm(sorted(os.listdir(input_path), key=natural_key)):
        if 'rxrx' in file:
            index = file.split('_')[-1].split('.')[0]
            perturbations_rxrx = json.load(open(os.path.join(input_path, file), 'r'))
                # remove duplicates
            perturbations_rxrx = dedupe_ignoring_index(perturbations_rxrx, keep="first", drop_index_in_output=True)
            reserved_bc_expt = [pert for pert in perturbations_rxrx if pert['perturbation']['all_reserved_bc_expt']]
            reserved_bc_comp = [pert for pert in perturbations_rxrx if pert['perturbation']['reserved_bc_comp']]
            reserved_all = [pert for pert in perturbations_rxrx if pert['perturbation']['all_reserved_bc_expt'] and pert['perturbation']['reserved_bc_comp']]
            reserved_bc_expt_count += len(reserved_bc_expt)
            reserved_bc_comp_count += len(reserved_bc_comp)
            reserved_all_count += len(reserved_all)
            total_count += len(perturbations_rxrx)
            perturbation_rxrx_filtered = [pert for pert in perturbations_rxrx if pert['perturbation']['all_reserved_bc_expt'] or pert['perturbation']['reserved_bc_comp']]

            with open(f'/rxrx/data/user/yunhui.jang/outgoing/preprocessed/preprocessed_perturbation_ner_mapping_rxrx_{index}.json', 'w') as f:
                print(len(perturbation_rxrx_filtered), len(perturbations_rxrx))
                json.dump(perturbation_rxrx_filtered, f)
        
    print(reserved_bc_expt_count, reserved_bc_comp_count, reserved_all_count, total_count)
    print(reserved_bc_expt_count/total_count, reserved_bc_comp_count/total_count, reserved_all_count/total_count)


def preprocess_filter_with_ner_matching(num_samples=30000, input_path='/rxrx/data/user/yunhui.jang/outgoing/hunflair'):
    data = []
    for file in tqdm(sorted(os.listdir(input_path), key=natural_key)[:2]):
        if 'rxrx' in file:
            perturbations_rxrx = json.load(open(os.path.join(input_path, file), 'r'))
            perturbations_rxrx = dedupe_ignoring_index(perturbations_rxrx, keep="first", drop_index_in_output=True)
            perturbations_rxrx = [pert for pert in perturbations_rxrx if not pert['perturbation']['all_reserved_bc_expt'] and not pert['perturbation']['reserved_bc_comp']]
            data.extend(perturbations_rxrx)
    
    data_with_chemical = [d for d in data if 'Chemical' in d['perturbation_entity'].values()]
    # TODO
    data_with_chemical = sorted(data_with_chemical, key=lambda x: len(x['perturbation_entity']))
    data_with_chemical = data_with_chemical[:num_samples]

    return data_with_chemical
        

def _freeze(obj, drop_keys=frozenset({"index"}), list_as_multiset=False):
        """
        Make a hashable, canonical representation of `obj` while dropping keys in `drop_keys`.
        - dict: sort keys; skip any key in drop_keys; recurse on values
        - list/tuple: keep order (or sort if list_as_multiset=True); recurse on items
        - set: sort canonically
        - other: return as-is (must be hashable)
        """
        if isinstance(obj, dict):
            return ("dict", tuple(
                (k, _freeze(v, drop_keys, list_as_multiset))
                for k, v in sorted(obj.items())
                if k not in drop_keys
            ))
        elif isinstance(obj, list):
            items = [_freeze(v, drop_keys, list_as_multiset) for v in obj]
            return ("list", tuple(sorted(items)) if list_as_multiset else tuple(items))
        elif isinstance(obj, tuple):
            return ("tuple", tuple(_freeze(v, drop_keys, list_as_multiset) for v in obj))
        elif isinstance(obj, set):
            return ("set", tuple(sorted(_freeze(v, drop_keys, list_as_multiset) for v in obj)))
        else:
            return obj  # assume hashable (str, int, float, bool, None)

def dedupe_ignoring_index(records, keep="first", drop_index_in_output=False, list_as_multiset=False):
    """
    Deduplicate a list of dicts by all fields except 'index'.
    - keep: 'first' | 'last' occurrence to retain
    - drop_index_in_output: if True, remove 'index' from the returned dicts
    - list_as_multiset: if True, list order won't affect equality
    """
    seen = {}
    for i, rec in enumerate(records):
        key = _freeze(rec, drop_keys=frozenset({"index"}), list_as_multiset=list_as_multiset)
        if keep == "first":
            seen.setdefault(key, i)  # keep earliest index
        elif keep == "last":
            seen[key] = i
        else:
            raise ValueError("keep must be 'first' or 'last'")

    # materialize selected records
    out = [deepcopy(records[i]) for i in sorted(seen.values())]
    if drop_index_in_output:
        def _strip(o):
            if isinstance(o, dict):
                return {k: _strip(v) for k, v in o.items() if k != "index"}
            elif isinstance(o, list):
                return [_strip(v) for v in o]
            elif isinstance(o, tuple):
                return tuple(_strip(v) for v in o)
            else:
                return o
        out = [_strip(r) for r in out]
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='3')
    parser.add_argument('--input_path', type=str, default='data/rxrx/reserved/reserved_perturbations_batch_0.json')
    parser.add_argument('--preprocessed_path', type=str, default='data/rxrx/reserved/reserved_perturbation_ner_mapping_batch_0.json')
    parser.add_argument('--mode', type=str, default='ner-pubchem', choices=['ner-pubchem', 'pubchem', 'ner'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    preprocess(args.input_path, args.preprocessed_path, mode=args.mode)