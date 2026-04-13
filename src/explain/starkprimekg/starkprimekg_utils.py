import json
import os
import warnings

from explain.util import extract_entity_from_text
from explain.utils.chem.pubchem import PubChemClient

DATA_DIR = os.environ.get('DATA_DIR', 'data')

# Lazy-loaded MONDO data (only initialized when KG tool is actually used)
_mondo_loaded = False
mondo_node_label_dict = {}
mondo_label_node_dict = {}
mondo_node_synonyms = {}
mondo_synonym_to_label = {}

pubchem_client = PubChemClient()


def _load_mondo():
    """Load MONDO ontology data lazily on first use."""
    global _mondo_loaded, mondo_node_label_dict, mondo_label_node_dict
    global mondo_node_synonyms, mondo_synonym_to_label
    if _mondo_loaded:
        return
    mondo_path = os.path.join(DATA_DIR, 'mondo.json')
    if not os.path.exists(mondo_path):
        warnings.warn(
            f"mondo.json not found at {mondo_path}. Disease synonym matching will be disabled. "
            f"Set DATA_DIR env var to the directory containing mondo.json.",
            stacklevel=2,
        )
        _mondo_loaded = True
        return
    with open(mondo_path) as f:
        mondo_data = json.load(f)
    mondo_nodes = mondo_data['graphs'][0]['nodes']
    mondo_node_label_dict.update({node['lbl']: idx for idx, node in enumerate(mondo_nodes) if 'lbl' in node.keys()})
    mondo_label_node_dict.update({v: k for k, v in mondo_node_label_dict.items()})
    mondo_node_synonyms.update({node['lbl']: node['meta']['synonyms'] for node in mondo_nodes if 'meta' in node.keys() and 'synonyms' in node['meta'].keys()})
    for lbl, syns in mondo_node_synonyms.items():
        try:
            for syn in syns:
                v = syn.get('val')
                if v:
                    mondo_synonym_to_label[v] = lbl
        except Exception:
            continue
    _mondo_loaded = True


def get_kg_info(index, perturbation, tool, kg, is_with_rel, num_neighbor):
    # NER - detect data format from structure
    pert_data = perturbation['perturbation']
    if 'perturbations' in pert_data:
        perturbation_partial_text = json.dumps(pert_data['perturbations'], indent=4)
    else:
        perturbation_partial_text = json.dumps(pert_data['perturbation'], indent=4)
    context_text = json.dumps(perturbation['perturbation']['context'], indent=4)
    perturbation_entity = extract_entity_from_text(perturbation_partial_text, index)
    # Get NER entities
    context_entity = extract_entity_from_text(context_text, index)
    # Get NER node indices
    extracted_graph_info = {}
    perturbation_ner_node_index, perturbation_unmatching_entities = get_ner_node_index(perturbation_entity, kg)
    perturbation_ner_node_index = [n['node'] for n in perturbation_ner_node_index]
    context_ner_node_index, context_unmatching_entities = get_ner_node_index(context_entity, kg)
    context_ner_node_index = [n['node'] for n in context_ner_node_index]
    # Search for similar nodes when matching node does not exist in KG
    unmatching_entity_node_index = []
    for unmatching_entity in perturbation_unmatching_entities+context_unmatching_entities:
        unmatching_entity_node_index.extend(list(kg.find_similar_nodes(unmatching_entity, k=num_neighbor).keys()))

    node_list = list(set(perturbation_ner_node_index + context_ner_node_index + unmatching_entity_node_index))
    # subgraph extraction or node information extraction
    if 'subgraph' in tool:
        subgraph = kg.get_subgraph_from_nodes(node_list)
        extracted_graph_info['node_list'] = [str(node) for node in node_list]
        extracted_graph_info['subgraph'] = subgraph
        kg_info = kg.subgraph_to_text(subgraph)
    else:
        extracted_graph_info['node_list'] = [str(node) for node in node_list]
        kg_info = ""
        for similar_node in node_list:
            if is_with_rel:
                doc_info = kg.doc_info_with_rel[similar_node]
            else:
                doc_info = kg.doc_info_without_rel[similar_node]
            kg_info += doc_info + '\n\n'

    return kg_info, extracted_graph_info


def get_ner_node_index(entity_dict, kg):
    """
    Faster version using precomputed caches and O(1) lookups.

    Returns:
    - node_list: list of matching node indices in starkPrimeKG
    - unmatching_entities: list of entities that are not matched
    """
    _load_mondo()

    index_list = []
    unmatching_entities = []
    drug_names_dict = kg._drugbank_id_to_name

    for entity, tag in entity_dict.items():
        entity_lower = entity.lower()
        cache_key = (tag, entity_lower)

        # Return memoized decision if present
        if cache_key in kg._ner_entity_cache:
            cached_idx = kg._ner_entity_cache[cache_key]
            if cached_idx is not None:
                index_list.append({'entity': entity, 'node': int(cached_idx)})
            else:
                unmatching_entities.append(entity)
                warnings.warn(f"Could not find the node index for {tag} {entity} in the KG.", stacklevel=2)
            continue

        node_index = kg._kg_node_name_lower.get(entity_lower)
        if node_index is None:
            node_index = kg.kg_node_name_dict.get(entity)

        if node_index is None:
            if tag == 'Disease':
                # Exact label
                if entity in mondo_node_label_dict:
                    mondo_node_index = mondo_node_label_dict[entity]
                    node_name = mondo_label_node_dict[mondo_node_index]
                    node_index = kg.kg_node_name_dict.get(node_name)
                else:
                    # Synonym to label
                    lbl = mondo_synonym_to_label.get(entity)
                    if lbl is not None:
                        mondo_node_index = mondo_node_label_dict.get(lbl)
                        if mondo_node_index is not None:
                            node_name = mondo_label_node_dict[mondo_node_index]
                            node_index = kg.kg_node_name_dict.get(node_name)

            elif tag == 'Chemical':
                # 0) Try PubChem cache
                cached = kg._pubchem_cache.get(entity_lower)
                dbid = None
                if cached:
                    node_index = cached.get('node_index')
                    dbid = cached.get('dbid')

                # 1) Fallback: DrugBank searcher
                if not dbid:
                    searcher = kg.get_drugbank_searcher()
                    if searcher is not None:
                        dbid = searcher.search_by_name(entity)
                # 2) Fallback: Try PubChem
                if not dbid:
                    compound = pubchem_client.get_compound_info(name=entity)
                    dbid = next((s for s in compound.get('synonyms', []) if s.startswith('DB')), None)
                # Set node index if dbid is found
                if dbid:
                    drug_name = drug_names_dict.get(dbid)
                    node_index = kg.kg_node_name_dict.get(drug_name)
                    if node_index:
                        kg._pubchem_cache[entity_lower] = {'node_index': node_index, 'dbid': dbid}

            elif tag == 'Gene':
                node_index = kg._alias_to_node_index.get(entity)

        if node_index is not None:
            node_index = int(node_index)
            index_list.append({'entity': entity, 'node': node_index})
            kg._ner_entity_cache[cache_key] = node_index
        else:
            unmatching_entities.append(entity)
            warnings.warn(f"Could not find the node index for {tag} {entity} in the KG.", stacklevel=2)
            kg._ner_entity_cache[cache_key] = None

    return index_list, unmatching_entities