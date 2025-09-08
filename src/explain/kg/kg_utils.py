import json
import time
import pubchempy as pcp
import warnings
import requests
import os

from explain.util import extract_entity_from_text

DATA_DIR = '/mnt/ps/home/CORP/yunhui.jang/research/hooke-explain/data'
with open(os.path.join(DATA_DIR, 'mondo.json'), 'r') as f:
    mondo_data = json.load(f)
mondo_nodes = mondo_data['graphs'][0]['nodes']
mondo_node_label_dict = {node['lbl']: idx for idx, node in enumerate(mondo_nodes) if 'lbl' in node.keys()}
mondo_label_node_dict = {v: k for k, v in mondo_node_label_dict.items()}
mondo_node_synonyms = {node['lbl']: node['meta']['synonyms'] for node in mondo_nodes if 'meta' in node.keys() and 'synonyms' in node['meta'].keys()}

# Fast MONDO synonym lookup: synonym value -> label
mondo_synonym_to_label = {}
for lbl, syns in mondo_node_synonyms.items():
    try:
        for syn in syns:
            v = syn.get('val')
            if v:
                mondo_synonym_to_label[v] = lbl
    except Exception:
        continue

def entity_matching_info(perturbation_index):
    """
    Get the corresponding KG node information from the perturbation index. (Exact entity matching)
    """
    index_tag = '_'.join(perturbation_index.split('_')[:-1])
    if len(index_tag) > 0:
        index_tag = '_' + index_tag
    perturbation_kg_mapping = json.load(open(f'data/starkprimeKG/perturbation_kg_mapping{index_tag}.json', 'r'))
    perturbation_index = str(perturbation_index)
    kg_infos = perturbation_kg_mapping[perturbation_index]['kg_info']
    org_infos = perturbation_kg_mapping[perturbation_index]['org_data']

    return kg_infos, org_infos

def get_kg_entity_doc(perturbation_index, kg, with_rel=False):
    kg_infos, org_infos = entity_matching_info(perturbation_index)
    doc_parts = []
    
    for kg_info, org_info in zip(kg_infos, org_infos):
        kg_node_index = int(kg_info['node_index'])
        if with_rel:
            doc_info = kg.doc_info_with_rel[kg_node_index]
        else:
            doc_info = kg.doc_info_without_rel[kg_node_index]

        doc_parts.append(f"This is the document information of {org_info['type']} {org_info['value']}.\n")
        doc_parts.append(doc_info)
        doc_parts.append("\n")

    return ''.join(doc_parts)

def get_kg_info(index, perturbation, tool, kg, is_with_rel, num_neighbor):
     # NER
    perturbation_text = json.dumps(perturbation, indent=4)
    if 'tahoe' in index or 'rxrx' in index:
        perturbation_partial_text = json.dumps(perturbation['perturbation']['perturbations'], indent=4)
    else:
        perturbation_partial_text = json.dumps(perturbation['perturbation']['perturbation'], indent=4)
    context_text = json.dumps(perturbation['perturbation']['context'], indent=4)
    perturbation_entity = extract_entity_from_text(perturbation_partial_text, index)
    
    context_entity = extract_entity_from_text(context_text, index)
    
    extracted_graph_info = {}
    # Phase 1: node selection 
    if 'entity-embedding' in tool:
        # KG-entity-embedding: prioritize the nodes that could be acquires (others: embedding-based)
        kg_node_perturbation, _ = entity_matching_info(index)
        if len(kg_node_perturbation) > 0:
            perturbation_similar_node_index = [int(kg_node['node_index']) for kg_node in kg_node_perturbation]
        else:
            perturbation_similar_node_index = list(kg.find_similar_nodes(perturbation_partial_text, k=num_neighbor).keys())
        context_similar_node_index = list(kg.find_similar_nodes(context_text, k=num_neighbor).keys())
        node_list = list(set(perturbation_similar_node_index + context_similar_node_index))
    elif 'entity' in tool:
        # KG-entity: get the excat matching entity information from the knowledge graph
        kg_info = get_kg_entity_doc(index, kg, is_with_rel)
    elif 'embedding' in tool:
        if 'subgraph' in tool:
            perturbation_similar_node_index = list(kg.find_similar_nodes(perturbation_partial_text, k=num_neighbor).keys())
            context_similar_node_index = list(kg.find_similar_nodes(context_text, k=num_neighbor).keys())
            node_list = list(set(perturbation_similar_node_index + context_similar_node_index))
        else:
            similar_node_dict = kg.find_similar_nodes(perturbation_text, k=num_neighbor)
            node_list = list(similar_node_dict.keys())
    elif 'ner' in tool:
        perturbation_ner_node_index, perturbation_unmatching_entities = get_ner_node_index(perturbation_entity, kg)
        perturbation_ner_node_index = [n['node'] for n in perturbation_ner_node_index]

        context_ner_node_index, context_unmatching_entities = get_ner_node_index(context_entity, kg)
        context_ner_node_index = [n['node'] for n in context_ner_node_index]
        unmatching_entity_node_index = []
        for unmatching_entity in perturbation_unmatching_entities+context_unmatching_entities:
            unmatching_entity_node_index.extend(list(kg.find_similar_nodes(unmatching_entity, k=num_neighbor).keys()))

        node_list = list(set(perturbation_ner_node_index + context_ner_node_index + unmatching_entity_node_index))
    # Phase 2: subgraph extraction or node information extraction
    if 'kg-entity' == tool or 'kg-entity-rephrase' == tool:
        # Get the single node information (exact matching nodes)
        pass
    elif 'subgraph' in tool:
        # KG-embedding-subgraph, KG-entity-embedding-subgraph: get the subgraph from the set of nodes
        subgraph = kg.get_subgraph_from_nodes(node_list)
        extracted_graph_info['node_list'] = [str(node) for node in node_list]
        extracted_graph_info['subgraph'] = subgraph
        kg_info = kg.subgraph_to_text(subgraph)
    else:
        # KG-embedding, KG-embedding-rephrase (get the top K similar nodes)
        extracted_graph_info['node_list'] = [str(node) for node in node_list]
        kg_info = ""
        for similar_node in node_list:
            if is_with_rel:
                doc_info = kg.doc_info_with_rel[similar_node]
            else:
                doc_info = kg.doc_info_without_rel[similar_node]
            kg_info += doc_info + '\n\n'

    return kg_info, extracted_graph_info

def get_dbid_from_pubchemid(cid):
    """
    Turn the pubchem CID into drugbank ID
    """
    rv = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    )
    j = rv.json()
    # Walk the PUG-View tree and collect DrugBank IDs
    def walk(section):
        hits = []
        if section.get("TOCHeading") == "DrugBank ID":
            for info in section.get("Information", []):
                for s in info.get("Value", {}).get("StringWithMarkup", []):
                    hits.append(s["String"])
        for child in section.get("Section", []) or []:
            hits.extend(walk(child))
        return hits

    drugbank_ids = []
    for rec in j.get("Record", {}).get("Section", []):
        drugbank_ids.extend(walk(rec))
    drugbank_ids = sorted(set(drugbank_ids))
    if len(drugbank_ids) == 0:
        return None
    return drugbank_ids[0]

def get_ner_node_index(entity_dict, kg):
    """
    Faster version using precomputed caches and O(1) lookups.

    Returns:
    - node_list: list of matching node indices in starkPrimeKG
    - unmatching_entities: list of entities that are not matched
    """

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
                warnings.warn(f"Could not find the node index for {tag} {entity} in the KG.")
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
                if node_index is None:
                    dbid = None
                    searcher = kg.get_drugbank_searcher()
                    if searcher is not None:
                        dbid = searcher.search_by_name(entity)
                    if dbid is None:
                        cached = kg._pubchem_cache.get(entity_lower)
                        if cached is not None:
                            node_index = cached.get('node_index')
                            dbid = cached.get('dbid')
                        if node_index is None and dbid is None:
                            try:
                                compound = pcp.get_compounds(entity, 'name')
                                if len(compound) > 0:
                                    compound = compound[0]
                                    for syn in compound.synonyms:
                                        if syn in drug_names_dict.values():
                                            node_index = kg.kg_node_name_dict.get(syn)
                                            break
                                    if node_index is None:
                                        cid = compound.cid
                                        dbid = get_dbid_from_pubchemid(cid)
                            except Exception:
                                pass
                            kg._pubchem_cache[entity_lower] = {'node_index': node_index, 'dbid': dbid}
                    if dbid is not None and node_index is None:
                        drug_name = drug_names_dict.get(dbid)
                        node_index = kg.kg_node_name_dict.get(drug_name)

            elif tag == 'Gene':
                node_index = kg._alias_to_node_index.get(entity)

        if node_index is not None:
            node_index = int(node_index)
            index_list.append({'entity': entity, 'node': node_index})
            kg._ner_entity_cache[cache_key] = node_index
        else:
            unmatching_entities.append(entity)
            warnings.warn(f"Could not find the node index for {tag} {entity} in the KG.")
            kg._ner_entity_cache[cache_key] = None

    return index_list, unmatching_entities