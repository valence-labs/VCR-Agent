import json
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

global perturbation_ner_mapping
perturbation_ner_mapping = json.load(open('data/perturbation_ner_mapping.json', 'r'))

def entity_matching_info(perturbation_index):
    """
    Get the corresponding KG node information from the perturbation index. (Exact entity matching)
    """
    perturbation_kg_mapping = json.load(open('data/starkprimeKG/perturbation_kg_mapping.json', 'r'))
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
    if index in [perturbation_ner_dict['index'] for perturbation_ner_dict in perturbation_ner_mapping]:
        perturbation_entity = perturbation_ner_mapping[index]['perturbation_entity']
        context_entity = perturbation_ner_mapping[index]['context_entity']
    else:
        perturbation_text = json.dumps(perturbation, indent=4)
        perturbation_partial_text = json.dumps(perturbation['perturbation'], indent=4)
        perturbation_entity = extract_entity_from_text(perturbation_partial_text, index)
        context_text = json.dumps(perturbation['context'], indent=4)
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
        kg_node_perturbation, _ = entity_matching_info(index)
        perturbation_exact_node_index = []
        if len(kg_node_perturbation) > 0:
            perturbation_exact_node_index = [int(kg_node['node_index']) for kg_node in kg_node_perturbation]
        perturbation_ner_node_index, perturbation_unmatching_entities = get_ner_node_index(perturbation_entity, kg)
        perturbation_ner_node_index = [n['node'] for n in perturbation_ner_node_index]
        perturbation_node_index = list(set(perturbation_exact_node_index + perturbation_ner_node_index))

        context_ner_node_index, context_unmatching_entities = get_ner_node_index(context_entity, kg)
        context_ner_node_index = [n['node'] for n in context_ner_node_index]
        unmatching_entity_node_index = []
        for unmatching_entity in perturbation_unmatching_entities+context_unmatching_entities:
            unmatching_entity_node_index.extend(list(kg.find_similar_nodes(unmatching_entity, k=num_neighbor).keys()))

        node_list = list(set(perturbation_node_index + context_ner_node_index + unmatching_entity_node_index))
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
    Get the KG node index from the entity dictionary.
    If exact matching entity is not found, search for synonyms in the KG.

    Returns:
    - node_list: list of matching node indices in starkPrimeKG
    - unmatching_entities: list of entities that are not matched to any node in starkPrimeKG
    """
    index_list = []
    unmatching_entities = []
    gene_protein_nodes = [node for node in kg.node_info.values() if node['type'] == 'gene/protein']
    gene_protein_name_alias_dict = {node['name']: node['details']['alias'] for node in gene_protein_nodes if 'alias' in node['details'].keys()}
    gene_protein_name_alias_dict = {k: v if type(v) is list else [v] for k, v in gene_protein_name_alias_dict.items()}

    drug_nodes = [node for node in kg.node_info.values() if node['type']=='drug']
    # Drug name to DrugBank ID
    drug_names_dict = {node['id']: node['name'] for node in drug_nodes}
    for entity, tag in entity_dict.items():
        node_index = None
        # Exact matching
        if entity.lower() in kg.kg_node_name_dict.keys():
            node_index = kg.kg_node_name_dict[entity.lower()]
        elif entity in kg.kg_node_name_dict.keys():
            node_index = kg.kg_node_name_dict[entity]
        
        # Synonym matching
        else:
            if tag == 'Disease':
                # map to MONDO name (if not in MONDO, map to disease name)
                mondo_node_index = None
                if entity in mondo_node_label_dict.values():
                    mondo_node_index = mondo_node_label_dict[entity]
                else:
                    for mondo_lbl, mondo_synonyms in mondo_node_synonyms.items():
                        mondo_synonyms_value = [synonym['val'] for synonym in mondo_synonyms]
                        if entity in mondo_synonyms_value:
                            mondo_node_index = mondo_node_label_dict[mondo_lbl]
                            break
                    if mondo_node_index is not None:
                        node_name = mondo_label_node_dict[mondo_node_index]
                        node_index = kg.kg_node_name_dict[node_name]

            elif tag == 'Chemical':
                # search for pubchem synonyms in the KG

                # Match Drug Bank ID with pubchemid (can be replaced with DrugBank access)
                if node_index is None:
                    # Search for DrugBank ID with drug name (only if drugbank_searcher is available)
                    dbid = None
                    drugbank_searcher = kg.get_drugbank_searcher()
                    if drugbank_searcher is not None:
                        dbid = drugbank_searcher.search_by_name(entity)
                    # Match Drug Bank ID with pubchemid
                    if dbid is None:
                        try:
                            compound = pcp.get_compounds(entity, 'name')
                            if len(compound) > 0:
                                compound = compound[0]
                                for syn in compound.synonyms:
                                    if syn in drug_names_dict.values():
                                        node_index = kg.kg_node_name_dict[syn]
                                        break
                                if node_index is None:
                                    cid = compound.cid
                                    dbid = get_dbid_from_pubchemid(cid)
                        except:
                            pass
                        
                    if dbid is not None:
                        drug_name = drug_names_dict.get(dbid, None)
                        node_index = kg.kg_node_name_dict.get(drug_name, None)

            elif tag == 'Gene':
                # search for alias in nodes with gene/protein node type
                for node_name, aliases in gene_protein_name_alias_dict.items():
                    if entity in aliases:
                        node_index = kg.kg_node_name_dict[node_name]
                        break
        if node_index is not None:
            index_list.append({'entity': entity, 'node': int(node_index)})
        else:
            unmatching_entities.append(entity)
            warnings.warn(f"Could not find the node index for {tag} {entity} in the KG.")
    return index_list, unmatching_entities