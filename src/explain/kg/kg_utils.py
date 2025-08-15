import json
from explain.kg.kg import KnowledgeGraph
from explain.util import extract_entity_from_text

with open('data/mondo.json', 'r') as f:
    mondo_data = json.load(f)
mondo_nodes = mondo_data['graphs'][0]['nodes']
mondo_node_label_dict = {node['lbl']: idx for idx, node in enumerate(mondo_nodes) if 'lbl' in node.keys()}
mondo_label_node_dict = {v: k for k, v in mondo_node_label_dict.items()}
mondo_node_synonyms = {node['lbl']: node['meta']['synonyms'] for node in mondo_nodes if 'meta' in node.keys() and 'synonyms' in node['meta'].keys()}


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
    doc_info_total = ""
    for kg_info, org_info in zip(kg_infos, org_infos):
        kg_node_index = int(kg_info['node_index'])
        if with_rel:
            doc_info = kg.doc_info_with_rel[kg_node_index]
        else:
            doc_info = kg.doc_info_without_rel[kg_node_index]

        doc_info_total += f"This is the document information of {org_info['type']} {org_info['value']}.\n"
        doc_info_total += doc_info
        doc_info_total += "\n"

    return doc_info_total

def get_kg_info(index, perturbation, tool, kg, is_with_rel, num_neighbor):
    perturbation_text = json.dumps(perturbation, indent=4)
    perturbation_partial_text = json.dumps(perturbation['perturbation'], indent=4)
    perturbation_entity = extract_entity_from_text(perturbation_partial_text)
    context_text = json.dumps(perturbation['context'], indent=4)
    context_entity = extract_entity_from_text(context_text)
    
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
        perturbation_ner_node_index = get_ner_node_index(perturbation_entity, kg)
        perturbation_node_index = list(set(perturbation_exact_node_index + perturbation_ner_node_index))

        
        context_ner_node_index = get_ner_node_index(context_entity, kg)

        # perturbation_ner_node_index = [ for pe in perturbation_entity]
        if len(context_ner_node_index) == 0:
            context_ner_node_index = list(kg.find_similar_nodes(context_text, k=num_neighbor).keys())
        else:
            pass
        
        node_list = list(set(perturbation_node_index + context_ner_node_index))

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

def get_ner_node_index(entity_dict, kg):
    """
    Get the KG node index from the entity dictionary.
    If exact matching entity is not found, search for synonyms in the KG.
    """
    index_list = []
    gene_protein_nodes = [node for node in kg.node_info.values() if node['type'] == 'gene/protein']
    gene_protein_name_alias_dict = {node['name']: node['details']['alias'] for node in gene_protein_nodes if 'alias' in node['details'].keys()}
    gene_protein_name_alias_dict = {k: v if type(v) is list else [v] for k, v in gene_protein_name_alias_dict.items()}
    
    for entity, tag in entity_dict.items():
        node_index = None
        # if entity.lower() in kg.kg_node_name_dict.keys() or entity in kg.kg_node_name_dict.keys():
        if entity.lower() in kg.kg_node_name_dict.keys():
            node_index = kg.kg_node_name_dict[entity.lower()]
        elif entity in kg.kg_node_name_dict.keys():
            node_index = kg.kg_node_name_dict[entity]
        
        else:
            # TODO
            if tag == 'Disease':
                # map to MONDO name (if not in MONDO, map to disease name)
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
                # map to Drugbank id 
                pass
                node_index = 0
            elif tag == 'Gene':
                # search for alias in nodes with gene/protein node type
                for node_name, aliases in gene_protein_name_alias_dict.items():
                    if entity in aliases:
                        node_index = kg.kg_node_name_dict[node_name]
                        break
        if node_index is not None:
            index_list.append(int(node_index))
    return index_list