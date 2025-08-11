import json
from explain.kg.kg import KnowledgeGraph

def get_kg_info(perturbation_index):
    perturbation_kg_mapping = json.load(open('data/starkprimeKG/perturbation_kg_mapping.json', 'r'))
    perturbation_index = str(perturbation_index)
    kg_infos = perturbation_kg_mapping[perturbation_index]['kg_info']
    org_infos = perturbation_kg_mapping[perturbation_index]['org_data']

    return kg_infos, org_infos

def get_kg_entity_doc(perturbation_index, with_rel=False):
    kg_infos, org_infos = get_kg_info(perturbation_index)
    kg = KnowledgeGraph()
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