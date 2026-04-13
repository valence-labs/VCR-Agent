import json
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
from urllib.request import urlopen

from explain.literature.harmonizome import Entity, Harmonizome
from explain.util import extract_entity_from_text


def harmonizome_gene_to_doc(gene_info):
    """
    Turn the harmonizome gene information into a text format.
    """
    if gene_info == "" or 'name' not in gene_info.keys():
        return ""
    text = dedent(f"""
    The gene name is {gene_info['name']} and the gene symbol is {gene_info['symbol']}.
    {gene_info['description']}
    The gene synonyms are {', '.join(gene_info['synonyms'])}.
    The gene is related to the proteins {', '.join([protein['symbol'] for protein in gene_info['proteins']])}.\n
    """)

    return text


def map_gene_set_to_related_genes(gene_set_info):
    """
    Turn the harmonizome gene set information into a list of related genes.
    """

    gene_set = gene_set_info['geneSets'][0]
    api_url = 'https://maayanlab.cloud/Harmonizome'
    full_url = api_url + gene_set['href']

    response = urlopen(full_url)
    data = response.read().decode('utf-8')
    gene_set_info = json.loads(data)

    # TODO: add rule-based filtering to the gene set (current version: only keep the first 20 genes)
    gene_set = [data['gene']['symbol'] for data in gene_set_info['associations']]
    gene_set = gene_set[:20]

    return gene_set

def harmonizome_gene_set_to_doc(gene_set_info, max_workers=8):


    gene_set = map_gene_set_to_related_genes(gene_set_info)
    genes = list(dict.fromkeys(gene_set))  # dedup, preserve order

    def fetch_gene_text(gene):
        try:
            gene_info = Harmonizome.get(Entity.GENE, name=gene)
            return harmonizome_gene_to_doc(gene_info)
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        gene_texts = list(pool.map(fetch_gene_text, genes))

    text = f"""
    The related genes are {', '.join(genes)}.\n
    The gene information is as follows:
    """
    text += "".join(gene_texts)
    return dedent(text)


def map_known_targets_to_genes(known_targets):
    """
    Map the known targets to the genes.
    """
    if isinstance(known_targets, str):
        genes = known_targets.split(',')
    elif isinstance(known_targets, list):
        genes = known_targets
    else:
        genes = []
    return genes

def get_harmonizome_info(index, perturbation, is_ner=False):
    '''
    Current best version: is_ner=False
    ner: use ner entities (if not, use rule-based mapping (target: gene / perturbation name: gene set / cell type: gene set / disease model: gene set))
    '''
    result = ""
    gene_info_list, gene_set_info_list = [], []
    # search from ner entities
    if is_ner:
        perturbation_text = json.dumps(perturbation, indent=4)
        entities = extract_entity_from_text(perturbation_text, index)
        for entity, tag in entities.items():
            if tag == 'Gene':
                gene_info = Harmonizome.get(Entity.GENE, name=entity)
                if 'status' in gene_info.keys():
                    continue
                result += "### GENE INFORMATION\nThe related gene of the perturbation is " + entity + ". "
                result += harmonizome_gene_to_doc(gene_info)
            elif tag in ['Disease', 'Chemical'] :
                try:
                    info = Harmonizome.get(Entity.ATTRIBUTE, name=entity)
                except Exception:
                    info = {'status': 'error'}
                if 'status' in info.keys():
                    continue
                if tag == 'Disease':
                    result += "### DISEASE INFORMATION\nThe disease of the perturbation is " + entity + ". "
                elif tag == 'Chemical':
                    result += "### CHEMICAL INFORMATION\nThe related drug of the perturbation is " + entity + ". "
                result += harmonizome_gene_set_to_doc(info)
        result += "\n\n"
    else:
        # without ner, search from perturbation text (target, name, disease model)
        # Detect data format: multi-perturbation list vs single perturbation dict
        is_multi_pert = 'perturbations' in perturbation['perturbation']
        # for target
        if is_multi_pert:
            chem_targets = [map_known_targets_to_genes(pert['known targets']) for pert in perturbation['perturbation']['perturbations'] if pert.get('perturbation type') == 'chemical' or pert.get('type') == 'chemical']
            chem_targets = [gene for sublist in chem_targets for gene in sublist]
            gene_targets = [pert.get('gene symbol', '') for pert in perturbation['perturbation']['perturbations'] if pert.get('perturbation type') == 'genetic' or pert.get('type') == 'genetic']
            targets = list(set(chem_targets + gene_targets))
            targets = [target for target in targets if len(target) > 0]
        else:
            targets = [perturbation['perturbation']['perturbation']['target']]
        target_doc = ""
        for target in targets:
            if len(target) == 0:
                continue
            try:
                gene_info = Harmonizome.get(Entity.GENE, name=target)
            except Exception:
                gene_info = {'status': 'error'}
            if 'status' not in gene_info.keys():
                target_doc += "### TARGET GENE INFORMATION\nThe target gene of the perturbation is " + target + ". "
                target_doc += harmonizome_gene_to_doc(gene_info)
                gene_info_list.append(gene_info)

        # for perturbation chemical name (get related genes)
        if is_multi_pert:
            names = [pert['name'] for pert in perturbation['perturbation']['perturbations'] if pert.get('perturbation type') == 'chemical' or pert.get('type') == 'chemical']
        else:
            names = [perturbation['perturbation']['perturbation']['name']]
        gene_set_doc = ""
        for name in names:
            if name is None:
                continue
            if len(name) == 0:
                continue
            try:
                gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=name)
            except Exception:
                gene_set_info = {'status': 'error'}
            if 'status' not in gene_set_info.keys():
                gene_set_doc += "### PERTURBATION NAME INFORMATION\nThe perturbation name is " + name + ". "
                gene_set_doc += harmonizome_gene_set_to_doc(gene_set_info)
                gene_set_info_list.append(gene_set_info)

        disease_model_doc = ""
        # Cell type / disease model context lookup
        context = perturbation['perturbation']['context']
        if isinstance(context, list):
            # Multi-perturbation format: context is a list
            cell_type = context[0].get('cell type') or context[0].get('cell_type')
            if cell_type is not None:
                try:
                    gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=cell_type)
                except Exception:
                    gene_set_info = {'status': 'error'}
                if 'status' not in gene_set_info.keys():
                    disease_model_doc += "### CELL TYPE INFORMATION\nThe cell type of the context is " + cell_type + ". "
                    disease_model_doc += harmonizome_gene_set_to_doc(gene_set_info)
                    gene_set_info_list.append(gene_set_info)

        else:
            perturbation_disease_model = context['disease_model']
            for word in perturbation_disease_model.split():
                if len(word) == 0:
                    continue
                gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=word)
                if 'status' not in gene_set_info.keys():
                    break
            disease_model_doc = "### DISEASE MODEL INFORMATION\nThe disease model of the context is " + perturbation_disease_model + ". "
            disease_model_doc += harmonizome_gene_set_to_doc(gene_set_info)
            gene_set_info_list.append(gene_set_info)

        result = target_doc + gene_set_doc + disease_model_doc

    final_gene_info = {'gene_info': gene_info_list, 'gene_set_info': gene_set_info_list}

    return result, final_gene_info


