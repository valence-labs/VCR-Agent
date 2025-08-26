import json
from urllib.request import urlopen
from textwrap import dedent


from explain.literature.harmonizome import Harmonizome, Entity
from explain.util import extract_entity_from_text

def harmonizome_gene_to_doc(gene_info):
    """
    gene_info:
    """
    if gene_info == "":
        return ""
    text = dedent(f"""
    The gene name is {gene_info['name']} and the gene symbol is {gene_info['symbol']}.
    {gene_info['description']}
    The gene synonyms are {', '.join(gene_info['synonyms'])}.
    The gene is related to the proteins {', '.join([protein['symbol'] for protein in gene_info['proteins']])}.\n
    """)

    return text

def harmonizome_gene_set_to_doc(gene_set_info):
    """
    gene_set_name:
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
    text = f"""
    The related genes are {', '.join(gene_set)}.\n
    The gene information is as follows:
    """
    # Add gene information for each gene
    for gene in gene_set:
        gene_info = Harmonizome.get(Entity.GENE, name=gene)
        gene_text = harmonizome_gene_to_doc(gene_info)
        text += gene_text

    return dedent(text)


def get_harmonizome_info(perturbation, is_ner=False):

    result = ""
    if is_ner:
        perturbation_text = json.dumps(perturbation, indent=4)
        entities = extract_entity_from_text(perturbation_text)
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
                except:
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
        # for target
        target = perturbation['perturbation']['target']
        try:
            gene_info = Harmonizome.get(Entity.GENE, name=target)
        except:
            gene_info = {'status': 'error'}
        if 'status' in gene_info.keys():
            target_doc = ""
        else:
            target_doc = "### TARGET GENE INFORMATION\nThe target gene of the perturbation is " + target + ". "
            target_doc += harmonizome_gene_to_doc(gene_info)

        # for preturbation disease name (get related genes)
        name = perturbation['perturbation']['name']
        try:
            gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=name)
        except:
            gene_set_info = {'status': 'error'}
        if 'status' not in gene_set_info.keys():
            gene_set_doc = "### PERTURBATION NAME INFORMATION\nThe perturbation name is " + name + ". "
            gene_set_doc += harmonizome_gene_set_to_doc(gene_set_info)
        else:
            gene_set_doc = ""

        # TODO: cell context mapping after NER for disease model (current version: search the matching word)
        perturbation_disease_model = perturbation['context']['disease_model']
        for word in perturbation_disease_model.split():
            gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=word)
            if 'status' not in gene_set_info.keys():
                break
        disease_model_doc = "### DISEASE MODEL INFORMATION\nThe disease model of the context is " + perturbation_disease_model + ". "
        disease_model_doc += harmonizome_gene_set_to_doc(gene_set_info)

        result = target_doc + gene_set_doc + disease_model_doc


    return result


