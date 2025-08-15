import json
from urllib.request import urlopen
from textwrap import dedent


from explain.literature.harmonizome import Harmonizome, Entity


def harmonizome_gene_to_doc(gene_info):
    """
    gene_info:
    """
    
    text = dedent(f"""
    {gene_info['description']}
    The gene name is {gene_info['name']} and the gene symbol is {gene_info['symbol']}.
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
    text = dedent(f"""
    The related genes are {', '.join(gene_set)}.\n
    """)

    return text


def get_harmonizome_info(perturbation):

    result = ""

    # for target
    target = perturbation['perturbation']['target']
    gene_info = Harmonizome.get(Entity.GENE, name=target)
    target_doc = "### TARGET GENE INFORMATION\nThe target gene of the perturbation is " + target + ". "
    target_doc += harmonizome_gene_to_doc(gene_info)

    # for preturbation disease name (get related genes)
    name = perturbation['perturbation']['name']
    gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=name)
    gene_set_doc = "### PERTURBATION NAME INFORMATION\nThe perturbation name is " + name + ". "
    gene_set_doc += harmonizome_gene_set_to_doc(gene_set_info)

    # TODO: cell context mapping after NER for disease model (current version: search the matching word)
    perturbation_disease_model = perturbation['context']['disease_model']
    for word in perturbation_disease_model.split():
        gene_set_info = Harmonizome.get(Entity.ATTRIBUTE, name=word)
        if 'status' not in gene_set_info.keys():
            break
    disease_model_doc = "### DISEASE MODEL INFORMATION\nThe disease model of the context is " + perturbation_disease_model + ". "
    disease_model_doc += harmonizome_gene_set_to_doc(gene_set_info)

    result += target_doc + gene_set_doc + disease_model_doc


    return result


