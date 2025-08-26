from Bio import Entrez
from paperqa import ask, Settings
from tqdm import tqdm
import os
import json

from explain.util import extract_entity_from_text

def search(query, num_papers=10):
    Entrez.email = 'yunhuijang@kaist.ac.kr'
    handle = Entrez.esearch(db='pubmed',
    sort='relevance',
    retmax=str(num_papers),
    retmode='xml',
    term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'yunhuijang@kaist.ac.kr'
    handle = Entrez.efetch(db='pubmed',
    retmode='xml',
    id=ids)
    results = Entrez.read(handle)
    return results

def id_list_to_txt(studiesIdList, file_path):
    chunk_size = 10000
    papers_info = []
    for chunk_i in range(0, len(studiesIdList), chunk_size):
        chunk = studiesIdList[chunk_i:chunk_i + chunk_size]
        papers = fetch_details(chunk)
        for i, paper in enumerate(papers['PubmedArticle']):


            title = paper['MedlineCitation']['Article']['ArticleTitle']
            try:
                abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            except:
                continue
            try:
                journal = paper['MedlineCitation']['Article']['Journal']['Title']
            except:
                journal = 'N/A'
            try:
                pubdate_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
            except:
                pubdate_year = 'N/A'
            try:
                authors = ''
                for author in paper['MedlineCitation']['Article']['AuthorList']:
                    authors += f"{author['ForeName']} {author['LastName']}, "
            except:
                authors = 'N/A'

            os.makedirs(file_path, exist_ok=True)

            with open(f'{file_path}/{i}.txt', 'w') as f:
                f.write(f'Title: {title}\n')
                f.write(f'Authors: {authors}\n')
                f.write(f'Venue: {journal}\n')
                f.write(f'Year: {pubdate_year}\n')
                f.write(f'Abstract\n {abstract}\n')
            papers_info.append({
                'title': title,
                'authors': authors,
                'journal': journal,
                'abstract': abstract
            })
        with open(f'{file_path}/papers_info.json', 'w') as f:
            json.dump(papers_info, f, indent=4)
        return papers_info

    
def get_paperqa_info(index, perturbation, question, num_papers=10, mode='paperqa'):
    ner_mapping_data = json.load(open("data/perturbation_ner_mapping.json"))
    ner_mapping_data_indices = [item['index'] for item in ner_mapping_data]
    perturbation_text = json.dumps(perturbation, indent=4)
    if index in ner_mapping_data_indices:
        entities = {**ner_mapping_data[index]['perturbation_entity'], **ner_mapping_data[index]['context_entity']}
    else:
        entities = extract_entity_from_text(perturbation_text)

    entity_list = list(entities.keys())
    entity_text = ', '.join(entity_list)
    file_path = f'data/papers/{index}_{'_'.join(entity_list)}'
    if not os.path.exists(file_path):
        studies = search(entity_text, num_papers)
        studiesIdList = studies['IdList']
        while(len(studiesIdList) < num_papers and len(entity_text) > 0):
            entity_text = entity_text[:entity_text.rfind(',')]
            studies = search(entity_text, num_papers)
            studiesIdList += studies['IdList']
        papers_info = id_list_to_txt(studiesIdList, file_path)
    else:
        with open(f'{file_path}/papers_info.json', 'r') as f:
            papers_info = json.load(f)
        print(f"File {file_path} already exists")

    # return answer for the question with paperqa
    if mode == 'paperqa':           
        answer_response = ask(
            question,
            settings=Settings(temperature=0.5, paper_directory=file_path)
        )
        
        answer = answer_response.session.answer 
    # return only list of papers
    elif mode == 'paperqa_list':
        answer = ""

    return answer, papers_info

