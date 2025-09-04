from Bio import Entrez
from paperqa import ask, Settings
import os
import json
from operator import itemgetter

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

    
def get_paperqa_info(index, perturbation, question, num_papers=10, mode='paperqa', paper_info_column=['title', 'abstract']):
    mode = mode.replace('-schema', '').replace('-rephrase', '')
    perturbation_text = json.dumps(perturbation, indent=4)
    entities = extract_entity_from_text(perturbation_text, index)

    entity_list = list(entities.keys())
    entity_text = ', '.join(entity_list)
    file_path = f'data/papers/{index}_{'_'.join(entity_list)}'
    
    # Check if papers_info.json already exists to avoid re-processing
    papers_info_path = f'{file_path}/papers_info.json'
    if os.path.exists(papers_info_path):
        with open(papers_info_path, 'r') as f:
            papers_info = json.load(f)
        print(f"File {file_path} already exists")
    else:
        if not os.path.exists(file_path):
            studies = search(entity_text, num_papers)
            studiesIdList = studies['IdList']
            while(len(studiesIdList) < num_papers and len(entity_text) > 0):
                entity_text = entity_text[:entity_text.rfind(',')]
                studies = search(entity_text, num_papers)
                studiesIdList += studies['IdList']
            papers_info = id_list_to_txt(studiesIdList, file_path)
        else:
            # Load from existing txt files if directory exists but no papers_info.json
            papers_info = load_papers_from_txt_files(file_path)
            with open(papers_info_path, 'w') as f:
                json.dump(papers_info, f, indent=4)
    papers_info = papers_info[:num_papers]
    get_ac = itemgetter(*paper_info_column)

    papers_info = [dict(zip(paper_info_column, get_ac(d))) for d in papers_info if all(k in d for k in paper_info_column)]
    # return answer for the question with paperqa
    if mode in ['paperqa', 'paperqa_only']:

        answer_response = ask(
            question,
            settings=Settings(temperature=0.5, paper_directory=file_path)
        )
        
        answer = answer_response.session.answer 
    # return only list of papers
    elif mode in ['paperqa_list', 'paperqa_list_rephrase']:
        answer = ""

    return answer, papers_info


def load_papers_from_txt_files(file_path):
    """
    Load all txt files from file_path and extract paper information.
    
    Args:
        file_path (str): Path to directory containing txt files
        
    Returns:
        list: List of dictionaries containing paper information
    """
    papers_info = []
    
    if not os.path.exists(file_path):
        print(f"Directory {file_path} does not exist")
        return papers_info
    
    txt_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        file_path_full = os.path.join(file_path, txt_file)
        try:
            with open(file_path_full, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the content to extract paper information
            lines = content.split('\n')
            paper_info = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Title: '):
                    paper_info['title'] = line[7:]  # Remove 'Title: ' prefix
                elif line.startswith('Authors: '):
                    paper_info['authors'] = line[9:]  # Remove 'Authors: ' prefix
                elif line.startswith('Venue: '):
                    paper_info['journal'] = line[7:]  # Remove 'Venue: ' prefix
                elif line.startswith('Year: '):
                    paper_info['year'] = line[6:]  # Remove 'Year: ' prefix
                elif line == 'Abstract':
                    # Find the abstract content (everything after 'Abstract' line)
                    abstract_start = content.find('Abstract\n')
                    if abstract_start != -1:
                        abstract_content = content[abstract_start + 9:].strip()  # Remove 'Abstract\n' prefix
                        paper_info['abstract'] = abstract_content
                    break
            
            # Only add if we have at least title and abstract
            if 'title' in paper_info and 'abstract' in paper_info:
                papers_info.append(paper_info)
                
        except Exception as e:
            print(f"Error reading file {txt_file}: {e}")
            continue
    
    return papers_info


def batch_get_paperqa_info(perturbations, questions, num_papers=10, mode='paperqa_list'):
    """
    Batch process multiple perturbations to get paper information efficiently.
    
    Args:
        perturbations (list): List of perturbation dictionaries
        questions (list): List of questions corresponding to perturbations
        num_papers (int): Number of papers to retrieve per perturbation
        mode (str): Mode for paperqa processing
        
    Returns:
        list: List of (answer, papers_info) tuples
    """
    results = []
    
    for i, (perturbation, question) in enumerate(zip(perturbations, questions)):
        try:
            answer, papers_info = get_paperqa_info(i, perturbation, question, num_papers, mode)
            results.append((answer, papers_info))
        except Exception as e:
            print(f"Error processing perturbation {i}: {e}")
            results.append(("", []))
    
    return results

