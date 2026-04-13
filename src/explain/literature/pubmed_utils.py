import json
import sys
from operator import itemgetter

import requests

from explain.util import extract_entity_from_text


def get_pubmed_info(index, perturbation, question, num_papers, mode='ner', paper_info_column=['title', 'abstract']):
    '''
    Current best version: pubmed-fast-ner
    fast: use fastrag server (if not, use AbstractRetriever)
    ner: use ner entities (if not, use question text)
    '''
    if 'ner' in mode:
        perturbation_text = json.dumps(perturbation, indent=4)
        entities = extract_entity_from_text(perturbation_text, index)
        query = ', '.join(entities.keys())
    else:
        query = question

    url = "http://localhost:8003/find_matches"
    payload = {"query": query, "k": num_papers+10}
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] request failed: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        data = r.json()
    except Exception:
        print(r.text)
        sys.exit(0)

    documents = [{'abstract': doc['abstract'], 'pmid': doc['pmid'], 'title': doc['title'], 'authors': doc['authors'], 'publication_year': doc['publication_year']} for doc in data]

    papers_info = []
    for abstract in documents:
        if abstract['abstract'] is None or abstract['abstract'] == 'Abstract not found':
            continue
        paper_info = {}
        paper_info['pmid'] = abstract['pmid']
        paper_info['title'] = abstract['title']
        paper_info['authors'] = abstract['authors']
        paper_info['abstract'] = abstract['abstract']
        paper_info['publication_year'] = abstract['publication_year']
        papers_info.append(paper_info)
        if len(papers_info) >= num_papers:
            break

    get_ac = itemgetter(*paper_info_column)
    papers_info = [dict(zip(paper_info_column, get_ac(d), strict=False)) for d in papers_info if all(k in d for k in paper_info_column)]


    return papers_info
