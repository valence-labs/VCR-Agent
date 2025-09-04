import json
from explain.util import extract_entity_from_text
from langchain_community.retrievers import WikipediaRetriever

def get_wikipedia_info(index, perturbation):
    retriever = WikipediaRetriever()
    perturbation_text = json.dumps(perturbation, indent=4)

    entities = extract_entity_from_text(perturbation_text, index)
    result = ""
    for entity in entities.keys():
        docs = retriever.invoke(entity)
        if len(docs) > 0:
            result += f"## {entity}\n"
            result += docs[0].page_content
            result += "\n\n"
        
    return result