import json
from explain.util import extract_entity_from_text
from langchain_community.retrievers import WikipediaRetriever
from concurrent.futures import ThreadPoolExecutor


def get_wikipedia_info(index, perturbation, max_workers=8):
    retriever = WikipediaRetriever()
    perturbation_text = json.dumps(perturbation, indent=4)

    entities = list(set(extract_entity_from_text(perturbation_text, index).keys()))

    # multi-threading (for faster retrieval)
    def fetch(entity):
        try:
            docs = retriever.invoke(entity)
            if docs:
                return f"## {entity}\n{docs[0].page_content}\n\n"
        except Exception:
            return None
        return None

    parts = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for content in pool.map(fetch, entities):
            if content:
                parts.append(content)

    return "".join(parts)