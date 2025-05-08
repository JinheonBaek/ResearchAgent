import argparse
from tqdm import tqdm

from utils import data_io, s2
from knowledge.store import KnowledgeStore


def fetch_resources(paper: dict, knowledge_store: KnowledgeStore):
    references = s2.get_relevant_references(paper)
    entities = knowledge_store.get_relevant_entities(
        [paper['corpusId']] + [reference['corpusId'] for reference in references]
    )
    return references, entities


def run(paper_ids: list, knowledge_store: KnowledgeStore):
    papers = s2.filter_papers(
        s2.get_papers(paper_ids),
        categories=['title', 'abstract', 'embedding']
    )

    results = None

    for paper in tqdm(papers):
        references, entities = fetch_resources(paper, knowledge_store)

    return results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    paper_ids = data_io.load_paper_ids('./data/papers.jsonl')
    knowledge_store = KnowledgeStore('./data/knowledge.jsonl')
    
    results = run(paper_ids, knowledge_store)
