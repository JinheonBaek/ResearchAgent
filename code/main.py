import argparse
from tqdm import tqdm

from utils import data_io, s2
from knowledge.store import KnowledgeStore
from models.openai import OpenAIClient
from pipelines.research_pipeline import ResearchPipeline


def fetch_resources(paper: dict, knowledge_store: KnowledgeStore):
    references = s2.get_relevant_references(paper)
    entities = knowledge_store.get_relevant_entities(
        [paper['corpusId']] + [reference['corpusId'] for reference in references]
    )
    return references, entities


def run(
    paper_ids: list,
    knowledge_store: KnowledgeStore,
    openai_client: OpenAIClient
):
    papers = s2.filter_papers(
        s2.get_papers(paper_ids),
        categories=['title', 'abstract', 'embedding']
    )

    results = []

    for paper in tqdm(papers):
        context = {'paper': {key: paper.get(key) for key in ('title', 'abstract')}}
        
        references, entities = fetch_resources(paper, knowledge_store)
        context.update(references=references, entities=entities)

        research_pipeline = ResearchPipeline(api_client=openai_client)
        context = research_pipeline.run(context)

        results.append(context)
        data_io.save_result('./results/ideas.jsonl', context)

    return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-path', '-d', default='./data/papers.jsonl')
    argparser.add_argument('--knowledge-path', '-k', default='./data/knowledge.jsonl')
    argparser.add_argument('--model-name', '-m', default='gpt-4o')
    args = argparser.parse_args()

    paper_ids = data_io.load_paper_ids(args.data_path)
    knowledge_store = KnowledgeStore(args.knowledge_path)

    openai_client = OpenAIClient(model=args.model_name)

    results = run(paper_ids, knowledge_store, openai_client)
