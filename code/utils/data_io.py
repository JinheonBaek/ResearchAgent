import os
import json


def load_jsonl(file_path: str):
    return [json.loads(line) for line in open(file_path, 'r')]

def save_result(file_path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as outfile:
        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

def load_paper_ids(file_path: str = None, num_papers: int = 300):
    paper_ids = (
        ['215416146', '259095910', '259370631', '7720039', '12424239']
        if not file_path else
        [paper['corpusid'] for paper in load_jsonl(file_path)]
    )
    return [f'CorpusId:{id}' for id in paper_ids[:num_papers]]
