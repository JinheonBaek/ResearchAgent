import requests
from itertools import chain

import torch
from sentence_transformers.util import cos_sim


def get_request(url, headers, params, timeout=60, return_type=[]):
    try:    response = requests.get(url, headers=headers, params=params, timeout=timeout).json()
    except: response = return_type
    return response


def post_request(url, headers, params, json, timeout=60, return_type=[]):
    try:    response = requests.post(url, headers=headers, params=params, json=json, timeout=timeout).json()
    except: response = return_type
    return response


def batched(items: list, batch_size: int):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def flatten_list(items: list):
    return list(chain.from_iterable(items))


def get_papers(
    ids: list, 
    fields: list = ['corpusId', 'title', 'abstract', 'year', 'publicationDate', 'referenceCount', 'citationCount', 'embedding.specter_v2'], 
    batch_size: int = 100
):
    response = [
        post_request(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            headers={},
            params={'fields': ','.join(fields)},
            json={'ids': ids_batched}
        )
        for ids_batched in batched(ids, batch_size)
    ]
    return flatten_list(response)


def filter_papers(
    papers: list,
    categories: list = ['title', 'abstract', 'embedding']
):
    return [
        paper for paper in papers
        if all(paper[category] is not None for category in categories)
    ]


def get_relevant_references(
    paper: dict, 
    fields: list = ['paperId', 'corpusId', 'isInfluential', 'title', 'abstract', 'year', 'publicationDate'], 
    batch_size: int = 1000, top_k: int = 10
):
    references = [
        get_request(
            f'https://api.semanticscholar.org/graph/v1/paper/{paper["paperId"]}/references',
            headers={},
            params={'fields': ','.join(fields), 'offset': index*batch_size, 'limit': batch_size},
            return_type={'data': []}
        )['data']
        for index in range(0, paper['referenceCount'] // batch_size + 1)
    ]

    references = [
        reference for reference in flatten_list(references)
        if all(reference['citedPaper'][category] is not None for category in ['paperId', 'title', 'abstract'])
    ]

    reference2embedding = get_paper2embedding([reference['citedPaper']['paperId'] for reference in references])    
    similar_reference_indices = torch.topk(
        cos_sim(
            paper['embedding']['vector'],
            list(reference2embedding.values())
        ),
        k = min(top_k, len(reference2embedding))
    ).indices[0].tolist() if len(reference2embedding) > 0 else []
    similar_reference_ids = [list(reference2embedding.keys())[index] for index in similar_reference_indices]
    
    return [
        reference['citedPaper'] for reference in references 
        if reference['citedPaper']['paperId'] in similar_reference_ids
    ]


def get_paper2embedding(ids: list):
    return {
        paper['paperId']: paper['embedding']['vector'] 
        for paper in get_papers(ids, fields=['embedding.specter_v2'])
        if (
            type(paper) == dict and 
            'embedding' in paper.keys() and 
            paper['embedding'] is not None
        )
    }
