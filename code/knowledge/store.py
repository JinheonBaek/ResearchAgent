import math
from collections import defaultdict, Counter

import torch

from utils.data_io import load_jsonl

class KnowledgeStore(object):
    def __init__(self, file_path):
        super(KnowledgeStore, self).__init__()

        self.knowledge_base = load_jsonl(file_path)
        self.paper2entities = self.build_paper2entities()
        self.entity_counter, self.entity_cooccurrence = self.build_entity_statistics()

    def build_paper2entities(self):
        return {instance['corpusid']: instance['knowledge'] for instance in self.knowledge_base}

    def build_entity_statistics(self):
        entity_counter = Counter()
        entity_cooccurrence = defaultdict(Counter)

        for instance in self.knowledge_base:
            entities = instance['knowledge']
            entity_counter.update(entities)

            for entity_name in entities.keys():
                entity_cooccurrence[entity_name].update(
                    {key: value for key, value in entities.items() if key != entity_name}
                )

        return entity_counter, entity_cooccurrence
    
    def get_entity_log_likelihood(self, entity: str, paper_entities: list):
        conditional_log_probabilities = [
            math.log2(
                (self.entity_cooccurrence[entity][paper_entity] + 1e-16) /
                (sum(self.entity_cooccurrence[entity].values()) + 1e-16)
            ) for paper_entity in paper_entities
        ]
        return sum(conditional_log_probabilities)

    def get_entity_probability(self, entity: str):
        return self.entity_counter[entity] / sum(self.entity_counter.values())

    def get_relevant_entities(self, paper_ids: list, top_k: int = 30):
        paper_entities = sum(
            [
                Counter(self.paper2entities[paper_id]) for paper_id in paper_ids 
                if paper_id in self.paper2entities.keys()
            ], 
            start = Counter()
        )
        paper_entities = list(paper_entities.elements())

        candidate_entities = sum(
            [self.entity_cooccurrence[entity] for entity in paper_entities],
            start = Counter()
        )
        candidate_entities = [entity for entity, count in candidate_entities.items() if count >= 3]

        candidate_entities_probs = [
            (
                self.get_entity_log_likelihood(entity, paper_entities) + 
                math.log2(self.get_entity_probability(entity) + 1e-16)
            )
            for entity in candidate_entities
        ]

        _, indices = torch.topk(torch.tensor(candidate_entities_probs), k=min(top_k, len(candidate_entities)), axis=-1)
        return [candidate_entities[index] for index in indices]
