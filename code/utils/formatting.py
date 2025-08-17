from typing import List


def list_of_items_to_grammatical_text(items: List[str]) -> str:
    if len(items) <= 1: return ''.join(items)
    if len(items) == 2: return ' and '.join(items)
    return '{}, and {}'.format(', '.join(items[:-1]), items[-1])
