from typing import Dict, Any

METRIC2DESCRIPTION = {
    'problem': {
        'Clarity': 'It assesses whether the problem is defined in a clear, precise, and understandable manner.',
        'Relevance': 'It measures whether the problem is pertinent and applicable to the current field or context of study.',
        'Originality': 'It evaluates whether the problem presents a novel challenge or unique perspective that has not been extensively explored before.',
        'Feasibility': 'It examines whether the problem can realistically be investigated or solved with the available resources and within reasonable constraints.',
        'Significance': 'It assesses the importance and potential impact of solving the problem, including its contribution to the field or its broader implications.'
    },
    'method': {
        'Clarity': 'It assesses whether the method is described in a clear, precise, and understandable manner that allows for replication and comprehension of the approach.',
        'Validity': 'It measures the accuracy, relevance, and soundness of the method in addressing the research problem, ensuring that it is appropriate and directly relevant to the objectives of the study.',
        'Rigorousness': 'It examines the thoroughness, precision, and consistency of the method, ensuring that the approach is systematic, well-structured, and adheres to high standards of research quality.',
        'Innovativeness': 'It evaluates whether the method introduces new techniques, approaches, or perspectives to the research field that differ from standard research practices and advance them in the field.',
        'Generalizability': 'It assesses the extent to which the method can be applied to or is relevant for other contexts, populations, or settings beyond the scope of the study.'
    },
    'experiment': {
        'Clarity': 'It determines whether the experiment design is described in a clear, precise, and understandable manner, enabling others to grasp the setup, procedure, and expected outcomes.',
        'Validity': 'It measures the appropriateness and soundness of the experimental design in accurately addressing the research questions or effectively validating the proposed methods, ensuring that the design effectively tests what it is intended to examine.',
        'Robustness': 'It evaluates the durability of the experimental design across a wide range of conditions and variables, ensuring that the outcomes are not reliant on a few specific cases and remain consistent across a broad spectrum of scenarios.',
        'Feasibility': 'It evaluates whether the experiment design can realistically be implemented with the available resources, time, and technological or methodological constraints, ensuring that the experiment is practical and achievable.',
        'Reproducibility': 'It examines whether the information provided is sufficient and detailed enough for other researchers to reproduce the experiment using the same methodology and conditions, ensuring the reliability of the findings.'
    }
}

def get_feedbacks_scores(feedbacks: Dict[str, Dict[str, Any]]):
    return [feedback['rating'] for feedback in feedbacks.values() if feedback['rating']]

def get_feedback2score(feedbacks: Dict[str, Dict[str, Any]]):
    return {metric: feedback['rating'] for metric, feedback in feedbacks.items() if feedback['rating']}

def get_num_feedbacks_scores(feedbacks: Dict[str, Dict[str, Any]]):
    return len(get_feedbacks_scores(feedbacks))

def get_avg_feedbacks_score(feedbacks: Dict[str, Dict[str, Any]]):
    return sum(get_feedbacks_scores(feedbacks)) / len(get_feedbacks_scores(feedbacks))

def get_min_feedbacks_score(feedbacks: Dict[str, Dict[str, Any]]):
    return min(get_feedbacks_scores(feedbacks))

def get_low_score_feedbacks(feedbacks: Dict[str, Dict[str, Any]], target_score: int = 5):
    target_feedbacks = [metric for metric, score in get_feedback2score(feedbacks).items() if score < target_score]
    other_feedbacks = list(set(feedbacks.keys()) - set(target_feedbacks))
    return target_feedbacks, other_feedbacks
