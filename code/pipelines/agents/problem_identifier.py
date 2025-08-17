import re
from typing import Dict, List

from .base import BaseAgent
from utils.evaluation import get_low_score_feedbacks
from utils.formatting import list_of_items_to_grammatical_text

class ProblemIdentifier(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems "
            "based on existing scientific literature, in order to aid researchers in discovering novel and significant "
            "research opportunities that can advance the field."
        )
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        self.generated: bool = False

    def reset(self):
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.generated = False

    def run(self, context: Dict) -> Dict:
        do_generation = (not self.generated) or (context.get('problem') is None)

        if do_generation:
            self.reset()
            response = self._chat(user_prompt=self._build_generation_prompt(context))
            self.generated = True
            return self.parse_output(response)
        else:
            response = self._chat(user_prompt=self._build_refinement_prompt(context))
            return self.parse_output(response)

    def _chat(self, user_prompt: str) -> str:
        self.messages.append({'role': 'user', 'content': user_prompt})
        assistant_reply = self.call(messages=self.messages)
        self.messages.append({'role': 'assistant', 'content': assistant_reply})
        return assistant_reply

    def _build_generation_prompt(self, context) -> str:
        # Intro
        prompt = (
            "You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. "
            f"This will be based on the title and abstract of the target paper, those of {len(context['references'])} related papers in the existing literature, "
            f"and {len(context['entities'])} entities potentially connected to the research area. \n\n"
        )
        # Understanding
        prompt += (
            "Understanding of the target paper, related papers, and entities is essential: \n"
            "- The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem. \n"
            "- The related papers are studies that have cited the target paper, indicating their direct relevance and connection to the primary research topic you are focusing on, and providing additional context and insights that are essential for understanding and expanding upon the target paper. \n"
            "- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, serving as auxiliary sources of inspiration or information that may be instrumental in formulating the research problem. \n\n"
        )
        # Approach
        prompt += (
            "Your approach should be systematic: \n"
            "- Start by thoroughly reading the title and abstract of the target paper to understand its core focus. \n"
            "- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective and insights relevant to the primary research topic. \n"
            "- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant. \n\n"
        )
        # Materials
        prompt += (
            "I am going to provide the target paper, related papers, and entities, as follows: \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'])
            + self._format_entities(context['entities'])
        )
        # Final
        prompt += (
            "With the provided target paper, related papers, and entities, your objective now is to formulate a research problem that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant. "
            "Before crafting the research problem, revisit the title and abstract of the target paper, to ensure it remains the focal point of your research problem identification process. \n\n"
            f"{self._format_target_paper(context['paper'])}"
            "Then, following your review of the above content, please proceed to generate one research problem with the rationale, in the format of \nProblem: \nRationale: \n"
        )
        return prompt

    def _build_refinement_prompt(self, context) -> str:
        target_feedbacks, other_feedbacks = get_low_score_feedbacks(context['problem_feedbacks'])

        # Intro
        prompt = f"You are going to refine the research problem that you formulated, which is: '{context.get('problem')}'. \n\n"
        # Understanding
        prompt += (
            f"Expert reviewers have evaluated this problem across five dimensions: Clarity, Relevance, Originality, Feasibility, and Significance, and identified key areas for improvement in {list_of_items_to_grammatical_text(target_feedbacks)}. "
            f"Your challenge is to enhance these aspects while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}. \n\n"
        )
        # Approach
        prompt += (
            "Please follow this systematic approach for refinement: \n"
            "- Begin with a comprehensive review of your research problem and its underlying rationale, while revisiting the context above in which it was formulated, including the target paper, related papers, and entities. \n"
            f"- Next, familiarize yourself with the definitions of the target evaluation criteria identified as the primary areas for improvement: {list_of_items_to_grammatical_text(target_feedbacks)}, and then proceed to carefully read the reviews and feedback provided by expert reviewers on these criteria. \n"
            "- Finally, based on the insights gained from the reviews and feedback, refine your research problem and its rationale, ensuring that the revised problem is original, clear, feasible, relevant, and significant to its field. \n\n"
        )
        # Materials
        prompt += (
            f"I am going to provide the previous problem with its rationale, followed by each of the evaluation criteria, reviews, and feedback for {list_of_items_to_grammatical_text(target_feedbacks)} needing improvement, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_feedbacks('problem', {metric: feedback for metric, feedback in context['problem_feedbacks'].items() if metric in target_feedbacks})
        )
        # Final
        prompt += (
            f"Finally, with these reviews and feedback about {list_of_items_to_grammatical_text(target_feedbacks)} in mind while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}, please craft a revised version of the problem with the rationale, in the format of \nProblem: \nRationale: \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, str]:
        match = re.search(r"Problem:\s*(.*?)\s*Rationale:\s*(.*)", text, re.DOTALL)
        return (
            {'problem': match.group(1).strip(), 'problem_rationale': match.group(2).strip()}
            if match else {'problem': None, 'problem_rationale': None}
        )
