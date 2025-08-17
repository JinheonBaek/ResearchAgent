import re
from typing import Dict, List

from .base import BaseAgent
from utils.evaluation import get_low_score_feedbacks
from utils.formatting import list_of_items_to_grammatical_text

class MethodDeveloper(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid methodologies "
            "to solve newly identified scientific problems derived from existing scientific literature, "
            "in order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in their fields."
        )
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        self.generated: bool = False

    def reset(self):
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.generated = False

    def run(self, context: Dict) -> Dict:
        do_generation = (not self.generated) or (context.get('method') is None)

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
            "You are going to propose a scientific method to address a specific research problem. Your method should be clear, innovative, rigorous, valid, and generalizable. "
            "This will be based on a deep understanding of the research problem, its rationale, existing studies, and various entities. \n\n"
        )
        # Understanding
        prompt += (
            "Understanding of the research problem, existing studies, and entities is essential: \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities, which should be the cornerstone of your method development. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the problem discovery phase, all serving as foundational material for developing the method. \n"
            "- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or information that may be instrumental in method development. \n\n"
        )
        # Approach
        prompt += (
            "Your approach should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale, to understand your primary focus. \n"
            "- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic. \n"
            "- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant. \n\n"
        )
        # Materials
        prompt += (
            "I am going to provide the research problem, existing studies (target paper & related papers), and entities, as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'])
            + self._format_entities(context['entities'])
        )
        # Final
        prompt += (
            "With the provided research problem, existing studies, and entities, your objective now is to formulate a method that not only leverages these resources but also strives to be clear, innovative, rigorous, valid, and generalizable. "
            "Before crafting the method, revisit the research problem, to ensure it remains the focal point of your method development process. \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            "Then, following your review of the above content, please proceed to propose your method with its rationale, in the format of \nMethod: \nRationale: \n"
        )
        return prompt

    def _build_refinement_prompt(self, context) -> str:
        target_feedbacks, other_feedbacks = get_low_score_feedbacks(context['method_feedbacks'])

        # Intro
        prompt = (
            "You are going to refine the scientific method that you proposed to address the specific research problem. "
            "The research problem and the scientific method are as follows: \n"
            f"Problem: {context.get('problem')} \n"
            f"Method: {context.get('method')} \n\n"
        )
        # Understanding
        prompt += (
            f"Expert reviewers have evaluated this method across five dimensions: Clarity, Validity, Rigorousness, Innovativeness, and Generalizability, and identified key areas for improvement in {list_of_items_to_grammatical_text(target_feedbacks)}. "
            f"Your challenge is to enhance these aspects while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}. \n\n"
        )
        # Approach
        prompt += (
            "Please follow this systematic approach for refinement: \n"
            "- Begin with a comprehensive review of your proposed method and its underlying rationale, while revisiting the context above in which it was formulated, including the research problem, target paper, related papers, and entities. \n"
            f"- Next, familiarize yourself with the definitions of the target evaluation criteria identified as the primary areas for improvement: {list_of_items_to_grammatical_text(target_feedbacks)}, and then proceed to carefully read the reviews and feedback provided by expert reviewers on these criteria. \n"
            "- Finally, based on the insights gained from the reviews and feedback, refine your scientific method and its rationale, ensuring that the revised method is clear, valid, rigorous, innovative, and generalizable to its field. \n\n"
        )
        # Materials
        prompt += (
            f"I am going to provide the previous method with its rationale, followed by each of the evaluation criteria, reviews, and feedback for {list_of_items_to_grammatical_text(target_feedbacks)} needing improvement, as follows: \n\n"
            f"Method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_feedbacks('method', {metric: feedback for metric, feedback in context['method_feedbacks'].items() if metric in target_feedbacks})
        )
        # Final
        prompt += (
            f"Finally, with these reviews and feedback about {list_of_items_to_grammatical_text(target_feedbacks)} in mind while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}, please craft a revised version of the method with the rationale, in the format of \nMethod: \nRationale: \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, str]:
        match = re.search(r"Method:\s*(.*?)\s*Rationale:\s*(.*)", text, re.DOTALL)
        return (
            {'method': match.group(1).strip(), 'method_rationale': match.group(2).strip()}
            if match else {'method': None, 'method_rationale': None}
        )
