import re
from typing import Dict, List

from .base import BaseAgent
from utils.evaluation import get_low_score_feedbacks
from utils.formatting import list_of_items_to_grammatical_text


class ExperimentDesigner(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to design robust, feasible, and impactful experiments "
            "based on identified scientific problems and proposed methodologies from existing scientific literature, "
            "in order to enable researchers to systematically test hypotheses and validate groundbreaking discoveries that can transform their respective fields."
        )
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        self.generated: bool = False

    def reset(self):
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.generated = False

    def run(self, context: Dict) -> Dict:
        do_generation = (not self.generated) or (context.get('experiment') is None)

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
            "You are going to design an experiment, aimed at validating a proposed method to address a specific research problem. Your experiment design should be clear, robust, reproducible, valid, and feasible. "
            "This will be based on a deep understanding of the research problem, scientific method, existing studies, and various entities. \n\n"
        )
        # Understanding
        prompt += (
            "Understanding of the research problem, scientific method, existing studies, and entities is essential: \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method, all serving as foundational material for designing the experiment. \n"
            "- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or information that may be instrumental in your experiment design. \n\n"
        )
        # Approach
        prompt += (
            "Your approach should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale followed by the proposed method and its rationale, to pinpoint your primary focus. \n"
            "- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic. \n"
            "- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant. \n\n"
        )
        # Materials
        prompt += (
            "I am going to provide the research problem, scientific method, existing studies (target paper & related papers), and entities, as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'])
            + self._format_entities(context['entities'])
        )
        # Final
        prompt += (
            "With the provided research problem, scientific method, existing studies, and entities, your objective now is to design an experiment that not only leverages these resources but also strives to be clear, robust, reproducible, valid, and feasible. "
            "Before crafting the experiment design, revisit the research problem and proposed method, to ensure they remain at the center of your experiment design process. \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            "Then, following your review of the above content, please proceed to outline your experiment with its rationale, in the format of \nExperiment: \nRationale: \n"
        )
        return prompt

    def _build_refinement_prompt(self, context) -> str:
        target_feedbacks, other_feedbacks = get_low_score_feedbacks(context['experiment_feedbacks'])

        # Intro
        prompt = (
            "You are going to refine the experiment that you designed to validate the proposed method for addressing the research problem. "
            "The research problem, scientific method, and experiment design are as follows: \n"
            f"Problem: {context.get('problem')} \n"
            f"Method: {context.get('method')} \n"
            f"Experiment: {context.get('experiment')} \n\n"
        )
        # Understanding
        prompt += (
            f"Expert reviewers have evaluated this experiment across five dimensions: Clarity, Validity, Robustness, Feasibility, and Reproducibility, and identified key areas for improvement in {list_of_items_to_grammatical_text(target_feedbacks)}. "
            f"Your challenge is to enhance these aspects while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}. \n\n"
        )
        # Approach
        prompt += (
            "Please follow this systematic approach for refinement: \n"
            "- Begin with a comprehensive review of your experiment design and its underlying rationale, while revisiting the context above in which it was formulated, including the research problem, scientific method, target paper, related papers, and entities. \n"
            f"- Next, familiarize yourself with the definitions of the target evaluation criteria identified as the primary areas for improvement: {list_of_items_to_grammatical_text(target_feedbacks)}, and then proceed to carefully read the reviews and feedback provided by expert reviewers on these criteria. \n"
            "- Finally, based on the insights gained from the reviews and feedback, refine your experiment design and its rationale, ensuring that the revised experiment is clear, robust, reproducible, valid, and feasible. \n\n"
        )
        # Materials
        prompt += (
            f"I am going to provide the experiment design with its rationale, followed by each of the evaluation criteria, reviews, and feedback for {list_of_items_to_grammatical_text(target_feedbacks)} needing improvement, as follows: \n\n"
            f"Experiment: {context.get('experiment')} \nRationale: {context.get('experiment_rationale')} \n\n"
            + self._format_feedbacks('experiment', {metric: feedback for metric, feedback in context['experiment_feedbacks'].items() if metric in target_feedbacks})
        )
        # Final
        prompt += (
            f"Finally, with these reviews and feedback about {list_of_items_to_grammatical_text(target_feedbacks)} in mind while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}, please craft a revised version of your experiment with its rationale, in the format of \nExperiment: \nRationale: \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, str]:
        match = re.search(r"Experiment:\s*(.*?)\s*Rationale:\s*(.*)", text, re.DOTALL)
        return (
            {'experiment': match.group(1).strip(), 'experiment_rationale': match.group(2).strip()}
            if match else {'experiment': None, 'experiment_rationale': None}
        )
