import re
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .base import BaseAgent


class ExperimentValidator(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to meticulously evaluate "
            "the experimental designs of scientific papers across diverse dimensions, "
            "in order to aid researchers in refining their experimental approaches "
            "based on your evaluations and feedback, thereby amplifying the quality and impact of their scientific contributions."
        )

        self.build_functions = {
            'Clarity': self._build_validation_prompt_for_clarity,
            'Validity': self._build_validation_prompt_for_validity,
            'Robustness': self._build_validation_prompt_for_robustness,
            'Feasibility': self._build_validation_prompt_for_feasibility,
            'Reproducibility': self._build_validation_prompt_for_reproducibility,
        }

    def run(self, context: Dict) -> Dict:
        if not context.get('experiment') or not context.get('experiment_rationale'):
            return {'experiment_feedbacks': {}}

        with ThreadPoolExecutor(max_workers=len(self.build_functions)) as executor:
            futures = {
                metric: executor.submit(self._chat, user_prompt=function(context))
                for metric, function in self.build_functions.items()
            }
            feedbacks = {metric: self.parse_output(future.result()) for metric, future in futures.items()}

        return {'experiment_feedbacks': feedbacks}

    def _chat(self, user_prompt: str) -> str:
        assistant_reply = self.call(
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return assistant_reply

    def _build_validation_prompt_for_clarity(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate an experiment design for its clarity in validating a scientific method to address a research problem, focusing on how well it is described in a clear, precise, and understandable manner, enabling others to grasp the setup, procedure, and expected outcomes. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, scientific method, existing studies, and entities, which will help in understanding the context of the designed experiment for a more comprehensive assessment. \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, method, and experiment, as well as the related papers that have been additionally referenced in their discovery phases. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem, developing the method, and designing the experiment. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, scientific method, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your clarity evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context provided by the research problem, scientific method, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the clarity of the experiment. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The experiment design is extremely unclear, with critical details missing or ambiguous, making it nearly impossible for others to understand the setup, procedure, or expected outcomes. \n"
            "-- 2. The experiment design lacks significant clarity, with many important aspects poorly explained or omitted, challenging others to grasp the essential elements of the setup, procedure, or expected outcomes. \n"
            "-- 3. The experiment design is moderately clear, but some aspects are not detailed enough, leaving room for interpretation or confusion about the setup, procedure, or expected outcomes. \n"
            "-- 4. The experiment design is mostly clear, with most aspects well-described, allowing others to understand the setup, procedure, and expected outcomes with minimal ambiguity. \n"
            "-- 5. The experiment design is exceptionally clear, precise, and detailed, enabling easy understanding of the setup, procedure, and expected outcomes, with no ambiguity or need for further clarification. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the designed experiment with its rationale, as follows: \n\n"
            f"Experiment: {context.get('experiment')} \nRationale: {context.get('experiment_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_validity(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate an experiment design for its validity in validating a scientific method to address a research problem, focusing on its appropriateness and soundness in accurately addressing the research questions or effectively validating the proposed methods to ensure that the design effectively tests what it is intended to examine. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, scientific method, existing studies, and entities, which will help in understanding the context of the designed experiment for a more comprehensive assessment. \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, method, and experiment, as well as the related papers that have been additionally referenced in their discovery phases. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem, developing the method, and designing the experiment. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, scientific method, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your validity evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context provided by the research problem, scientific method, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the validity of the experiment. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The experiment design demonstrates a fundamental misunderstanding of the research problem, lacks alignment with scientific methods, and shows no evidence of validity in addressing the research questions or testing the proposed methods. \n"
            "-- 2. The experiment design has significant flaws in its approach to the research problem and scientific method, with minimal or questionable evidence of validity, making it largely ineffective in addressing the research questions or testing the proposed methods. \n"
            "-- 3. The experiment design is generally aligned with the research problem and scientific method but has some limitations in its validity, offering moderate evidence that it can somewhat effectively address the research questions or test the proposed methods. \n"
            "-- 4. The experiment design is well-aligned with the research problem and scientific method, providing strong evidence of validity and effectively addressing the research questions and testing the proposed methods, despite minor limitations. \n"
            "-- 5. The experiment design excellently aligns with the research problem and scientific method, demonstrating robust evidence of validity and outstandingly addressing the research questions and testing the proposed methods without significant limitations. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the designed experiment with its rationale, as follows: \n\n"
            f"Experiment: {context.get('experiment')} \nRationale: {context.get('experiment_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_robustness(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate an experiment design for its robustness in validating a scientific method to address a research problem, focusing on its durability across a wide range of conditions and variables to ensure that the outcomes are not reliant on a few specific cases and remain consistent across a broad spectrum of scenarios. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, scientific method, existing studies, and entities, which will help in understanding the context of the designed experiment for a more comprehensive assessment. \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, method, and experiment, as well as the related papers that have been additionally referenced in their discovery phases. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem, developing the method, and designing the experiment. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, scientific method, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your robustness evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context provided by the research problem, scientific method, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the robustness of the experiment. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The experiment design demonstrates a fundamental lack of understanding of the scientific method, with no evidence of durability or adaptability across varying conditions, leading to highly unreliable and non-replicable results. \n"
            "-- 2. The experiment design shows minimal consideration for robustness, with significant oversights in addressing variability and ensuring consistency across different scenarios, resulting in largely unreliable outcomes. \n"
            "-- 3. The experiment design adequately addresses some aspects of robustness but lacks comprehensive measures to ensure durability and consistency across a wide range of conditions, leading to moderate reliability. \n"
            "-- 4. The experiment design incorporates a solid understanding of robustness, with clear efforts to ensure the experiment's durability and consistency across diverse conditions, though minor improvements are still possible for optimal reliability. \n"
            "-- 5. The experiment design exemplifies an exceptional commitment to robustness, with meticulous attention to durability and adaptability across all possible conditions, ensuring highly reliable and universally applicable results. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the designed experiment with its rationale, as follows: \n\n"
            f"Experiment: {context.get('experiment')} \nRationale: {context.get('experiment_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_feasibility(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate an experiment design for its feasibility in validating a scientific method to address a research problem, focusing on how well it can realistically be implemented with the available resources, time, and technological or methodological constraints to ensure that the experiment is practical and achievable. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, scientific method, existing studies, and entities, which will help in understanding the context of the designed experiment for a more comprehensive assessment. \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, method, and experiment, as well as the related papers that have been additionally referenced in their discovery phases. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem, developing the method, and designing the experiment. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, scientific method, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your feasibility evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context provided by the research problem, scientific method, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the feasibility of the experiment. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The experiment design is fundamentally unfeasible, with insurmountable resource, time, or technological constraints that make implementation virtually impossible within the proposed framework. \n"
            "-- 2. The experiment design faces significant feasibility challenges, including major resource, time, or technological limitations, that heavily compromise its practical execution and likelihood of success. \n"
            "-- 3. The experiment design is somewhat feasible, with moderate constraints on resources, time, or technology that could be addressed with adjustments, though these may not guarantee success. \n"
            "-- 4. The experiment design is largely feasible, with minor resource, time, or technological limitations that can be effectively managed or mitigated, ensuring a high probability of successful implementation. \n"
            "-- 5. The experiment design is highly feasible, with no significant constraints on resources, time, or technology, indicating that it can be implemented smoothly and successfully within the proposed framework. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the designed experiment with its rationale, as follows: \n\n"
            f"Experiment: {context.get('experiment')} \nRationale: {context.get('experiment_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_reproducibility(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate an experiment design for its reproducibility in validating a scientific method to address a research problem, focusing on how well the information provided is sufficient and detailed enough for other researchers to reproduce the experiment using the same methodology and conditions to ensure the reliability of the findings. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, scientific method, existing studies, and entities, which will help in understanding the context of the designed experiment for a more comprehensive assessment. \n"
            "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, method, and experiment, as well as the related papers that have been additionally referenced in their discovery phases. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem, developing the method, and designing the experiment. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, scientific method, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            f"Scientific method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your reproducibility evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context provided by the research problem, scientific method, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the reproducibility of the experiment. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The experiment design lacks critical details, making it virtually impossible for other researchers to replicate the study under the same conditions or methodologies. \n"
            "-- 2. The experiment provides some essential information but omits significant details needed for replication, leading to considerable ambiguity in methodology or conditions. \n"
            "-- 3. The experiment design includes sufficient details for replication, but lacks clarity or completeness in certain areas, posing challenges for seamless reproducibility. \n"
            "-- 4. The experiment is well-documented with clear, detailed instructions and methodologies that allow for consistent replication, albeit with minor areas for improvement. \n"
            "-- 5. The experiment design is exemplary in its clarity, detail, and comprehensiveness, ensuring that other researchers can precisely and effortlessly replicate the study under identical conditions and methodologies. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the designed experiment with its rationale, as follows: \n\n"
            f"Experiment: {context.get('experiment')} \nRationale: {context.get('experiment_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, Any]:
        match = re.search(r"Review:\s*(.*?)\nFeedback:\s*(.*?)\nRating(?:\s*\(1-5\))?:\s*([1-5])", text, re.DOTALL | re.IGNORECASE)
        return (
            {'review': match.group(1).strip(), 'feedback': match.group(2).strip(), 'rating': int(match.group(3))}
            if match else {'review': None, 'feedback': None, 'rating': None}
        )
