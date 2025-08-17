import re
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .base import BaseAgent


class MethodValidator(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to assess the quality and "
            "soundness of scientific methods across diverse dimensions, in order to aid "
            "researchers in refining their methods based on your evaluations and feedback, "
            "thereby enhancing the impact and reach of their work."
        )

        self.build_functions = {
            'Clarity': self._build_validation_prompt_for_clarity,
            'Validity': self._build_validation_prompt_for_validity,
            'Rigorousness': self._build_validation_prompt_for_rigorousness,
            'Innovativeness': self._build_validation_prompt_for_innovativeness,
            'Generalizability': self._build_validation_prompt_for_generalizability,
        }

    def run(self, context: Dict) -> Dict:
        if not context.get('method') or not context.get('method_rationale'):
            return {'method_feedbacks': {}}

        with ThreadPoolExecutor(max_workers=len(self.build_functions)) as executor:
            futures = {
                metric: executor.submit(self._chat, user_prompt=function(context))
                for metric, function in self.build_functions.items()
            }
            feedbacks = {metric: self.parse_output(future.result()) for metric, future in futures.items()}

        return {'method_feedbacks': feedbacks}

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
            "You are going to evaluate a scientific method for its clarity in addressing a research problem, focusing on how well it is described in a clear, precise, and understandable manner that allows for replication and comprehension of the approach. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, existing studies, and entities, which will help in understanding the context of the proposed method for a more comprehensive assessment. \n"
            "- The research problem has been used as the cornerstone of the method development, formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem and developing the method. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your clarity evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context provided by the research problem, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the clarity of the method. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The method is explained in an extremely vague or ambiguous manner, making it impossible to understand or replicate the approach without additional information or clarification. \n"
            "-- 2. The method is described with some detail, but significant gaps in explanation or logic leave the reader with considerable confusion and uncertainty about how to apply or replicate the approach. \n"
            "-- 3. The method is described with sufficient detail to understand the basic approach, but lacks the precision or specificity needed to fully replicate or grasp the nuances of the methodology without further guidance. \n"
            "-- 4. The method is clearly and precisely described, with most details provided to allow for replication and comprehension, though minor areas may benefit from further clarification or elaboration. \n"
            "-- 5. The method is articulated in an exceptionally clear, precise, and detailed manner, enabling straightforward replication and thorough understanding of the approach with no ambiguities. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the proposed method with its rationale, as follows: \n\n"
            f"Method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_validity(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a scientific method for its validity in addressing a research problem, focusing on the accuracy, relevance, and soundness of the approach to ensure that it is appropriate and directly relevant to the objectives of the study. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, existing studies, and entities, which will help in understanding the context of the proposed method for a more comprehensive assessment. \n"
            "- The research problem has been used as the cornerstone of the method development, formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem and developing the method. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your validity evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context provided by the research problem, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the validity of the method. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The method shows a fundamental misunderstanding of the research problem and lacks any credible alignment with established scientific principles or relevant studies. \n"
            "-- 2. The method partially addresses the research problem but exhibits significant flaws in its scientific underpinning, making its validity questionable despite some alignment with existing literature. \n"
            "-- 3. The method adequately addresses the research problem but with some limitations in its scientific validity, showing a mix of strengths and weaknesses in its alignment with related studies. \n"
            "-- 4. The method effectively addresses the research problem, demonstrating a strong scientific basis and sound alignment with existing literature, albeit with minor areas for improvement. \n"
            "-- 5. The method exemplifies an exceptional understanding of the research problem, grounded in a robust scientific foundation, and shows exemplary integration and advancement of existing studies' findings. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the proposed method with its rationale, as follows: \n\n"
            f"Method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_rigorousness(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a scientific method for its rigorousness in addressing a research problem, focusing on the thoroughness, precision, and consistency of the approach to ensure that it is systematic, well-structured, and adheres to high standards of research quality. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, existing studies, and entities, which will help in understanding the context of the proposed method for a more comprehensive assessment. \n"
            "- The research problem has been used as the cornerstone of the method development, formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem and developing the method. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your rigorousness evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context provided by the research problem, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the rigorousness of the method. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The method demonstrates a fundamental lack of systematic approach, with significant inconsistencies and inaccuracies in addressing the research problem, showing a disregard for established research standards. \n"
            "-- 2. The method shows a minimal level of systematic effort but is marred by notable inaccuracies, lack of precision, and inconsistencies that undermine the rigorousness of the method in tackling the research problem. \n"
            "-- 3. The method exhibits an average level of systematic structure and adherence to research standards but lacks the thoroughness, precision, and consistency required for a rigorous scientific inquiry. \n"
            "-- 4. The method is well-structured and systematic, with a good level of precision and consistency, indicating a strong adherence to research standards, though it falls short of exemplifying the highest level of rigorousness. \n"
            "-- 5. The method exemplifies exceptional rigorousness, with outstanding thoroughness, precision, and consistency in its systematic approach, setting a benchmark for high standards in scientific research quality. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the proposed method with its rationale, as follows: \n\n"
            f"Method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_innovativeness(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a scientific method for its innovativeness in addressing a research problem, focusing on how well it introduces new techniques, approaches, or perspectives to the research field that differ from standard research practices and advance them in the field. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, existing studies, and entities, which will help in understanding the context of the proposed method for a more comprehensive assessment. \n"
            "- The research problem has been used as the cornerstone of the method development, formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem and developing the method. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your innovativeness evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context provided by the research problem, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the innovativeness of the method. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The method introduces no novel elements, fully relying on existing techniques without any attempt to modify or adapt them for the specific research problem, showing a lack of innovativeness. \n"
            "-- 2. The method shows minimal innovation, with only slight modifications to existing techniques that do not substantially change or improve the approach to the research problem. \n"
            "-- 3. The method demonstrates moderate innovativeness, incorporating known techniques with some new elements or combinations that offer a somewhat fresh approach to the research problem but fall short of a significant breakthrough. \n"
            "-- 4. The method is highly innovative, introducing new techniques or novel combinations of existing methods that significantly differ from standard practices, offering a new perspective or solution to the research problem. \n"
            "-- 5. The method represents a groundbreaking innovation, fundamentally transforming the approach to the research problem with novel techniques or methodologies that redefine the field's standard practices. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the proposed method with its rationale, as follows: \n\n"
            f"Method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_generalizability(self, context: Dict) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a scientific method for its generalizability in addressing a research problem, focusing on how well it can be applied to or is relevant for other contexts, populations, or settings beyond the scope of the study. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the research problem, existing studies, and entities, which will help in understanding the context of the proposed method for a more comprehensive assessment. \n"
            "- The research problem has been used as the cornerstone of the method development, formulated based on an in-depth review of existing studies and a potential exploration of relevant entities. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information in formulating the problem and developing the method. \n\n"
        )
        # Materials
        prompt += (
            "The research problem, existing studies (target paper & related papers), and entities are as follows: \n\n"
            f"Research problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your generalizability evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context provided by the research problem, existing studies, and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the generalizability of the method. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The method shows no adaptability, failing to extend its applicability beyond its original context or dataset, showing a complete lack of generalizability. \n"
            "-- 2. The method demonstrates minimal adaptability, with limited evidence of potential applicability to contexts slightly different from the original. \n"
            "-- 3. The method exhibits some level of adaptability, suggesting it could be applicable to related contexts or datasets with modifications. \n"
            "-- 4. The method is adaptable and shows evidence of applicability to a variety of contexts or datasets beyond the original. \n"
            "-- 5. The method is highly adaptable, demonstrating clear evidence of broad applicability across diverse contexts, populations, and settings. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the proposed method with its rationale, as follows: \n\n"
            f"Method: {context.get('method')} \nRationale: {context.get('method_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, Any]:
        match = re.search(r"Review:\s*(.*?)\nFeedback:\s*(.*?)\nRating(?:\s*\(1-5\))?:\s*([1-5])", text, re.DOTALL | re.IGNORECASE)
        return (
            {'review': match.group(1).strip(), 'feedback': match.group(2).strip(), 'rating': int(match.group(3))}
            if match else {'review': None, 'feedback': None, 'rating': None}
        )
