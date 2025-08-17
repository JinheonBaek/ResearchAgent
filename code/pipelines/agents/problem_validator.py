import re
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .base import BaseAgent


class ProblemValidator(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to assess the quality and validity "
            "of scientific problems across diverse dimensions, in order to aid researchers in "
            "refining their problems based on your evaluations and feedback, thereby enhancing "
            "the impact and reach of their work."
        )

        self.build_functions = {
            'Clarity': self._build_validation_prompt_for_clarity,
            'Relevance': self._build_validation_prompt_for_relevance,
            'Originality': self._build_validation_prompt_for_originality,
            'Feasibility': self._build_validation_prompt_for_feasibility,
            'Significance': self._build_validation_prompt_for_significance,
        }

    def run(self, context: Dict) -> Dict:
        if not context.get('problem') or not context.get('problem_rationale'):
            return {'problem_feedbacks': {}}

        with ThreadPoolExecutor(max_workers=len(self.build_functions)) as executor:
            futures = {
                metric: executor.submit(self._chat, user_prompt=function(context))
                for metric, function in self.build_functions.items()
            }
            feedbacks = {metric: self.parse_output(future.result()) for metric, future in futures.items()}

        return {'problem_feedbacks': feedbacks}

    def _chat(self, user_prompt: str) -> str:
        assistant_reply = self.call(
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return assistant_reply

    def _build_validation_prompt_for_clarity(self, context) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a research problem for its clarity, focusing on how well it is defined in a clear, precise, and understandable manner. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the existing studies and entities that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information when formulating the research problem. \n\n"
        )
        # Materials
        prompt += (
            "The existing studies (target paper & related papers) and entities are as follows: \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your clarity evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the clarity of the problem. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The problem is presented in a highly ambiguous manner, lacking clear definition and leaving significant room for interpretation or confusion. \n"
            "-- 2. The problem is somewhat defined but suffers from vague terms and insufficient detail, making it challenging to grasp the full scope or objective. \n"
            "-- 3. The problem is stated in a straightforward manner, but lacks the depth or specificity needed to fully convey the nuances and boundaries of the research scope. \n"
            "-- 4. The problem is clearly articulated with precise terminology and sufficient detail, providing a solid understanding of the scope and objectives with minimal ambiguity. \n"
            "-- 5. The problem is exceptionally clear, concise, and specific, with every term and aspect well-defined, leaving no room for misinterpretation and fully encapsulating the research scope and aims. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the research problem with its rationale, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_relevance(self, context) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a research problem for its relevance, focusing on how well it is pertinent and applicable to the current field or context of study. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the existing studies and entities that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information when formulating the research problem. \n\n"
        )
        # Materials
        prompt += (
            "The existing studies (target paper & related papers) and entities are as follows: \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your relevance evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the relevance of the problem. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The problem shows almost no relevance to the current field, failing to connect with the established context or build upon existing work. \n"
            "-- 2. The problem has minimal relevance, with only superficial connections to the field and a lack of meaningful integration with prior studies. \n"
            "-- 3. The problem is somewhat relevant, making a moderate attempt to align with the field but lacking significant innovation or depth. \n"
            "-- 4. The problem is relevant and well-connected to the field, demonstrating a good understanding of existing work and offering promising contributions. \n"
            "-- 5. The problem is highly relevant, deeply integrated with the current context, and represents a significant advancement in the field. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the research problem with its rationale, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_originality(self, context) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a research problem for its originality, focusing on how well it presents a novel challenge or unique perspective that has not been extensively explored before. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the existing studies and entities that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information when formulating the research problem. \n\n"
        )
        # Materials
        prompt += (
            "The existing studies (target paper & related papers) and entities are as follows: \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your originality evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the originality of the problem. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The problem exhibits no discernible originality, closely mirroring existing studies without introducing any novel perspectives or challenges. \n"
            "-- 2. The problem shows minimal originality, with slight variations from known studies, lacking significant new insights or innovative approaches. \n"
            "-- 3. The problem demonstrates moderate originality, offering some new insights or angles, but these are not sufficiently groundbreaking or distinct from existing work. \n"
            "-- 4. The problem is notably original, presenting a unique challenge or perspective that is well-differentiated from existing studies, contributing valuable new understanding to the field. \n"
            "-- 5. The problem is highly original, introducing a pioneering challenge or perspective that has not been explored before, setting a new direction for future research. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the research problem with its rationale, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_feasibility(self, context) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a research problem for its feasibility, focusing on how well it can realistically be investigated or solved with the available resources and within reasonable constraints. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the existing studies and entities that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information when formulating the research problem. \n\n"
        )
        # Materials
        prompt += (
            "The existing studies (target paper & related papers) and entities are as follows: \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your feasibility evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the feasibility of the problem. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The problem is fundamentally infeasible due to insurmountable resource constraints, lack of foundational research, or critical methodological flaws. \n"
            "-- 2. The problem faces significant feasibility challenges related to resource availability, existing knowledge gaps, or technical limitations, making progress unlikely. \n"
            "-- 3. The problem is feasible to some extent but faces notable obstacles in resources, existing research support, or technical implementation, which could hinder significant advancements. \n"
            "-- 4. The problem is mostly feasible with manageable challenges in resources, supported by adequate existing research, and has a clear, achievable methodology, though minor issues may persist. \n"
            "-- 5. The problem is highly feasible with minimal barriers, well-supported by existing research, ample resources, and a robust, clear methodology, promising significant advancements. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the research problem with its rationale, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def _build_validation_prompt_for_significance(self, context) -> str:
        # Intro
        prompt = (
            "You are going to evaluate a research problem for its significance, focusing on how well it demonstrates the importance and potential impact of solving the problem, including its contribution to the field or its broader implications. \n\n"
        )
        # Understanding
        prompt += (
            "As part of your evaluation, you can refer to the existing studies and entities that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment. \n"
            "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem. \n"
            "- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information when formulating the research problem. \n\n"
        )
        # Materials
        prompt += (
            "The existing studies (target paper & related papers) and entities are as follows: \n\n"
            + self._format_target_paper(context['paper'])
            + self._format_related_papers(context['references'], include_abstract=False)
            + self._format_entities(context['entities'])
        )
        # Approach
        prompt += (
            "Now, proceed with your significance evaluation approach that should be systematic: \n"
            "- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies and entities mentioned above. \n"
            "- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the significance of the problem. \n"
            "- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified: \n"
            "-- 1. The problem shows minimal to no significance, lacking relevance or potential impact in advancing the field or contributing to practical applications. \n"
            "-- 2. The problem has limited significance, with a narrow scope of impact and minor contributions to the field, offering little to no practical implications. \n"
            "-- 3. The problem demonstrates average significance, with some contributions to the field and potential practical implications, but lacks innovation or broader impact. \n"
            "-- 4. The problem is significant, offering notable contributions to the field and valuable practical implications, with evidence of potential for broader impact and advancement. \n"
            "-- 5. The problem presents exceptional significance, with groundbreaking contributions to the field, broad and transformative potential impacts, and substantial practical applications across diverse domains. \n\n"
        )
        # Final
        prompt += (
            "I am going to provide the research problem with its rationale, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            "After your evaluation of the above content, please provide your review, feedback, and rating, in the format of \nReview: \nFeedback: \nRating (1-5): \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, Any]:
        match = re.search(r"Review:\s*(.*?)\nFeedback:\s*(.*?)\nRating(?:\s*\(1-5\))?:\s*([1-5])", text, re.DOTALL | re.IGNORECASE)
        return (
            {'review': match.group(1).strip(), 'feedback': match.group(2).strip(), 'rating': int(match.group(3))}
            if match else {'review': None, 'feedback': None, 'rating': None}
        )
