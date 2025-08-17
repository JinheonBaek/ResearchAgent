from typing import Dict
from pipelines.agents import ProblemIdentifier, ProblemValidator

class ResearchPipeline:
    def __init__(self, api_client=None, problem_iterations: int = 3):
        self.problem_iterations = max(1, problem_iterations)
        self.problem_identifier = ProblemIdentifier(api_client)
        self.problem_validator = ProblemValidator(api_client)

    def run(self, context: Dict) -> Dict:
        history = {'problems': [], 'methods': [], 'experiments': []}

        for _ in range(self.problem_iterations):
            context.update(self.problem_identifier.run(context))
            context.update(self.problem_validator.run(context))
            history['problems'].append(
                {
                    'problem': context.get('problem'),
                    'rationale': context.get('problem_rationale'),
                    'feedbacks': context.get('problem_feedbacks')
                }
            )

        context.update({'history': history})
        return context
