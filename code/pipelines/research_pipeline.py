from typing import Dict
from pipelines.agents import (
    ProblemIdentifier,
    ProblemValidator,
    MethodDeveloper,
    MethodValidator,
    ExperimentDesigner,
    ExperimentValidator,
)
from utils.evaluation import get_avg_feedbacks_score, get_num_feedbacks_scores


class ResearchPipeline:
    def __init__(self, api_client=None, iterations: int = 3):
        self.iterations = max(1, iterations)
        self.problem_identifier = ProblemIdentifier(api_client)
        self.problem_validator = ProblemValidator(api_client)
        self.method_developer = MethodDeveloper(api_client)
        self.method_validator = MethodValidator(api_client)
        self.experiment_designer = ExperimentDesigner(api_client)
        self.experiment_validator = ExperimentValidator(api_client)

    def run(self, context: Dict) -> Dict:
        history = {'problems': [], 'methods': [], 'experiments': []}

        # Problem ideation and validation
        for _ in range(self.iterations):
            context.update(self.problem_identifier.run(context))
            context.update(self.problem_validator.run(context))
            history['problems'].append(
                {
                    'problem': context.get('problem'),
                    'rationale': context.get('problem_rationale'),
                    'feedbacks': context.get('problem_feedbacks'),
                }
            )

        # Select the best-scored problem to use for context
        best_problem = max(
            history['problems'], 
            key=lambda problem: get_avg_feedbacks_score(problem.get('feedbacks') or {}) 
            if get_num_feedbacks_scores(problem.get('feedbacks') or {}) > 0 
            else -1
        )
        context.update(
            problem=best_problem.get('problem'),
            problem_rationale=best_problem.get('rationale'),
            problem_feedbacks=best_problem.get('feedbacks'),
        )

        # Method development and validation
        for _ in range(self.iterations):
            context.update(self.method_developer.run(context))
            context.update(self.method_validator.run(context))
            history['methods'].append(
                {
                    'method': context.get('method'),
                    'rationale': context.get('method_rationale'),
                    'feedbacks': context.get('method_feedbacks'),
                }
            )

        # Select the best-scored method to use for context
        best_method = max(
            history['methods'],
            key=lambda method: get_avg_feedbacks_score(method.get('feedbacks') or {})
            if get_num_feedbacks_scores(method.get('feedbacks') or {}) > 0
            else -1,
        )
        context.update(
            method=best_method.get('method'),
            method_rationale=best_method.get('rationale'),
            method_feedbacks=best_method.get('feedbacks'),
        )

        # Experiment design and validation
        for _ in range(self.iterations):
            context.update(self.experiment_designer.run(context))
            context.update(self.experiment_validator.run(context))
            history['experiments'].append(
                {
                    'experiment': context.get('experiment'),
                    'rationale': context.get('experiment_rationale'),
                    'feedbacks': context.get('experiment_feedbacks'),
                }
            )

        # Select the best-scored experiment to use for context
        best_experiment = max(
            history['experiments'],
            key=lambda experiment: get_avg_feedbacks_score(experiment.get('feedbacks') or {})
            if get_num_feedbacks_scores(experiment.get('feedbacks') or {}) > 0
            else -1,
        )
        context.update(
            experiment=best_experiment.get('experiment'),
            experiment_rationale=best_experiment.get('rationale'),
            experiment_feedbacks=best_experiment.get('feedbacks'),
        )

        context.update({'history': history})
        return context
