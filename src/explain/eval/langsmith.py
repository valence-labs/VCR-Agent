from verifiers.rubrics.rubric import Rubric

try:
    from langchain_core.tracers.schemas import Run
    from langsmith.schemas import EvaluationResult, Example
except ImportError as e:
    raise ImportError(
        "langchain_core and langsmith packages are required for this module. "
        "Please run `pip install langchain_core langsmith`"
    ) from e


class LangSmithRubricWrapper:
    """
    A wrapper to adapt a 'verifiers' Rubric for use with LangSmith evaluation.

    This class provides a bridge between the 'verifiers' evaluation framework and
    the LangSmith platform, allowing custom rubrics to be used as evaluators
    in LangSmith.

    Example:
        from langsmith import Client
        from explain.eval.rubrics.plausibility import PlausibilityRubric
        from explain.eval.langsmith import LangSmithRubricWrapper

        # 1. Initialize your rubric
        plausibility_rubric = PlausibilityRubric(llm_provider="openai")

        # 2. Wrap it for LangSmith
        evaluator = LangSmithRubricWrapper(rubric=plausibility_rubric)

        # 3. Run evaluation on a LangSmith dataset
        client = Client()
        client.run_on_dataset(
            dataset_name="my-evaluation-dataset",
            llm_or_chain_factory=my_llm_chain,
            evaluation=evaluator.evaluate_run,
        )
    """

    def __init__(self, rubric: Rubric):
        """
        Initializes the wrapper with a rubric instance.

        Args:
            rubric: An instance of a Rubric class from the 'verifiers' library.
        """
        self.rubric = rubric

    def evaluate_run(self, run: Run, example: Example | None = None) -> list[EvaluationResult]:
        """
        Evaluates a run using the wrapped rubric.

        This method is designed to be passed to the `evaluation` parameter of
        a LangSmith client's `run_on_dataset` method.

        Args:
            run: The LangSmith run object to evaluate.
            example: The corresponding LangSmith example object with ground truth data.

        Returns:
            A list of EvaluationResult objects, one for each metric in the rubric.
        """
        if not example or not example.inputs or not example.outputs:
            return [
                EvaluationResult(
                    key="error",
                    score=0,
                    comment="Example with populated inputs and outputs is required for rubric evaluation.",
                )
            ]

        try:
            prompt = example.inputs["question"]
            answer = example.outputs["answer"]

            if run.outputs is None:
                raise ValueError("Run outputs are missing.")

            if isinstance(run.outputs, dict):
                output_keys = ["output", "answer", "completion", "result", "response"]
                completion_key = next((k for k in output_keys if k in run.outputs), None)
                if completion_key:
                    completion = run.outputs[completion_key]
                else:
                    completion = next(iter(run.outputs.values()))
            else:
                completion = str(run.outputs)

        except (KeyError, TypeError, StopIteration, ValueError) as e:
            return [
                EvaluationResult(
                    key="error",
                    score=0,
                    comment=f"Failed to extract required data from run/example: {e}",
                )
            ]

        try:
            state = {}
            scores = self.rubric.score_completion(prompt=prompt, completion=completion, answer=answer, state=state)
        except Exception as e:
            return [
                EvaluationResult(
                    key="error",
                    score=0,
                    comment=f"Rubric evaluation failed: {e}",
                )
            ]

        if not isinstance(scores, dict):
            return [
                EvaluationResult(
                    key="error",
                    score=0,
                    comment=f"Rubric did not return a dictionary of scores. Got: {type(scores)}",
                )
            ]

        results = []
        for metric_name, score_value in scores.items():
            if isinstance(score_value, int | float):
                results.append(
                    EvaluationResult(
                        key=metric_name,
                        score=float(score_value),
                        comment=f"Evaluated by: {self.rubric.__class__.__name__}",
                    )
                )

        return results
