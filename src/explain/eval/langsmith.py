from typing import Any

try:
    from langchain_core.tracers.schemas import Run
except ImportError as e:
    raise ImportError(
        "langchain_core and langsmith packages are required for this module. "
        "Please run `pip install langchain_core langsmith`"
    ) from e

from explain.llm import LLMResponse


class LangSmithRubricWrapper:
    """
    Wrapper to adapt a custom rubric for use with LangSmith evaluation.

    This enables using your internal evaluators (e.g., Correctness, Plausibility)
    as LangSmith-compatible evaluators via the `evaluate_run` interface.

    Example:
        from explain.eval.rubrics.correctness import CorrectnessRubric
        from explain.eval.langsmith import LangSmithRubricWrapper
        evaluator = LangSmithRubricWrapper(CorrectnessRubric(), "scientific_correctness")
    """

    def __init__(self, rubric, rubric_name: str):
        """
        Args:
            rubric: A custom rubric object with an `evaluate(...)` method.
            rubric_name: The name of the rubric used for LangSmith evaluation output.
        """
        self.rubric = rubric
        self.rubric_name = rubric_name

    def evaluate_run(self, run: Run, example: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Evaluate a LangSmith run using the wrapped rubric.

        Args:
            run: LangSmith Run object.
            example: Corresponding LangSmith example with inputs and expected outputs.

        Returns:
            A LangSmith-compatible evaluation dictionary.
        """
        # Fail early if no example
        if not example:
            return {"key": self.rubric_name, "score": 0.0, "comment": "Example is required for rubric evaluation."}

        try:
            # Extract inputs and ground truth
            inputs = example.get("inputs", {})
            outputs = example.get("outputs", {})

            question = inputs.get("question", "")
            ground_truth = outputs.get("ground_truth") or outputs.get("answer", "")

            # Get model output
            if not run.outputs:
                return {"key": self.rubric_name, "score": 0.0, "comment": "Run outputs are missing."}

            if isinstance(run.outputs, dict):
                # Check common output keys
                output_keys = ["answer", "output", "completion", "result", "response", "content"]
                model_answer = next((run.outputs.get(k) for k in output_keys if k in run.outputs), None)
                if model_answer is None:
                    model_answer = next(iter(run.outputs.values()), "")
            else:
                model_answer = str(run.outputs)

            # Handle LLMResponse
            if isinstance(model_answer, LLMResponse):
                actual_answer = model_answer.content
                conversation_history = model_answer.messages
            else:
                actual_answer = str(model_answer)
                conversation_history = None

            # Default conversation history if needed
            if conversation_history is None:
                conversation_history = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": actual_answer},
                ]

        except Exception as e:
            return {"key": self.rubric_name, "score": 0.0, "comment": f"Error extracting data from run/example: {e}"}

        # Try to evaluate using the rubric
        try:
            score, feedback = self.rubric.evaluate(
                answer=actual_answer,
                question=question,
                ground_truth=ground_truth,
                conversation_history=conversation_history,
            )
        except TypeError:
            # Fallback if some rubrics do not support all arguments
            try:
                score, feedback = self.rubric.evaluate(answer=actual_answer, question=question)
            except TypeError:
                score, feedback = self.rubric.evaluate(actual_answer)

        except Exception as e:
            return {"key": self.rubric_name, "score": 0.0, "comment": f"Rubric evaluation failed: {e}"}

        # Prepare comment and optional tags
        comment = feedback.get("reasoning", "") if isinstance(feedback, dict) else str(feedback)
        tags = feedback.get("tags", []) if isinstance(feedback, dict) else []

        return {
            "key": self.rubric_name,
            "score": float(score),
            "comment": comment[:500] + "..." if len(comment) > 500 else comment,
            "tags": tags,
        }

    def __call__(self, run: Run, example: dict[str, Any] | None = None) -> dict[str, Any]:
        """For compatibility with LangSmith versions that expect a callable evaluator."""
        return self.evaluate_run(run, example)
