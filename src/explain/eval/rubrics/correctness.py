import json
from typing import Any

from verifiers.parsers.parser import Parser
from verifiers.rubrics.judge_rubric import JudgeRubric

from explain.llm._client import create_llm_client


class BinaryCorrectnessEvaluator:
    """
    A decoupled evaluator for assessing the correctness of a response against a ground truth answer.
    """

    def __init__(self, llm_provider: str = "anthropic", **kwargs):
        """
        Initializes the correctness evaluator.

        Args:
            llm_provider: The LLM provider to use ('anthropic', 'gemini', or 'openai').
            **kwargs: Additional arguments for the LLM client.
        """
        self.llm_client = create_llm_client(provider=llm_provider, **kwargs)
        self.parser = Parser()
        self.judge_prompt_template = """
        You are an expert biologist comparing a model's response to a ground truth answer.
        
        Question:
        ```
        {question}
        ```

        Ground Truth Answer:
        ```
        {answer}
        ```

        Model's Response:
        ```
        {response}
        ```
                
        Assess how similar the model's response is to the ground truth answer.
        Consider the factual accuracy, completeness, and key biological concepts mentioned.
        
        Respond either "yes" or "no" only. If the response is correct, respond "yes". If the response is incorrect, respond "no".
        """

    def _evaluate_correctness(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs):
        """
        Runs the LLM judge to evaluate correctness and stores results in the state dictionary.
        """
        if "correctness" in state:
            return

        if isinstance(prompt, list):
            question = prompt[-1]["content"]
        else:
            question = prompt

        response_text = self.parser.parse_answer(completion)

        judge_prompt = self.judge_prompt_template.format(question=question, answer=answer, response=response_text)

        llm_response = self.llm_client.generate(messages=[{"role": "user", "content": judge_prompt}])

        try:
            state["correctness"] = llm_response.lower() == "yes"

        except (json.JSONDecodeError, TypeError):
            state["correctness"] = 0
        return state

    def correctness(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs) -> float:
        """
        Reward function for correctness.
        """
        state = self._evaluate_correctness(prompt, completion, answer, state, **kwargs)
        return state.get("correctness", 0.0)


class CorrectnessRubric(JudgeRubric):
    """
    A rubric that evaluates the correctness of a response against a ground truth answer.
    """

    def __init__(self, llm_provider: str = "anthropic", **kwargs):
        """
        Initializes the correctness rubric.

        Args:
            llm_provider: The LLM provider to use ('anthropic', 'gemini', or 'openai').
            **kwargs: Additional arguments for the JudgeRubric.
        """
        super().__init__(judge_prompt="", parallelize_scoring=False, **kwargs)

        evaluator = BinaryCorrectnessEvaluator(llm_provider=llm_provider)

        self.judge_client = evaluator.llm_client
        self.judge_model = evaluator.llm_client.config.model
        self.add_reward_func(evaluator.correctness, weight=1.0)
