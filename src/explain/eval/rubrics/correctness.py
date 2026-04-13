from typing import Any

from verifiers.parsers.parser import Parser
from verifiers.rubrics.judge_rubric import JudgeRubric

from explain.llm import LLMResponse, create_client


class BinaryCorrectnessEvaluator:
    """
    A decoupled evaluator for assessing the correctness of a response against a ground truth answer.
    """

    def __init__(self, llm_provider: str | None = None, **kwargs):
        """
        Initializes the correctness evaluator.

        Args:
            llm_provider: The LLM provider to use ('anthropic', 'gemini', or 'openai').
            **kwargs: Additional arguments for the LLM client.
        """
        self.llm_client = create_client(provider=llm_provider, **kwargs)
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
        Your final respond should be in the tag <answer>yes</answer> or <answer>no</answer>.
        """

    def _evaluate(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs):
        """
        Runs the LLM judge to evaluate correctness and stores results in the state dictionary.
        """
        if "judge_response" in state:
            return state

        if isinstance(prompt, list):
            question = prompt[-1]["content"]
        else:
            question = prompt

        response_text = self.parser.parse_answer(completion)

        judge_prompt = self.judge_prompt_template.format(question=question, answer=answer, response=response_text)

        llm_response = self.llm_client.generate(messages=[{"role": "user", "content": judge_prompt}])
        state["judge_response"] = llm_response
        return state

    def correctness(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs) -> float:
        """
        Reward function for correctness.
        """
        state = self._evaluate(prompt, completion, answer, state, **kwargs)
        judge_response: LLMResponse = state["judge_response"]
        extracted_response = judge_response.content.split('<answer>')[1].split('</answer>')[0]
        if "yes" == extracted_response.lower():
            return {'score': 1.0, 'full_response': judge_response.content}
        return {'score': 0.0, 'full_response': judge_response.content}


class CorrectnessRubric(JudgeRubric):
    """
    A rubric that evaluates the correctness of a response against a ground truth answer.
    """

    def __init__(self, llm_provider: str = "litellm", **kwargs):
        """
        Initializes the correctness rubric.

        Args:
            llm_provider: The LLM provider to use ('anthropic', 'gemini', or 'openai').
            **kwargs: Additional arguments for the JudgeRubric.
        """

        self.evaluator = BinaryCorrectnessEvaluator(llm_provider=llm_provider)
        super().__init__(judge_prompt="", judge_client=self.evaluator.llm_client, parallelize_scoring=False, **kwargs)
        self.judge_model = self.evaluator.llm_client.config.model

    def judge(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs) -> float:
        response = self.evaluator.correctness(prompt, completion, answer, state, **kwargs)
        return {"correctness": response['score'], "full_response": response['full_response']}
