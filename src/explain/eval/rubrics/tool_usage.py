import json
from typing import Any

from verifiers.rubrics.rubric import Rubric

from explain.eval.tools import ToolVerifier


class ToolUsageRubric(Rubric):
    """
    A rubric that evaluates the usage of tools within a conversation history.

    This rubric provides rewards for:
    - The total number of tool calls made.
    - The number of tool calls that execute successfully.
    - The number of tool calls that match a provided ground truth.
    """

    def __init__(self, ground_truth: list[dict[str, Any]] | None = None):
        """
        Initializes the tool usage rubric.

        Args:
            ground_truth: An optional list of ground truth tool calls to compare against.
                          Each item should be a dictionary with 'tool_name' and 'tool_args'.
        """
        self.ground_truth = ground_truth

        # Define the reward functions to be used.
        reward_funcs = [
            self.total_tool_calls,
            self.successful_tool_calls,
        ]

        # Only add the ground truth reward function if ground truth is provided.
        if self.ground_truth is not None:
            reward_funcs.append(self.correct_tool_calls)

        # Initialize the base Rubric with the reward functions.
        # Weights can be set by the user after initialization.
        super().__init__(funcs=reward_funcs)

    def _get_assistant_tool_calls(self, completion: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extracts all tool calls from assistant messages in a conversation history."""
        all_tool_calls = []
        for msg in completion:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    all_tool_calls.extend(tool_calls)
        return all_tool_calls

    def total_tool_calls(self, completion: list[dict[str, Any]], **kwargs) -> float:
        """Counts the total number of tool calls in the conversation history."""
        return float(len(self._get_assistant_tool_calls(completion)))

    def successful_tool_calls(self, completion: list[dict[str, Any]], **kwargs) -> float:
        """Counts the number of tool calls that execute without returning an error."""
        tool_calls = self._get_assistant_tool_calls(completion)
        if not tool_calls:
            return 0.0

        successful_calls = 0
        for tool_call in tool_calls:
            result_str = ToolVerifier.call_tool(tool_call)
            try:
                result_dict = json.loads(result_str)
                # A successful call is one that does not have an 'error' key in its result.
                if "error" not in result_dict:
                    successful_calls += 1
            except (json.JSONDecodeError, TypeError):
                # If the result isn't valid JSON, it's considered an error.
                continue

        return float(successful_calls)

    def correct_tool_calls(self, completion: list[dict[str, Any]], **kwargs) -> float:
        """
        Counts the number of tool calls that match the provided ground truth.
        This reward is only calculated if ground_truth is provided.
        """
        if self.ground_truth is None:
            return 0.0

        tool_calls = self._get_assistant_tool_calls(completion)
        if not tool_calls:
            return 0.0

        correct_calls = 0
        for tool_call in tool_calls:
            try:
                # Extract details from the agent's tool call
                called_name = tool_call["function"]["name"]
                called_args_str = tool_call["function"]["arguments"]
                called_args = json.loads(called_args_str)

                # Check if this call matches any of the ground truth calls
                if any(gt["tool_name"] == called_name and gt["tool_args"] == called_args for gt in self.ground_truth):
                    correct_calls += 1
            except (KeyError, json.JSONDecodeError, TypeError):
                continue

        return float(correct_calls)
