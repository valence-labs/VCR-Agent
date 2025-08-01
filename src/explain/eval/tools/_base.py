import inspect
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolVerifier(ABC):
    """Base class for tool verifiers."""

    _registry: dict[str, "ToolVerifier"] = {}

    name: str
    description: str
    args_schema: BaseModel

    def __init_subclass__(cls, **kwargs):
        """This method is called when a new class inherits from ToolVerifier,
        automating the registration process."""
        super().__init_subclass__(**kwargs)
        # We instantiate and register the class, but only if it's a concrete
        # subclass (not an abstract class).
        if not inspect.isabstract(cls):
            instance = cls()
            cls.register_tool(instance)

    def __call__(self, **kwargs) -> str:
        """
        Validates input arguments against the tool's args_schema and executes the tool logic.
        """
        try:
            validated_args = self.args_schema(**kwargs)
        except ValidationError as e:
            return json.dumps({"reward": 0.0, "feedback": f"Input Validation Error: {e}"}, indent=2)

        reward, feedback = self._tool_logic(validated_args)

        result = {"reward": reward, "feedback": feedback}

        return json.dumps(result, indent=2)

    @abstractmethod
    def _tool_logic(self, args: BaseModel) -> tuple[float, dict[str, Any]]:
        """
        Core tool logic.

        Args:
            args: Validated Pydantic model of arguments.

        Returns:
            A tuple containing the reward (float) and feedback (dict).
        """
        raise NotImplementedError

    @classmethod
    def register_tool(cls, tool: "ToolVerifier"):
        """Registers a tool instance in the central registry."""
        if not isinstance(tool, ToolVerifier):
            raise TypeError("Can only register instances of ToolVerifier")
        cls._registry[tool.name] = tool

    @classmethod
    def call_tool(cls, tool_call: dict[str, Any]) -> str:
        """
        Calls a registered tool by name using the tool call request from an LLM.

        Args:
            tool_call: A dictionary representing the tool call from an LLM.
                       It should have a 'function' key with 'name' and 'arguments'.

        Returns:
            A JSON string with the result of the tool execution.
        """
        try:
            tool_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]

            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            elif isinstance(arguments_str, dict):
                arguments = arguments_str
            else:
                raise TypeError(f"Unexpected type for arguments: {type(arguments_str)}")

        except (KeyError, json.JSONDecodeError, TypeError) as e:
            return json.dumps({"error": f"Invalid tool call format: {e}"})

        if tool_name not in cls._registry:
            return json.dumps({"error": f"Tool '{tool_name}' not found in registry."})

        tool_instance = cls._registry[tool_name]
        try:
            return tool_instance(**arguments)
        except Exception as e:
            return json.dumps({"error": f"Error executing tool '{tool_name}': {e}"})

    def get_schema(self) -> dict[str, Any]:
        """Converts a ToolVerifier instance to a generic schema dictionary."""
        return {"name": self.name, "description": self.description, "parameters": self.args_schema.model_json_schema()}
