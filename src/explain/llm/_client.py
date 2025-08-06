"""
Multi-provider LLM Client supporting Anthropic Vertex, Google Gemini, and OpenAI
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from explain.eval.tools._base import ToolVerifier

load_dotenv()


@dataclass
class LLMResponse:
    """Standardized response from an LLM call."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]] | None = None  # NEW: Full message history

    def to_dict(self):
        """Convert response to a dictionary."""
        return asdict(self)


@dataclass
class LLMConfig:
    """Configuration for LLM clients"""

    provider: str = "litellm"  # litellm, anthropic, gemini, openai
    model: str = "claude-sonnet-4@20250514"
    max_tokens: int = 10000
    temperature: float = 0.1
    location: str | None = None
    project_id: str | None = None

    def __post_init__(self):
        """Load from environment variables if available"""
        self.location = self.location or os.getenv("VERTEXAI_LOCATION", "global")
        self.project_id = self.project_id or os.getenv("VERTEXAI_PROJECT")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously."""
        pass

    @abstractmethod
    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        pass

    @abstractmethod
    def format_tool_response(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tool responses for the next turn."""
        pass

    def _format_tools(self, tools: list[Any]):
        """Format tools to the provider's format."""
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, ToolVerifier):
                tool_schema = tool.get_schema()
            else:
                tool_schema = tool
            formatted_tools.append(tool_schema)
        return formatted_tools

    def _get_message_historic(
        self, messages: list[dict[str, Any]], content: str | None, tool_calls: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """
        Construct updated message history from input messages and LLM outputs.
        Each subclass can override this if needed.
        """
        assistant_msg = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        return messages + [assistant_msg]


class AnthropicVertexClient(BaseLLMClient):
    """Client for Anthropic's models on Vertex AI."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.sync_client = None
        self.async_client = None
        try:
            from anthropic import AnthropicVertex

            self.sync_client = AnthropicVertex(project_id=config.project_id, region=config.location)
            from anthropic import AsyncAnthropicVertex

            self.async_client = AsyncAnthropicVertex(project_id=config.project_id, region=config.location)
        except ImportError as e:
            raise ImportError("anthropic package required for Anthropic Vertex client") from e

    def _format_tools(self, tools: list[Any]):
        """
        Converts generic tool schemas to Anthropic's format by renaming 'parameters' to 'input_schema'.
        """
        anthropic_tools = []
        tools = super()._format_tools(tools)
        for tool in tools:
            anthropic_tools.append(
                {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
            )
        return anthropic_tools

    def format_tool_response(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Formats tool responses for Anthropic.

        The response should be a list containing a single user message.
        This message, in turn, contains a list of tool_result content blocks.
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": output["tool_call_id"],
                        "content": output["content"],
                    }
                    for output in tool_outputs
                ],
            }
        ]

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
            **kwargs,
        }

        try:
            response = self.sync_client.messages.create(**request_params)
            content_text = ""
            tool_calls = []
            assistant_content = []

            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.input,
                            },
                        }
                    )
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            if content_text:
                assistant_content.insert(0, {"type": "text", "text": content_text})

            full_messages = self._get_message_historic(messages, content_text, tool_calls)
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"Anthropic Vertex generation failed: {e}")
            raise

    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
            **kwargs,
        }

        try:
            response = await self.async_client.messages.create(**request_params)
            content_text = ""
            tool_calls = []
            assistant_content = []

            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.input,
                            },
                        }
                    )
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            if content_text:
                assistant_content.insert(0, {"type": "text", "text": content_text})

            full_messages = self._get_message_historic(messages, content_text, tool_calls)
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"Anthropic Vertex async generation failed: {e}")
            raise

    def _get_message_historic(self, messages, content, tool_calls):
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        if tool_calls:
            for tool_call in tool_calls:
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": args,
                    }
                )
        return messages + [{"role": "assistant", "content": blocks}]


class GeminiClient(BaseLLMClient):
    """Client for Google's Gemini models."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from google import genai

            self.client = genai.Client(vertexai=True, project=config.project_id, location=config.location)

            # Default Gemini models
            if not config.model:
                self.config.model = "gemini-2.5-flash"

            logger.info(f"Initialized Google Gemini client with model {self.config.model}")

        except ImportError as e:
            raise ImportError("google-genai package required for Gemini client") from e

    def _messages_to_content(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Converts a list of messages to Gemini's content format."""
        from google.genai import types

        gemini_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                role = "model"

            parts = []

            if role == "tool":
                parts.append(
                    types.Part.from_function_response(name=msg["tool_call_id"], response=json.loads(msg["content"]))
                )
            else:  # user or assistant/model
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))

                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        args = {}
                        if tool_call.get("function", {}).get("arguments"):
                            args = json.loads(tool_call["function"]["arguments"])

                        parts.append(types.Part.from_function_call(name=tool_call["function"]["name"], args=args))

            if parts:
                gemini_messages.append(types.Content(role=role, parts=parts))
        return gemini_messages

    def _format_tools(self, tools: list[Any]):
        """Converts generic tool schemas to the Gemini SDK's `Tool` format."""
        from google.genai import types

        tools = super()._format_tools(tools)
        gemini_tools = []
        for tool_dict in tools:
            func_decl = types.FunctionDeclaration(
                name=tool_dict["name"],
                description=tool_dict["description"],
                parameters=types.Schema(**tool_dict["parameters"]),
            )
            gemini_tools.append(types.Tool(function_declarations=[func_decl]))
        return gemini_tools

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        from google.genai import types

        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        system_instruction = None
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0].get("content")
            messages = messages[1:]

        contents = self._messages_to_content(messages)

        config_params = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            **kwargs,
        }
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(**config_params)

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            )
            content_text = response.text
            tool_calls = []
            if response.function_calls:
                for func_call in response.function_calls:
                    tool_calls.append(
                        {
                            "id": func_call.name,
                            "type": "function",
                            "function": {
                                "name": func_call.name,
                                "arguments": json.dumps(dict(func_call.args)),
                            },
                        }
                    )

            full_messages = self._get_message_historic(messages, content_text, tool_calls)
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        from google.genai import types

        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        system_instruction = None
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0].get("content")
            messages = messages[1:]

        contents = self._messages_to_content(messages)

        config_params = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            **kwargs,
        }
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(**config_params)

        try:
            response = await self.client.aio.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            )
            content_text = response.text
            tool_calls = []
            if response.function_calls:
                for func_call in response.function_calls:
                    tool_calls.append(
                        {
                            "id": func_call.name,
                            "type": "function",
                            "function": {
                                "name": func_call.name,
                                "arguments": json.dumps(dict(func_call.args)),
                            },
                        }
                    )

            full_messages = self._get_message_historic(messages, content_text, tool_calls)
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"Gemini async generation failed: {e}")
            raise

    def format_tool_response(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Formats tool responses for Gemini.

        The response should be a list of tool messages, one for each tool output.
        """
        return [
            {
                "role": "tool",
                "tool_call_id": output["tool_call_id"],
                "name": output["name"],
                "content": output["content"],
            }
            for output in tool_outputs
        ]


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI's models."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        try:
            from openai import AsyncOpenAI, OpenAI

            from explain.llm._access_token import set_env_secrets

            # Set up environment secrets
            set_env_secrets()

            self.sync_client = OpenAI()
            self.async_client = AsyncOpenAI()

            # Default OpenAI models
            if not config.model or config.model.startswith("claude") or config.model.startswith("gemini"):
                self.config.model = "gpt-4.1"

            logger.info(f"Initialized OpenAI client with model {self.config.model}")

        except ImportError as e:
            raise ImportError("openai package and explain.llm._access_token required for OpenAI client") from e

    def _format_tools(self, tools: list[Any]):
        """Wraps generic tool schemas in the format required by the OpenAI API."""
        tools = super()._format_tools(tools)
        return [{"type": "function", "function": tool} for tool in tools]

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        try:
            response = self.sync_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

            full_messages = self._get_message_historic(messages, message.content, tool_calls)
            return LLMResponse(content=message.content, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

            full_messages = self._get_message_historic(messages, message.content, tool_calls)
            return LLMResponse(content=message.content, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            raise

    def format_tool_response(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Formats tool responses for OpenAI.

        The response should be a list of tool messages, one for each tool output.
        """
        return [
            {
                "role": "tool",
                "tool_call_id": output["tool_call_id"],
                "name": output["name"],
                "content": output["content"],
            }
            for output in tool_outputs
        ]


class LiteLLMClient(BaseLLMClient):
    """Client for LiteLLM."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError("litellm package required for LiteLLM client. Please run `pip install litellm`") from e

        logger.info(f"Initialized LiteLLM client with model {self.config.model}")

    def _format_tools(self, tools: list[Any]):
        """Wraps generic tool schemas in the format required by LiteLLM (same as OpenAI)."""
        tools = super()._format_tools(tools)
        return [{"type": "function", "function": tool} for tool in tools]

    def __get_model_name(self) -> str:
        """Get the model name."""
        if self.config.model.startswith("claude") or self.config.model.startswith("gemini"):
            return f"vertex_ai/{self.config.model}"
        return self.config.model

    def generate(self, messages: list[dict[str, Any]], **kwargs) -> LLMResponse:
        """Generate response synchronously using litellm."""
        import litellm

        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        try:
            response = litellm.completion(
                model=self.__get_model_name(),
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

            full_messages = self._get_message_historic(messages, message.content, tool_calls)
            return LLMResponse(content=message.content, tool_calls=tool_calls or None, messages=full_messages)

        except Exception as e:
            logger.error(f"LiteLLM generation failed: {e}")
            raise

    async def agenerate(self, messages: list[dict[str, Any]], **kwargs) -> LLMResponse:
        """Generate response asynchronously using litellm."""
        import litellm

        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])
        try:
            response = await litellm.acompletion(
                model=self.__get_model_name(),
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
            full_messages = self._get_message_historic(messages, message.content, tool_calls)
            return LLMResponse(content=message.content, tool_calls=tool_calls or None, messages=full_messages)
        except Exception as e:
            logger.error(f"LiteLLM async generation failed: {e}")
            raise

    def format_tool_response(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Formats tool responses for LiteLLM (same as OpenAI).
        """
        return [
            {
                "role": "tool",
                "tool_call_id": output["tool_call_id"],
                "name": output["name"],
                "content": output["content"],
            }
            for output in tool_outputs
        ]


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

        # Initialize the appropriate client based on provider
        if config.provider == "anthropic":
            self.client = AnthropicVertexClient(config)
        elif config.provider == "gemini":
            self.client = GeminiClient(config)
        elif config.provider == "openai":
            self.client = OpenAIClient(config)
        elif config.provider == "litellm":
            self.client = LiteLLMClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        logger.info(f"Initialized unified LLM client with provider: {config.provider}")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously."""
        return self.client.generate(messages, **kwargs)

    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        return await self.client.agenerate(messages, **kwargs)

    def format_tool_response(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tool responses for the next turn."""
        return self.client.format_tool_response(tool_outputs)

    async def generate_async(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        return await self.client.agenerate(messages, **kwargs)
