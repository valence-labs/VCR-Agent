"""
Multi-provider LLM Client supporting Anthropic Vertex, Google Gemini, and OpenAI
"""

import asyncio
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import json

from dotenv import load_dotenv
from loguru import logger
from explain.eval.tools._base import ToolVerifier

load_dotenv()


@dataclass
class LLMResponse:
    """Standardized response from an LLM call."""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMConfig:
    """Configuration for LLM clients"""
    
    provider: str = "anthropic"  # anthropic, gemini, openai
    model: str = "claude-sonnet-4@20250514"
    max_tokens: int = 10000
    temperature: float = 0.1
    location: str | None = None
    project_id: str | None = None

    def __post_init__(self):
        """Load from environment variables if available"""
        self.location = self.location or os.getenv("VERTEX_AI_LOCATION", "us-east5")
        self.project_id = self.project_id or os.getenv("VERTEX_AI_PROJECT_ID", "vertexai-sandbox-e8a925d0")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously."""
        pass
    
    @abstractmethod
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        pass

 
    def _format_tools(self, tools: List[Any]):
        """Format tools to the provider's format."""
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, ToolVerifier):
                tool_schema = tool.get_schema()
            else:
                tool_schema = tool
            formatted_tools.append(tool_schema)
        return formatted_tools
    
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
        except ImportError:
            raise ImportError("anthropic package required for Anthropic Vertex client")
    
    def _format_tools(self, tools: List[Any]):
        """
        Converts generic tool schemas to Anthropic's format by renaming 'parameters' to 'input_schema'.
        """
        anthropic_tools = []
        tools = super()._format_tools(tools)
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            })
        return anthropic_tools

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously using Anthropic Vertex."""
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
            **kwargs
        }
        
        try:
            response = self.sync_client.messages.create(**request_params)
            
            content_text = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    })
            
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None)

        except Exception as e:
            logger.error(f"Anthropic Vertex generation failed: {e}")
            raise
    
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously using Anthropic Vertex."""
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])

        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
            **kwargs
        }
        
        try:
            response = await self.async_client.messages.create(**request_params)

            content_text = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    })

            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None)

        except Exception as e:
            logger.error(f"Anthropic Vertex async generation failed: {e}")
            raise


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
            
        except ImportError:
            raise ImportError("google-genai package required for Gemini client")

    def _messages_to_content(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Converts a list of messages to Gemini's content format."""
        from google.genai import types
        
        gemini_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                role = "model"

            parts = []
            
            if role == "tool":
                parts.append(types.Part.from_function_response(
                    name=msg["tool_call_id"],
                    response=json.loads(msg["content"])
                ))
            else: # user or assistant/model
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        args = {}
                        if tool_call.get("function", {}).get("arguments"):
                            args = json.loads(tool_call["function"]["arguments"])
                        
                        parts.append(types.Part.from_function_call(
                            name=tool_call["function"]["name"],
                            args=args
                        ))

            if parts:
                gemini_messages.append(types.Content(role=role, parts=parts))
        return gemini_messages

    def _format_tools(self, tools: List[Any]):
        """Converts generic tool schemas to the Gemini SDK's `Tool` format."""
        from google.genai import types
        tools = super()._format_tools(tools)
        gemini_tools = []
        for tool_dict in tools:
            func_decl = types.FunctionDeclaration(
                name=tool_dict["name"],
                description=tool_dict["description"],
                parameters=types.Schema(**tool_dict["parameters"])
            )
            gemini_tools.append(types.Tool(function_declarations=[func_decl]))
        return gemini_tools

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously using Gemini."""
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
            **kwargs
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
                    tool_calls.append({
                        "id": func_call.name,
                        "type": "function",
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(dict(func_call.args)),
                        },
                    })
            
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None)

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously using Gemini."""
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
            **kwargs
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
                    tool_calls.append({
                        "id": func_call.name,
                        "type": "function",
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(dict(func_call.args)),
                        },
                    })
            
            return LLMResponse(content=content_text or None, tool_calls=tool_calls or None)

        except Exception as e:
            logger.error(f"Gemini async generation failed: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI's models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        try:
            from openai import OpenAI, AsyncOpenAI
            from explain.llm._access_token import set_env_secrets
            
            # Set up environment secrets
            set_env_secrets()
            
            self.sync_client = OpenAI()
            self.async_client = AsyncOpenAI()
            
            # Default OpenAI models
            if not config.model or config.model.startswith("claude") or config.model.startswith("gemini"):
                self.config.model = "gpt-4.1"
            
            logger.info(f"Initialized OpenAI client with model {self.config.model}")
            
        except ImportError:
            raise ImportError("openai package and explain.llm._access_token required for OpenAI client")
    
    def _format_tools(self, tools: List[Any]):
        """Wraps generic tool schemas in the format required by the OpenAI API."""
        tools = super()._format_tools(tools)
        return [{"type": "function", "function": tool} for tool in tools]

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously using OpenAI."""
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])
        try:
            response = self.sync_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })

            return LLMResponse(content=message.content, tool_calls=tool_calls or None)

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously using OpenAI."""
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self._format_tools(kwargs["tools"])
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            message = response.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })
            return LLMResponse(content=message.content, tool_calls=tool_calls or None)
        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            raise


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
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        logger.info(f"Initialized unified LLM client with provider: {config.provider}")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response synchronously."""
        return self.client.generate(messages, **kwargs)
    
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        return await self.client.agenerate(messages, **kwargs)

    async def generate_async(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        return await self.client.agenerate(messages, **kwargs)


def create_llm_client(
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """Create an LLM client with the specified provider.
    
    Args:
        provider: LLM provider - 'anthropic', 'gemini', or 'openai'
        model: Model name (uses provider defaults if not specified)
        **kwargs: Additional configuration options
        
    Returns:
        LLMClient: Configured LLM client
    """
    # Set default models for each provider
    default_models = {
        "anthropic": "claude-sonnet-4@20250514",
        "gemini": "gemini-2.5-flash", 
        "openai": "gpt-4.1"
    }
    
    if model is None:
        model = default_models.get(provider, "claude-sonnet-4@20250514")
    
    config = LLMConfig(
        provider=provider,
        model=model,
        **kwargs
    )
    
    return LLMClient(config)