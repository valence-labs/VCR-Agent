"""
Improved LLM Client for Anthropic Vertex AI with native async support
"""

import os
import re
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for Anthropic Vertex AI client"""
    model: str = "claude-sonnet-4@20250514"
    max_tokens: int = 10000
    temperature: float = 0.1
    location: Optional[str] = None
    project_id: Optional[str] = None
    
    def __post_init__(self):
        """Load from environment variables if available"""
        self.location = self.location or os.getenv("VERTEX_AI_LOCATION")
        self.project_id = self.project_id or os.getenv("VERTEX_AI_PROJECT_ID")

class LLMClient:
    """Improved Anthropic Vertex AI client with native async support"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        # Initialize both sync and async clients
        client_kwargs = {
            "region": config.location,
            "project_id": config.project_id
        }
        
        self.sync_client = AnthropicVertex(**client_kwargs)
        self.async_client = AsyncAnthropicVertex(**client_kwargs)
        
        logger.info(f"Initialized Anthropic Vertex clients for {config.location}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response synchronously using native sync client"""
        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            **kwargs
        }
        
        try:
            response = self.sync_client.messages.create(messages=messages, **request_params)
            return response.content[0].text if response.content else ""
        except Exception as e:
            logger.error(f"Error in sync generation: {e}")
            raise

    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response asynchronously using native async client"""
        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            **kwargs
        }
        
        try:
            response = await self.async_client.messages.create(messages=messages, **request_params)
            return response.content[0].text if response.content else ""
        except Exception as e:
            logger.error(f"Error in async generation: {e}")
            raise

    async def agenerate_batch(self, batch_requests: List[Dict], max_concurrent: int = 5, **kwargs) -> List[str]:
        """Generate batch responses using concurrent async requests"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(request):
            async with semaphore:
                try:
                    messages = request.get("messages", [])
                    request_kwargs = {k: v for k, v in request.items() if k != "messages"}
                    request_kwargs.update(kwargs)
                    return await self.agenerate(messages, **request_kwargs)
                except Exception as e:
                    logger.error(f"Error in batch request: {e}")
                    return f"Error: {str(e)}"
        
        return await asyncio.gather(*[generate_single(req) for req in batch_requests])

    def count_tokens(self, text: str) -> int:
        """Estimate token count (4 chars ≈ 1 token for English)"""
        return len(text) // 4

    async def aclose(self):
        """Close the async client"""
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()

    def close(self):
        """Close the sync client"""
        if hasattr(self.sync_client, 'close'):
            self.sync_client.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.aclose()


def create_llm_client(model: Optional[str] = None, **kwargs) -> LLMClient:
    """Create improved LLM client"""
    config = LLMConfig(model=model or "claude-sonnet-4@20250514", **kwargs)
    return LLMClient(config) 
