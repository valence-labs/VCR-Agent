import os
from typing import Any

import dotenv
from rxrx.enhanced_chat.api.bulk_controller_v1 import (
    bulk_controller_v1_ner,
)
from rxrx.enhanced_chat.client import AuthenticatedClient

from explain.enhanced_chat._access_token import get_access_token

dotenv.load_dotenv()


NER_OPTIONS = ["GENE", "CHEMICAL", "DISEASE"]


class NER:
    """
    A tool that uses the NER service to verify if a gene is in the knowledge graph.

    Supports both synchronous and asynchronous operation.
    """

    EC_BASE_URL = os.getenv("EC_BASE_URL", "https://enhanced-chat.centaur-platform-dev.com/")

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = get_access_token()

        self.client = AuthenticatedClient(
            base_url=self.EC_BASE_URL, token=self.api_key, raise_on_unexpected_status=True
        )

    def __call__(self, text: str, options: list[str] = NER_OPTIONS) -> Any:
        """
        Synchronous call - wraps the async method.
        """
        return self.verify(text, options)

    async def __acall__(self, text: str) -> Any:
        """
        Asynchronous call.
        """
        return await self.averify(text)

    def verify(self, text: str, options: list[str] = NER_OPTIONS) -> Any:
        """
        Synchronous verification of text using NER service.

        Args:
            text: Text to analyze for named entities

        Returns:
            NER response from the service
        """
        if isinstance(options, str):
            options = [options]
        q = text
        if options is not None and len(options) > 0:
            options = [x for x in options if x in NER_OPTIONS]
            q = {"text": text, "options": options}
        # Use the sync version of the API call
        ner_response = bulk_controller_v1_ner.sync(client=self.client, q=q)
        return ner_response

    async def averify(self, text: str, options: list[str] = NER_OPTIONS) -> Any:
        """
        Asynchronous verification of text using NER service.

        Args:
            text: Text to analyze for named entities

        Returns:
            NER response from the service
        """
        if isinstance(options, str):
            options = [options]
        q = text
        if options is not None and len(options) > 0:
            options = [x for x in options if x in NER_OPTIONS]
            q = {"text": text, "options": options}
        # Use the sync version of the API call
        ner_response = await bulk_controller_v1_ner.asyncio(client=self.client, q=q)
        return ner_response
