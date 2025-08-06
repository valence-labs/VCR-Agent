# Notebooks for LLM Client Testing

Notebook to test and verify the functionality of the unified LLM client located in `src/explain/llm/_client.py`.

## Purpose

The primary goal of these notebooks is to ensure that the `LLMClient` and its underlying provider-specific clients (Anthropic, OpenAI, Gemini, LiteLLM) behave as expected. This includes:

-   **Basic API Calls**: Verifying that simple, single-turn conversations work correctly across all supported providers.
-   **Tool Calling**: Testing the multi-turn, tool-calling functionality to ensure that tools are correctly invoked and their outputs are properly formatted and sent back to the model.
-   **Provider-Specific Formatting**: Ensuring that the client correctly handles the unique message and tool-calling formats required by each LLM provider.
-   **Debugging**: Providing an interactive environment to debug issues with API calls, response parsing, and message history management.
