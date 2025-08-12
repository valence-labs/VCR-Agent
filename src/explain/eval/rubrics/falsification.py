import re
from typing import Any

from verifiers.parsers.parser import Parser
from verifiers.rubrics.judge_rubric import JudgeRubric

from explain.eval.tools import REGISTERED_TOOLS, ToolVerifier
from explain.eval.utils import guess_max_turns
from explain.llm import LLMClient, create_client


class FalsificationEvaluator:
    """
    Evaluator that uses an LLM agent to try and falsify a claim.
    """

    SYSTEM_PROMPT = """
You are an expert scientist tasked with verifying a scientific claim in response to a research question.

Each claim consists of:
- A **narrative explanation** (free-text paragraph) in the <answer> block, and
- A **structured explanation block** enclosed in <explain> ... </explain>, containing one or more mechanistic or associative actions (e.g., `binds_to`, `modulates_activity`, `regulates_expression`, etc.),
- A **DAG** enclosed in <dag> ... </dag>, representing the relationship between the primitives statements in the <explain> block.

You have access to the following tools to assist your analysis. But not all tools are relevant to the claim.
{tool_descriptions}

---

**Original Question:**
{question}

**Claim to Verify:**
{claim}

---

### Your task

1. **Decompose the Claim**
   Identify each individual sub-claim contained in the <explain> block. Treat each line (e.g., `binds_to(A, B, ...)`) as a separate hypothesis to verify.

2. **Use the Tools**
   Query the tools as needed to collect supporting or contradicting evidence for each sub-claim.
   You may chain tool outputs when necessary (e.g., retrieve structure → query binding).
   If tools yield no result, apply expert reasoning using your scientific knowledge and plausible biological inference.

3. **Evaluate the Evidence**
   For each sub-claim, determine whether the evidence:
   - **Supports** the sub-claim
   - **Contradicts** the sub-claim
   - Is **inconclusive or unavailable**

Only consider a sub-claim **contradicted** if there is **strong, direct, and specific** evidence **clearly contradicting** it.
If no contradiction is found, but evidence is sparse, or the sub-claim relies on plausible inference, consider it **inconclusive**.

4. **Conclude with a Final Verdict**
   Choose one of the following outcomes:

   - (A) **Consistent** – All sub-claims are supported or reasonably plausible. No contradiction found.
   - (B) **Falsified** – One or more **important** sub-claims are **clearly contradicted** by the evidence.
   - (C) **Inconclusive** – One or more important sub-claims lack sufficient evidence, or rely on assumptions that cannot be verified or rejected.

---

### Output Format

You must follow this format exactly — no additional text outside the tags.

<reason>
Your detailed reasoning here:
- List which sub-claims were supported, falsified, or lacked data
- Cite specific tool outputs (e.g., ToolName#CallID) when relevant
</reason>
<answer>A|B|C</answer>
"""

    def __init__(
        self,
        llm_provider: str = "litellm",
        max_turns: int | None = 5,
        allowed_primitives: list[str] | None = None,
        tools: list[ToolVerifier] | None = None,
        **kwargs,
    ):
        """
        Initializes the falsification evaluator.

        Args:
            llm_provider: The LLM provider to use ('litellm', 'anthropic', 'gemini', or 'openai').
            max_turns: The maximum number of conversational turns before stopping. If None, it will be set based on the claims predicates (primitives)
            allowed_primitives: The primitives to use for the falsification. If None, all primitives will be used.
            tools: The tools to use for the falsification. If None, all tools will be used.
            **kwargs: Additional arguments for the LLM client.
        """
        self.llm_client: LLMClient = create_client(provider=llm_provider, **kwargs)
        self.max_turns = max_turns
        self.allowed_primitives = allowed_primitives or []
        self.parser = Parser()
        self.tools = tools or list(REGISTERED_TOOLS.values())

    def _parse_final_verdict(self, text: str) -> tuple[str, str]:
        """Parses the final verdict and reasoning from the agent's response."""
        reason_match = re.search(r"<reason>([\s\S]*)</reason>", text, re.IGNORECASE)
        answer_match = re.search(r"<answer>\s*([ABC])\s*</answer>", text, re.IGNORECASE)
        reasoning = reason_match.group(1).strip() if reason_match else text
        verdict_map = {"A": "consistent", "B": "falsified", "C": "inconclusive"}
        verdict_code = answer_match.group(1).upper() if answer_match else "C"
        verdict = verdict_map.get(verdict_code, "inconclusive")
        return verdict, reasoning

    def _evaluate(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs):
        """
        Runs the LLM agent to evaluate the claim and stores results in the state.
        `prompt` is the question, `completion` is the claim to be verified.
        """

        # Avoid re-computing if already evaluated (following plausibility.py pattern)
        if "judge_response" in state:
            return state

        if isinstance(prompt, list):
            question = prompt[-1]["content"]
        else:
            question = prompt

        claim = self.parser.parse_answer(completion)

        tool_descriptions = "\n".join([f"- `{tool.name}`: {tool.description}" for tool in REGISTERED_TOOLS.values()])

        system_prompt = self.SYSTEM_PROMPT.format(question=question, claim=claim, tool_descriptions=tool_descriptions)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please proceed with the verification."},
        ]
        final_response_text = ""
        max_turns = self.max_turns

        if max_turns is None:
            max_turns = guess_max_turns(claim, self.allowed_primitives)

        for turn in range(max_turns):
            # Generate response with tools
            response = self.llm_client.generate(messages, tools=self.tools)

            # Update messages with the response (client handles provider-specific formatting)
            messages = response.messages

            # Check if we have tool calls
            if response.tool_calls:
                # Execute tools and format responses
                tool_outputs = []
                for tool_call in response.tool_calls:
                    try:
                        # Extract tool information
                        tool_name = "unknown"  # Default value
                        if isinstance(tool_call, dict):
                            if "function" in tool_call:
                                # OpenAI format
                                tool_name = tool_call["function"]["name"]
                                content = ToolVerifier.call_tool(tool_call["function"])
                            else:
                                # Direct format
                                tool_name = tool_call.get("name", tool_name)
                                content = ToolVerifier.call_tool(tool_call)
                        else:
                            content = ToolVerifier.call_tool(tool_call)

                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.get("id", f"call_{turn}_{len(tool_outputs)}"),
                                "name": tool_name,
                                "content": content,
                            }
                        )
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.get("id", f"call_{turn}_{len(tool_outputs)}"),
                                "name": tool_call.get("function", {}).get("name", "unknown")
                                if isinstance(tool_call, dict)
                                else "unknown",
                                "content": f"Error executing tool: {str(e)}",
                            }
                        )

                # Format tool responses using client-specific formatting
                tool_response_messages = self.llm_client.format_tool_response(tool_outputs)
                messages.extend(tool_response_messages)
            else:
                # No more tool calls, we have the final response
                final_response_text = response.content or ""
                break
        else:
            # Reached max turns without conclusion
            final_response_text = "Agent reached maximum turns without reaching a conclusion."

        verdict, reasoning = self._parse_final_verdict(final_response_text)

        score = {"consistent": 1.0, "falsified": 0.0, "inconclusive": 0.5}.get(verdict, 0.5)

        # Store results in state following plausibility.py pattern
        state["judge_response"] = response  # Store the final LLM response
        state["falsification_score"] = score
        state["falsification_verdict"] = verdict
        state["falsification_reasoning"] = reasoning
        state["falsification_conversation_history"] = messages  # Use messages directly from final response

        return state

    def falsification_score(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs) -> float:
        """Reward function for falsification."""
        state = self._evaluate(prompt, completion, answer, state, **kwargs)
        return float(state.get("falsification_score", 0.0))


class FalsificationRubric(JudgeRubric):
    """
    A rubric that uses a FalsificationEvaluator to assess a claim's validity.

    The final reward is based on whether the claim is consistent (1.0),
    falsified (0.0), or inconclusive (0.5) after an agentic investigation.
    """

    def __init__(self, llm_provider: str = "litellm", tools: list[ToolVerifier] | None = None, **kwargs):
        """
        Initializes the falsification rubric.

        Args:
            llm_provider: The LLM provider to use ('litellm', 'anthropic', 'gemini', or 'openai').
            tools: The tools to use for the falsification. If None, all tools will be used.
            **kwargs: Additional arguments for the JudgeRubric.
        """
        # Create the decoupled evaluator which contains the core logic
        self.evaluator = FalsificationEvaluator(llm_provider=llm_provider, tools=tools, **kwargs)

        # Initialize the base rubric with evaluator's client
        super().__init__(judge_prompt="", judge_client=self.evaluator.llm_client, parallelize_scoring=False, **kwargs)

        self.judge_model = self.evaluator.llm_client.config.model

    def judge(self, prompt: Any, completion: str, answer: str, state: dict[str, Any], **kwargs) -> dict[str, float]:
        """
        Judge method that returns a dictionary of scores, compatible with other rubrics.
        """
        return {"falsification_score": self.evaluator.falsification_score(prompt, completion, answer, state, **kwargs)}
