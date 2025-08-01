import re
from typing import List, Dict, Any, Optional

from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.parsers.parser import Parser
from explain.llm._client import create_llm_client, LLMClient
from explain.eval.tools import ToolVerifier, REGISTERED_TOOLS
from explain.eval.utils import guess_max_turns


class FalsificationEvaluator:
    """
    Evaluator that uses an LLM agent to try and falsify a claim.
    """

    SYSTEM_PROMPT = """
You are an expert scientist and your task is to verify a scientific claim based on an original question. You have access to a set of tools to help you gather evidence.

**Original Question:**
{question}

**Claim to Verify:**
{claim}

You have access to the following tools:
{tool_descriptions}

Your task is to determine if the claim can be falsified by the available information.

1.  **Analyze the Claim**: Break down the claim into verifiable sub-claims.
2.  **Use Tools**: Call the necessary tools to gather evidence for each sub-claim. You may need to chain tool calls, using the output of one as the input for another.
3.  **Evaluate Evidence**: Analyze the evidence from all tool outputs collectively.
4.  **Conclude**: Based on all the evidence, provide a final verdict.

Provide your final verdict by choosing one of the following options:
(A) Consistent: Your reasoning and the tool outputs support the claim.
(B) Falsified: Your reasoning and the tool outputs contradict the claim.
(C) Inconclusive: There is insufficient evidence to either support or falsify the claim.

Your final response MUST be in the following format, with no other text.

<reason>
Your detailed reasoning for the verdict, citing the evidence you gathered.
</reason>
<answer>A|B|C</answer>
"""

    def __init__(self, llm_provider: str = "anthropic", max_turns: Optional[int] = 5, allowed_primitives: Optional[List[str]] = None, **kwargs):
        """
        Initializes the falsification evaluator.

        Args:
            llm_provider: The LLM provider to use ('anthropic', 'gemini', or 'openai').
            max_turns: The maximum number of conversational turns before stopping. If None, it will be set based on the claims predicates (primitives)
            **kwargs: Additional arguments for the LLM client.
        """
        self.llm_client: LLMClient = create_llm_client(provider=llm_provider, **kwargs)
        self.max_turns = max_turns
        self.allowed_primitives = allowed_primitives or []
        self.parser = Parser()

    def _get_llm_tools_schema(self) -> List[Dict[str, Any]]:
        """Formats tools into a schema compatible with the LLM."""
        return [tool.get_schema() for tool in REGISTERED_TOOLS]

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single tool call using the central tool registry."""
        tool_id = tool_call.get("id")
        content = ToolVerifier.call_tool(tool_call)
        return {"role": "tool", "tool_call_id": tool_id, "content": content}

    def _parse_final_verdict(self, text: str) -> tuple[str, str]:
        """Parses the final verdict and reasoning from the agent's response."""
        reason_match = re.search(r"<reason>([\s\S]*)</reason>", text, re.IGNORECASE)
        answer_match = re.search(r"<answer>\s*([ABC])\s*</answer>", text, re.IGNORECASE)
        reasoning = reason_match.group(1).strip() if reason_match else text
        verdict_map = {"A": "consistent", "B": "falsified", "C": "inconclusive"}
        verdict_code = answer_match.group(1).upper() if answer_match else "C"
        verdict = verdict_map.get(verdict_code, "inconclusive")
        return verdict, reasoning

    def _evaluate_falsification(self, prompt: Any, completion: str, answer: str, state: Dict[str, Any], **kwargs):
        """
        Runs the LLM agent to evaluate the claim and stores results in the state.
        `prompt` is the question, `completion` is the claim to be verified.
        """
        
        if "falsification_score" in state:
            return state

        if isinstance(prompt, list):
            question = prompt[-1]["content"]
        else:
            question = prompt
        
        claim = self.parser.parse_answer(completion)

        tool_descriptions = "\n".join(
            [f"- `{tool.name}`: {tool.description}" for tool in REGISTERED_TOOLS]
        )
        
        system_prompt = self.SYSTEM_PROMPT.format(
            question=question, claim=claim, tool_descriptions=tool_descriptions
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please proceed with the verification."},
        ]
        conversation_history = list(messages)
        final_response_text = ""
        max_turns = self.max_turns

        if max_turns is None:
            max_turns = guess_max_turns(claim, self.allowed_primitives)

        for _ in range(max_turns):
            response = self.llm_client.generate(messages, tools=self._get_llm_tools_schema())
            assistant_message = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_message["tool_calls"] = response.tool_calls
            
            messages.append(assistant_message)
            conversation_history.append(assistant_message)

            if not response.tool_calls:
                final_response_text = response.content or ""
                break

            tool_outputs = [self._execute_tool_call(tc) for tc in response.tool_calls]
            messages.extend(tool_outputs)
            conversation_history.extend(tool_outputs)
        else:
            final_response_text = "Agent reached maximum turns without reaching a conclusion."

        verdict, reasoning = self._parse_final_verdict(final_response_text)
        score = {"consistent": 1.0, "falsified": 0.0, "inconclusive": 0.5}.get(verdict, 0.5)

        state["falsification_score"] = score
        state["falsification_verdict"] = verdict
        state["falsification_reasoning"] = reasoning
        state["falsification_conversation_history"] = conversation_history
        
        return state

    def falsification_score(self, prompt: Any, completion: str, answer: str, state: Dict[str, Any], **kwargs) -> float:
        """Reward function for falsification."""
        state = self._evaluate_falsification(prompt, completion, answer, state, **kwargs)
        return float(state.get("falsification_score", 0.0))


class FalsificationRubric(JudgeRubric):
    """
    A rubric that uses a FalsificationEvaluator to assess a claim's validity.
    
    The final reward is based on whether the claim is consistent (1.0),
    falsified (0.0), or inconclusive (0.5) after an agentic investigation.
    """
    
    def __init__(self, llm_provider: str = "anthropic", **kwargs):
        """
        Initializes the plausibility rubric.
        
        Args:
            llm_provider: The LLM provider to use ('anthropic', 'gemini', or 'openai').
            **kwargs: Additional arguments for the JudgeRubric.
        """
        super().__init__(judge_prompt="", parallelize_scoring=False, **kwargs)
        
        evaluator = FalsificationEvaluator(llm_provider=llm_provider, **kwargs)
        self.judge_client = evaluator.llm_client
        self.judge_model = evaluator.llm_client.config.model

        self.add_reward_func(evaluator.falsification_score, weight=1.0) 