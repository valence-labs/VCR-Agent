"""Biological hypothesis verification environment using verifiers framework."""

from typing import List, Dict, Any, Optional
from datasets import Dataset
from verifiers.envs.tool_env import ToolEnv
from verifiers.rubrics.rubric_group import RubricGroup

from explain.tools.bio import BIOLOGICAL_TOOLS
from explain.evaluation.rubrics import BiologicalExplanationRubricGroup


def create_biological_verification_env(
    dataset: Optional[Dataset] = None,
    tools: Optional[List] = None,
    rubrics: Optional[RubricGroup] = None,
    max_turns: int = 10,
    **kwargs
) -> ToolEnv:
    """Create a biological hypothesis verification environment.
    
    This function sets up a verifiers ToolEnv configured specifically for
    biological hypothesis verification tasks with appropriate tools and rubrics.
    
    Args:
        dataset: HuggingFace dataset with biological hypotheses to verify
        tools: List of verification tools (defaults to BIOLOGICAL_TOOLS)
        rubrics: Evaluation rubrics (defaults to BiologicalExplanationRubricGroup)
        max_turns: Maximum number of interaction turns
        **kwargs: Additional arguments passed to ToolEnv
        
    Returns:
        ToolEnv: Configured environment for biological verification
    """
    # Use default biological tools if none provided
    if tools is None:
        tools = BIOLOGICAL_TOOLS
    
    # Use default biological rubrics if none provided  
    if rubrics is None:
        rubrics = BiologicalExplanationRubricGroup()
    
    # Create the verifiers ToolEnv
    env = ToolEnv(
        tools=tools,
        max_turns=max_turns,
        **kwargs
    )
    
    # Add our custom rubrics for biological evaluation
    env.rubrics = rubrics
    
    return env