# Hooke-Explain: Structured Biomedical Explanation Generation

A comprehensive module for generating structured biomedical explanations from reports using Anthropic Vertex AI and managing enhanced chat interactions.

## Overview

This module provides tools for:
1. **Structured Explanation Generation**: Convert detailed biomedical reports into structured explanations using predefined action primitives
2. **Batch Processing**: Process multiple question-report pairs efficiently with DataFrame output
3. **Enhanced Chat Management**: Handle bulk report generation through enhanced chat APIs
4. **Large Report Processing**: Intelligently handle large markdown reports through chunking and summarization

## Installation

```bash
pip install pandas tqdm anthropic
```

## Module Structure

```
hooke-explain/
├── src/
│   ├── __init__.py              # Module initialization
│   ├── llm_client.py            # Anthropic Vertex AI client wrapper
│   ├── response_parser.py       # Parse structured LLM responses
│   ├── report_processor.py      # Handle large markdown reports
│   ├── structure_explainer.py   # Main structured explanation class
│   └── enhanced_chat_manager.py # Enhanced chat bulk operations
├── templates/
│   └── structure-explain.txt    # Prompt template for structured explanations
├── data/                        # Data directory for inputs/outputs
├── action_primitives.json       # Action primitive definitions
├── example_usage.py            # Usage examples
└── README.md                   # This file
```

## Quick Start

### 1. Structured Explanation Generation

```python
from src import create_structure_explainer

explainer = create_structure_explainer(
    model="claude-sonnet-4@20250514",
    location="us-east5",
    project_id="your-project-id",
    max_tokens=4000,
    temperature=0.1
)

question = "How does PARP inhibition affect BRCA1-deficient cells?"
report = """
# PARP Inhibition in BRCA1-Deficient Cells

PARP inhibitors cause synthetic lethality in BRCA1-deficient cells...
"""

parsed_response = explainer.generate_explanation(
    question=question,
    report=report,
    validate_response=True
)

print("Thinking:", parsed_response.thinking)
print("Answer:", parsed_response.answer)
print("Explain:", parsed_response.explain)
```

### 2. Batch Processing

```python
import pandas as pd

data = [
    {
        "question": "How does EGFR inhibition affect signaling?",
        "report": "# EGFR Analysis\n\nEGFR inhibition blocks...",
        "id": "sample_1"
    },
]

results_df = explainer.process_batch(
    data=data,
    question_column="question",
    report_column="report",
    id_column="id",
    show_progress=True
)

results_df.to_parquet("results.parquet", index=False)

stats = explainer.get_statistics(results_df)
print(f"Success rate: {stats['success_rate']:.1%}")
```

### 3. Enhanced Chat Management

```python
from src import EnhancedChatManager, ChatConfiguration, Model, OutputStyle
from enhanced_chat_client import AuthenticatedClient

client = AuthenticatedClient(base_url="...", token="...")

chat_manager = EnhancedChatManager(client)

chat_manager.update_config(
    model=Model.CLAUDE_V3_7_SONNET,
    pubmed_enabled=True,
    max_pubmed_to_read=5,
    output_style=OutputStyle.DETAILED
)

result = await chat_manager.submit_single_and_collect(
    template="Analyze this perturbation: {perturbation}",
    data={"perturbation": "BRCA1 knockout"},
    job_tag="brca1-analysis",
    tags=["biomedical", "cancer"],
    timeout=300
)

if result.success:
    print(f"Completed in {result.duration:.1f}s")
    conversations = result.conversations
    messages = chat_manager.extract_messages(conversations)

prompts = chat_manager.create_prompts(
    templates=[template],
    data=[{"perturbation": "p1"}, {"perturbation": "p2"}],
    tags=["analysis"]
)

await chat_manager.submit_bulk_job(prompts, "bulk-job-v1")

status = await chat_manager.check_job_status("bulk-job-v1")
print(f"Progress: {status['completed']}/{status['total']}")

result = await chat_manager.collect_results("bulk-job-v1")

summary = chat_manager.get_job_status_summary()
print(f"Active: {summary['total_active']}, Completed: {summary['total_completed']}")
```

## Core Components

### 1. LLMClient

Anthropic Vertex AI client wrapper:

```python
from src import LLMClient, LLMConfig, create_llm_client

client = create_llm_client(
    model="claude-sonnet-4@20250514",
    location="us-east5",
    project_id="your-project-id"
)

response = client.generate([{"role": "user", "content": "Hello"}])
```

### 2. ResponseParser

Extract structured sections from LLM responses:

```python
from src import ResponseParser

parser = ResponseParser()
parsed = parser.parse_response(llm_response)

print(parsed.thinking)  # <think>...</think> content
print(parsed.answer)    # <answer>...</answer> content
print(parsed.explain)   # <explain>...</explain> content

is_valid = parser.validate_response(parsed)
```

### 3. ReportProcessor

Handle large markdown reports:

```python
from src import ReportProcessor

processor = ReportProcessor(max_tokens_per_section=2000)

processed = processor.process_report(
    markdown_text,
    strategy="auto"  # "chunk", "summarize", or "auto"
)

condensed = processor.create_condensed_report(processed)
```

## Action Primitives

The system uses predefined action primitives for structured explanations:

- **Causal primitives**: `binds_to`, `modulates_activity`, `regulates_expression`, `causes_phenotype`
- **Associative primitives**: `correlates_with`, `similar_to`
- **Context primitives**: `set_context`, `participates_in`

See `action_primitives.json` for the complete list with descriptions and signatures.

## File Formats

### Input Data

**CSV/JSON format for questions:**
```csv
id,question,report_file
1,"How does X affect Y?","report_1.md"
2,"What is the mechanism of Z?","report_2.md"
```

**Markdown reports:**
```markdown
# Report Title

## Background
...

## Key Findings
- Finding 1
- Finding 2

## Conclusions
...
```

### Output Format

**DataFrame with structured results:**
```python
results_df.columns = [
    'index', 'question', 'thinking', 'answer', 'explain', 
    'raw_response', 'success', 'error', 'id'
]
```

## Configuration

### LLM Configuration

```python
explainer = create_structure_explainer(
    model="claude-sonnet-4@20250514",
    location="us-east5", 
    project_id="your-project-id",
    max_tokens=4000,
    temperature=0.1
)
```

### Enhanced Chat Configuration

```python
config = ChatConfiguration(
    model=Model.CLAUDE_V3_7_SONNET,
    pubmed_enabled=True,
    max_pubmed_to_read=5,
    fulltext_enabled=True,
    max_fulltext_to_read=3,
    output_style=OutputStyle.DETAILED,
    is_private=False
)
```

## Best Practices

1. **Report Processing**: Use "auto" strategy for mixed-size reports, "chunk" for consistency, "summarize" for large reports
2. **Validation**: Always enable response validation for production use
3. **Error Handling**: Check the `success` column in batch results and handle failed items
4. **Rate Limiting**: For large batches, consider processing in smaller chunks
5. **Token Management**: Monitor token usage, especially with large reports

## License

This module is part of the cb-reach project. See the main project license for details. 