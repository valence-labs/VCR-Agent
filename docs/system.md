# Biological Hypothesis Verification System

A comprehensive framework for verifying biological hypotheses using LLM-callable tools, designed for both evaluation and multi-turn reinforcement learning (RL) training. **Now supports multiple LLM providers, enhanced class-based tools with Pydantic validation, and a standardized reward/feedback output format**.

## 🎯 Overview

This system enables Large Language Models (LLMs) to systematically verify biological hypotheses through:

- **Enhanced class-based verification tools** with detailed experimental context (cell types, perturbations, timepoints)
- **Robust Pydantic input validation** for all tool parameters
- **Standardized JSON output** with `reward` and `feedback` keys for RL training
- **Sophisticated gene regulation analysis** supporting separate upregulated/downregulated gene lists
- **Multi-turn RL environment** compatible with GRPO finetuning using the `verifiers` framework
- **LangChain integration** with LangSmith observability for production deployment
- **Multi-provider LLM support** - works with Anthropic, Gemini, and OpenAI
- **Comprehensive evaluation rubrics** for biological explanation quality assessment

## 🚀 Key Features

### 1. Enhanced Class-Based Biological Tools
Sophisticated callable classes that verify specific biological claims with detailed experimental context:

```python
from explain.tools.biological import check_regulation_expression

# Enhanced regulation verification with Pydantic validation
result = check_regulation_expression(
    source_entity="p53",
    upregulated_genes=["CDKN1A", "BAX", "PUMA"],
    downregulated_genes=["MYC", "CCND1"],
    cell_type="HepG2",
    gkos=["MDM2"],  # Gene knockouts
    compounds=["nutlin-3a"],
    dose=10.0  # Dose in μM
)
# Returns detailed verification with fold changes, p-values, and context
```

**Enhanced Tool Features:**
- **`RegulationExpressionVerifier`**: Separate upregulated/downregulated gene lists, cell type context, gene knockouts (GKO), compound perturbations, and dose
- **`DrugTargetVerifier`**: Verifies drug-target interactions with interaction type and strength
- **Pydantic Validation**: All inputs are validated against `args_schema` for robustness
- **Standardized Output**: All tools return a JSON object with `reward` and `feedback` keys

### 2. Multi-Provider LLM Support
Unified interface supporting three major LLM providers:

```python
from explain.llm._client import create_llm_client

# Anthropic Vertex AI
anthropic_client = create_llm_client(provider="anthropic")

# Google Gemini  
gemini_client = create_llm_client(provider="gemini")

# OpenAI
openai_client = create_llm_client(provider="openai")

# All use the same interface
response = client.generate([{"role": "user", "content": "Your query"}])
```

**Supported Models:**
- **Anthropic**: `claude-sonnet-4@20250514` (default)
- **Gemini**: `gemini-2.5-flash` (default)
- **OpenAI**: `gpt-4.1` (default)

### 3. Verifiers Framework Integration
Multi-turn RL environment for GRPO training:

```python
from explain.environments.biological import create_biological_verification_env

env = create_biological_verification_env(
    tools=BIOLOGICAL_TOOLS,
    max_turns=10
)
# Ready for multi-turn RL training with standardized reward feedback
```

### 4. LangChain Compatibility
Seamless integration with LangChain agents and enhanced tool parameter guidance:

```python
from explain.tools.langchain_integration import BiologicalVerificationAgent

agent = BiologicalVerificationAgent(
    llm_provider="anthropic",  # or "gemini", "openai"
    enable_langsmith=True
)
result = agent.verify_hypothesis(
    "p53 activation leads to cell cycle arrest",
    context="DNA damage response in HepG2 cells"
)
```

### 5. Enhanced Multi-Faceted Evaluation
Comprehensive rubrics with provider-specific LLM evaluation:

```python
from explain.evaluation.evaluator import BiologicalHypothesisEvaluator

evaluator = BiologicalHypothesisEvaluator(
    backend="langchain",
    llm_provider="anthropic"  # or "gemini", "openai"
)
results = evaluator.evaluate_hypothesis("Your biological hypothesis")
```

## 📁 Architecture

```
src/explain/
├── llm/
│   └── _client.py                 # Multi-provider LLM client
├── tools/
│   ├── biological.py              # Enhanced class-based verification tools with Pydantic validation
│   └── langchain_integration.py   # LangChain/LangSmith integration
├── environments/
│   └── biological.py             # Verifiers ToolEnv setup
├── evaluation/
│   ├── rubrics.py               # Custom evaluation rubrics
│   └── evaluator.py             # Unified evaluation framework
└── README.md                    # This file
```

## 🛠 Usage Examples

### Enhanced Biological Tool Usage
```python
from explain.tools.biological import check_regulation_expression, check_drug_target_interaction

# Sophisticated gene regulation verification
regulation_result = check_regulation_expression(
    source_entity="doxorubicin",
    upregulated_genes=["TP53", "CDKN1A", "BAX"],
    downregulated_genes=["MYC", "CCND1", "BCL2"],
    cell_type="MCF7",
    compounds=["doxorubicin"],
    dose=1.0
)

# Enhanced drug-target verification with validation
drug_result = check_drug_target_interaction(
    drug="imatinib",
    target="BCR-ABL1",
    interaction_type="inhibitor",
    strength=0.05  # Strength in μM
)
```

### Multi-Provider Evaluation
```python
from explain.evaluation.evaluator import BiologicalHypothesisEvaluator

# Test with different LLM providers
for provider in ["anthropic", "gemini", "openai"]:
    evaluator = BiologicalHypothesisEvaluator(
        backend="langchain",
        llm_provider=provider
    )
    results = evaluator.evaluate_hypothesis(
        "p53 activation upregulates CDKN1A and downregulates MYC in response to DNA damage"
    )
    print(f"{provider}: {results['verification_results']['agent_type']}")
```

### LangChain Integration with Enhanced Parameters
```python
from explain.tools.langchain_integration import BiologicalVerificationAgent

# Create agent with detailed tool parameter guidance
agent = BiologicalVerificationAgent(
    llm_provider="anthropic",
    enable_langsmith=True
)

# The agent automatically understands enhanced parameters
result = agent.verify_hypothesis(
    "Nutlin-3a treatment upregulates p53 target genes in HepG2 cells",
    context="24h treatment, 10μM dose, MDM2 pathway inhibition"
)
```

## 🎓 Use Cases

### 1. Enhanced Hypothesis Falsification
- **Detailed Experimental Context**: Verify claims with specific cell types, perturbations
- **Separate Gene Lists**: Test upregulated and downregulated genes independently
- **Robust Validation**: Pydantic validation ensures high-quality inputs
- **Multi-Provider Validation**: Cross-validate findings across different LLM providers

### 2. Sophisticated RL Training  
- **Standardized Rewards**: All tools provide `reward` key for RL
- **Context-Aware Rewards**: Train models to consider experimental conditions
- **Enhanced Parameter Learning**: Learn to specify detailed experimental contexts
- **Provider-Specific Training**: Optimize for different LLM providers

### 3. Production Deployment with Rich Context
- **Detailed Tool Calls**: LangChain agents with comprehensive parameter usage
- **Quality Assurance**: Input validation prevents errors
- **Quality Assessment**: Enhanced rubrics that evaluate experimental rigor

### 4. Advanced Research & Benchmarking
- **Experimental Context Sensitivity**: Compare how context affects verification
- **Parameter Importance**: Assess which experimental details matter most
- **Provider Performance**: Benchmark biological reasoning across providers

## 🔧 Technical Details

### Enhanced Tool Architecture
- **Class-Based Design**: Tools as callable classes with `__call__` methods
- **Pydantic Validation**: All inputs validated with `args_schema`
- **Standardized Output**: All tools return `reward` and `feedback`
- **Rich Parameter Sets**: Detailed experimental context in Pydantic schemas
- **Context-Aware Scoring**: Confidence scores that reflect experimental detail
- **Structured Outputs**: Comprehensive JSON responses with detailed feedback

### Experimental Context Support
- **Cell Type Specificity**: HepG2, MCF7, K562, primary cells, etc.
- **Dose Information**: Concentrations in μM
- **Perturbation Tracking**: Gene knockouts (GKO) and compound treatments

### LangChain Enhanced Integration
- **Parameter Guidance**: Agents understand enhanced tool parameters
- **Context Prompting**: Automatic guidance for experimental detail specification
- **Rich Tool Descriptions**: Detailed parameter explanations for LLM agents

## 📝 Requirements

### Core Dependencies
- Python 3.8+
- `datasets` (for data handling)
- `pydantic` (for structured validation)
- `numpy` (for reward calculation)

### LLM Provider Dependencies
- **Anthropic**: `anthropic` (for Vertex AI)
- **Gemini**: `google-genai` (for Gemini API)
- **OpenAI**: `openai` (for OpenAI API)

### Framework Dependencies
- **Verifiers**: `verifiers` (for RL training)
- **LangChain**: `langchain` + `langchain-openai` (for production)

## 🚀 Getting Started

1. **Install core dependencies**:
   ```bash
   pip install datasets pydantic numpy
   ```

2. **Install LLM provider packages** (choose what you need):
   ```bash
   # Anthropic Vertex AI
   pip install anthropic
   
   # Google Gemini
   pip install google-genai
   
   # OpenAI
   pip install openai
   ```

3. **Install framework packages** (optional):
   ```bash
   pip install verifiers langchain langchain-openai
   ```

4. **Set up credentials**:
   ```bash
   # Anthropic Vertex AI
   export VERTEX_AI_LOCATION="us-east5"
   export VERTEX_AI_PROJECT_ID="your-project-id"
   
   # Google Gemini (if using)
   export GOOGLE_API_KEY="your-api-key"
   
   # OpenAI (if using) 
   export OPENAI_API_KEY="your-api-key"
   
   # LangSmith (optional)
   export LANGCHAIN_API_KEY="your-api-key"
   ```

5. **Run the demo**:
   ```bash
   jupyter notebook notebooks/biological-verification-demo.ipynb
   ```

## 🔬 Example: Enhanced Multi-Context Workflow

```python
# 1. Enhanced regulation verification with full experimental context
from explain.tools.biological import check_regulation_expression

regulation_result = check_regulation_expression(
    source_entity="p53",
    upregulated_genes=["CDKN1A", "BAX", "NOXA"],
    downregulated_genes=["MYC", "CCND1", "BCL2"],
    cell_type="HepG2",
    gkos=["MDM2"],  # MDM2 knockout to stabilize p53
    compounds=["nutlin-3a"],
    dose=10.0
)

# 2. Multi-provider evaluation with enhanced context
from explain.evaluation.evaluator import BiologicalHypothesisEvaluator

evaluators = {}
for provider in ["anthropic", "gemini", "openai"]:
    evaluators[provider] = BiologicalHypothesisEvaluator(
        backend="langchain",
        llm_provider=provider
    )

# 3. Evaluate hypothesis with rich experimental detail
hypothesis = """p53 activation in MDM2-knockout HepG2 cells treated with 10μM nutlin-3a 
upregulates CDKN1A, BAX, and NOXA while downregulating MYC, CCND1, and BCL2"""

results = {}
for provider, evaluator in evaluators.items():
    results[provider] = evaluator.evaluate_hypothesis(
        hypothesis, 
        context="DNA damage response with MDM2 pathway inhibition"
    )

# 4. Compare enhanced verification across providers
for provider in results:
    print(f"\n{provider.upper()} EVALUATION:")
    print(f"Claims parsed: {sum(len(c) for c in results[provider]['parsed_claims'].values())}")
    print(f"Verification status: {results[provider].get('verification_results', {}).get('agent_type', 'N/A')}")
```

## 🤝 Contributing

The enhanced multi-provider framework is designed to be extensible:

1. **Add experimental contexts**: Extend tool parameters for new experimental conditions
2. **New LLM providers**: Implement `BaseLLMClient` interface
3. **Enhanced rubrics**: Create context-aware evaluation criteria
4. **Assay-specific tools**: Add new tool classes for specific experimental methods

## 📄 License

This enhanced biological verification system is part of the broader explain package for scientific hypothesis verification and evaluation with multi-provider LLM support and detailed experimental context. 