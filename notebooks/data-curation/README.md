# Data Curation Workflow

This folder contains the first data curation example demonstrating how to use enhanced chat to transform perturbation and context data into structured explanations and DAG

## Workflow Overview

The data curation process follows a systematic approach:

1. **Input**: Perturbation + Context data
2. **Enhanced Chat Processing**: Transform raw data into comprehensive reports
3. **Structured Explanation**: Convert reports into structured, machine-readable explanations

## Example

#### Input

```yaml
How does the following perturbation influence the cell in the described context, mechanistically and functionally?

{
    'context': {
        'perturbation_type': 'soluble factor', 
        'description': 'Soluble factor addition of VEGF', 'cell_type': 'N/A', 
        'disease_model': 'Angiogenic factor/tumors'
    }, 
    'perturbation': {
        'type': 'chemical', 
        'smiles': None, 
        'name': 'Bevacizumab', 
        'target': 'VEGFR', 
        'moa_type': 'antibody'
    }
}
```

#### Answer
```xml
<answer>
Bevacizumab is a humanized monoclonal antibody that binds directly to circulating VEGF-A with high specificity, functioning as a VEGF trap that prevents VEGF-A from ligating to its receptors VEGFR1 and VEGFR2. This sequestration causally blocks VEGFR2 tyrosine phosphorylation and downstream signaling cascades including PI3K/AKT, MAPK, and ERK1/2 pathways. The loss of VEGFR2 signaling directly inhibits endothelial cell proliferation, migration, and survival while promoting apoptosis and loss of endothelial fenestrations. These cellular effects lead to vessel regression with >70% reduction in vascular density, ultimately causing tumor growth inhibition and reduced metastasis. Transcriptomic changes correlatively include decreased VEGFR2/VEGFR3 expression and reduced proliferation/survival gene signatures, culminating in the measurable phenotype of decreased microvessel density and tumor growth suppression.
</answer>
```

#### Structured Explain
```php
<explain>
set_context(id="n0", cell_type="endothelial cell", disease_context="angiogenic tumor model", prior_perturbation="VEGF addition")
binds_to(id="n1", actor="Bevacizumab", target="VEGF-A", affinity="high", via="monoclonal antibody recognition")
modulates_activity(id="n2", entity="VEGFR2", direction="down", via="ligand sequestration preventing receptor binding")
modulates_activity(id="n3", entity="PI3K/AKT pathway", direction="down", via="loss of VEGFR2 phosphorylation")
modulates_activity(id="n4", entity="MAPK/ERK1/2 pathway", direction="down", via="blocked VEGFR2 signaling")
causes_phenotype(id="n5", source="VEGFR2 inhibition", phenotype="endothelial cell apoptosis", via="loss of survival signaling")
causes_phenotype(id="n6", source="VEGFR2 inhibition", phenotype="loss of endothelial fenestrations", via="morphological changes within 24h")
regulates_expression(id="n7", source="blocked VEGFR2 signaling", gene_or_signature=["VEGFR2","VEGFR3"], direction="down", via="reduced endothelial survival")
causes_phenotype(id="n8", source="endothelial dysfunction", phenotype="vessel regression", via=">70% reduction in vascular density")
rescues_phenotype(id="n9", actor="Bevacizumab", abnormal_phenotype="tumor angiogenesis", via="decreased microvessel density and tumor growth inhibition")
</explain>
```

#### DAG (relationship between predicates)

```graphql
edge("n1", "n2", relation="causal")
edge("n2", "n3", relation="causal")
edge("n2", "n4", relation="causal")
edge("n3", "n5", relation="causal")
edge("n4", "n5", relation="causal")
edge("n2", "n6", relation="causal")
edge("n3", "n7", relation="correlative")
edge("n4", "n7", relation="correlative")
edge("n5", "n8", relation="causal")
edge("n6", "n8", relation="causal")
edge("n8", "n9", relation="causal")
```

#### Thinking process (given report)
```
Let me work through this step by step.

The context is: soluble factor addition of VEGF in an angiogenic factor/tumor model, and the perturbation is Bevacizumab (an antibody targeting VEGFR - though the report clarifies it actually targets VEGF-A directly, not the receptor).

From the report, I can trace the following mechanistic pathway:

1. CAUSAL: Bevacizumab binds directly to circulating VEGF-A with high specificity, preventing VEGF-A from binding to its receptors (VEGFR1 and VEGFR2). This is a direct protein-protein interaction.

2. CAUSAL: By sequestering VEGF-A, bevacizumab prevents VEGF-induced VEGFR2 tyrosine phosphorylation, receptor dimerization, and activation.

3. CAUSAL: This blocks downstream signaling cascades including PI3K/AKT pathway, MAPK pathway, ERK1/2 cascade, and PLCγ1/Src/FAK pathways.

4. CAUSAL: Loss of VEGFR2 signaling leads to:
   - Decreased endothelial cell proliferation (blocks VEGF-stimulated KDR activity, prevents MAPK activation and DNA synthesis)
   - Inhibited endothelial cell migration and sprouting
   - Increased endothelial cell apoptosis
   - Loss of endothelial fenestrations within 24 hours

5. CAUSAL: These cellular effects lead to vessel regression - cessation of blood flow, >70% decrease in vascular density after 7 days.

6. CAUSAL: The vascular effects result in tumor growth inhibition, reduced metastasis, and decreased ascites formation.

For transcriptomic endpoints, the report mentions:
- CORRELATIVE: Decreased VEGFR-2 and VEGFR-3 expression in surviving endothelial cells
- CORRELATIVE: Potential upregulation of alternative angiogenic factors like VEGF-D
- CORRELATIVE: Reduced expression of genes involved in endothelial cell proliferation and survival

For phenotypic endpoints:
- CAUSAL: Decreased microvessel density, increased tumor cell apoptosis, reduced vascular permeability, decreased blood flow

The key thing to note is that the perturbation description says "target: VEGFR" but the actual mechanism is binding to VEGF-A ligand, not the receptor directly. I'll use the correct mechanism from the report.

I need to structure this as primitives with proper IDs and then create the DAG.
```
