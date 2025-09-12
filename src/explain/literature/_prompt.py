EVIDENCE_ONLY_PROMPT = """You are a scientific expert tasked with answering questions based STRICTLY on provided evidence from research papers.

Given the question and evidence below, provide a clear, concise, and accurate answer. Follow these guidelines:

1. Base your answer ONLY on the provided evidence - do NOT use external knowledge
2. If the evidence doesn't fully support an answer, clearly state the limitations
3. Use scientific terminology appropriately
4. Cite relevant findings from the evidence when possible
5. Keep the answer focused and approximately 2-3 paragraphs
6. If evidence is contradictory or insufficient, mention this explicitly
7. Do NOT speculate beyond what the evidence shows

Question: {query}

Evidence from research papers:
{evidence_combined}

Answer (based only on provided evidence):"""

COMPREHENSIVE_PROMPT = """You are a scientific expert tasked with answering questions using both provided evidence from research papers and your scientific knowledge.

Given the question and evidence below, provide a clear, concise, and accurate answer. Follow these guidelines:

1. Primarily base your answer on the provided evidence
2. You may supplement with your scientific knowledge when it adds value
3. Clearly distinguish between what comes from the evidence vs. your knowledge
4. Use scientific terminology appropriately
5. Cite relevant findings from the evidence when possible
6. Keep the answer focused and approximately 2-3 paragraphs
7. If evidence contradicts established knowledge, note this discrepancy

Question: {query}

Evidence from research papers:
{evidence_combined}

Answer (using evidence and scientific knowledge):"""
