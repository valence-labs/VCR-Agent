"""
Demo notebook for Hooke-Explain module with Anthropic Vertex AI

This demonstrates the complete end-to-end workflow from generating sample reports
to converting them into structured explanations.
"""

import pandas as pd
import asyncio
from pathlib import Path

from src import (
    create_structure_explainer,
    EnhancedChatManager,
    ChatConfiguration
)

def demo_single_explanation():
    """Demo single structured explanation generation"""
    print("=== Single Explanation Demo ===\n")
    
    explainer = create_structure_explainer(
        model="claude-sonnet-4@20250514",
        location="us-east5",
        project_id="vertexai-sandbox-e8a925d0"
    )
    
    question = "How does PARP inhibition affect BRCA1-deficient cells mechanistically and functionally?"
    
    report = """
    # PARP Inhibition in BRCA1-Deficient Cancer Cells
    
    ## Background
    Poly ADP-ribose polymerase (PARP) enzymes are critical for DNA repair processes,
    particularly base excision repair. BRCA1-deficient cells have compromised 
    homologous recombination repair capacity due to mutations in the BRCA1 gene.
    
    ## Experimental Setup
    - Cell line: BRCA1-knockout HCC1937 breast cancer cells
    - Treatment: 10 μM Olaparib (PARP inhibitor)
    - Control: Vehicle-treated cells
    - Timepoints: 24h, 48h, 72h
    
    ## Key Findings
    
    ### DNA Damage Response
    - Significant accumulation of γ-H2AX foci (DNA damage marker)
    - Increased PARP trapping at DNA lesions
    - Conversion of single-strand breaks to double-strand breaks
    
    ### Cell Cycle Effects
    - G2/M checkpoint activation
    - Prolonged S-phase due to replication fork stalling
    - Increased p53 and p21 expression
    
    ### Apoptosis Induction
    - Caspase-3 activation at 48h
    - PARP cleavage (distinct from PARP inhibition)
    - Cytochrome c release from mitochondria
    
    ### Synthetic Lethality Mechanism
    - PARP inhibition blocks base excision repair
    - BRCA1 deficiency prevents homologous recombination
    - Cells forced to use error-prone non-homologous end joining
    - Accumulation of chromosomal aberrations
    - Ultimate cell death via apoptosis
    
    ## Molecular Pathways
    1. PARP1/2 normally detect and bind to DNA breaks
    2. Olaparib prevents PARP auto-modification and release
    3. Trapped PARP complexes block replication fork progression
    4. Replication fork collision creates double-strand breaks
    5. BRCA1-deficient cells cannot repair via homologous recombination
    6. Alternative repair pathways insufficient
    7. Persistent DNA damage triggers apoptosis
    
    ## Clinical Relevance
    This synthetic lethality mechanism forms the basis for PARP inhibitor
    therapy in BRCA1/2-mutated breast and ovarian cancers. The selectivity
    for cancer cells over normal cells provides a therapeutic window.
    """
    
    try:
        result = explainer.generate_explanation(
            question=question,
            report=report,
            validate_response=True
        )
        
        print("THINKING:")
        print("-" * 60)
        print(result.thinking)
        print("\nANSWER:")
        print("-" * 60)
        print(result.answer)
        print("\nSTRUCTURED EXPLANATION:")
        print("-" * 60)
        print(result.explain)
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def demo_batch_processing():
    """Demo batch processing with multiple samples"""
    print("\n\n=== Batch Processing Demo ===\n")
    
    explainer = create_structure_explainer()
    
    sample_data = [
        {
            "id": "egfr_inhibition",
            "question": "How does EGFR inhibition affect cancer cell signaling?",
            "report": """
            # EGFR Inhibition Study
            
            ## Background
            EGFR (Epidermal Growth Factor Receptor) is a receptor tyrosine kinase
            frequently overexpressed in cancer cells.
            
            ## Treatment
            - Gefitinib 1 μM treatment for 24h
            - A549 lung cancer cells
            
            ## Results
            - Reduced EGFR autophosphorylation
            - Decreased PI3K/AKT pathway activation
            - Reduced MAPK/ERK signaling
            - G1 cell cycle arrest
            - Increased apoptosis via BAD dephosphorylation
            """
        },
        {
            "id": "p53_activation", 
            "question": "What are the cellular consequences of p53 activation?",
            "report": """
            # p53 Activation Response
            
            ## Background
            p53 is a tumor suppressor protein that responds to DNA damage
            and cellular stress signals.
            
            ## Experimental Conditions
            - Doxorubicin-induced DNA damage
            - HCT116 colon cancer cells
            - 6h and 24h timepoints
            
            ## Outcomes
            - Rapid p53 stabilization and nuclear accumulation
            - Transcriptional activation of p21 (cell cycle arrest)
            - PUMA and BAX upregulation (apoptosis pathway)
            - MDM2 upregulation (negative feedback)
            - Choice between arrest and apoptosis depends on damage severity
            """
        },
        {
            "id": "wnt_signaling",
            "question": "How does Wnt pathway activation affect stem cell behavior?", 
            "report": """
            # Wnt Signaling in Stem Cells
            
            ## Background
            Wnt signaling is crucial for stem cell maintenance and differentiation.
            
            ## Experimental Setup
            - Intestinal organoids
            - Wnt3a treatment vs control
            - Single-cell RNA sequencing
            
            ## Key Findings
            - β-catenin nuclear translocation
            - TCF/LEF transcriptional activation
            - Lgr5+ stem cell expansion
            - c-Myc and cyclin D1 upregulation
            - Enhanced self-renewal capacity
            - Reduced differentiation markers
            """
        }
    ]
    
    try:
        results_df = explainer.process_batch(
            data=sample_data,
            question_column="question",
            report_column="report",
            id_column="id",
            validate_responses=True,
            show_progress=True
        )
        
        print(f"\nProcessed {len(results_df)} samples")
        print(f"Success rate: {results_df['success'].mean():.1%}")
        
        print("\nSample results:")
        for idx, row in results_df.iterrows():
            if row['success']:
                print(f"\n{row['id'].upper()}:")
                print(f"Answer: {row['answer'][:100]}...")
                print(f"Explain (first 2 lines):")
                explain_lines = row['explain'].split('\n')[:2]
                for line in explain_lines:
                    print(f"  {line}")
        
        stats = explainer.get_statistics(results_df)
        print(f"\nTop action primitives used:")
        for action, count in stats.get('most_used_actions', [])[:5]:
            print(f"  {action}: {count}")
        
        results_df.to_parquet("demo_batch_results.parquet", index=False)
        print("\nResults saved to demo_batch_results.parquet")
        
        return results_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

async def demo_enhanced_chat_integration():
    """Demo enhanced chat integration for report generation"""
    print("\n\n=== Enhanced Chat Integration Demo ===\n")
    
    print("This demo shows how to use enhanced chat to generate initial reports")
    print("that can then be processed by StructureExplainer.\n")
    
    try:
        from enhanced_chat_client import AuthenticatedClient
        from enhanced_chat_client.models.model import Model
        
        print("Setting up enhanced chat client...")
        client = AuthenticatedClient(
            base_url="https://enhanced-chat.centaur-platform-dev.com/", 
            token="your-token-here"
        )
        client = await client.__aenter__()
        
        chat_manager = EnhancedChatManager(client)
        
        chat_manager.update_config(
            model=Model.CLAUDE_V3_7_SONNET,
            pubmed_enabled=True,
            max_pubmed_to_read=3,
            fulltext_enabled=True,
            max_fulltext_to_read=2
        )
        
        report_template = """
        Generate a comprehensive biomedical report on the following topic:
        
        Topic: {topic}
        Focus: {focus}
        
        Include:
        1. Background and molecular context
        2. Experimental evidence from literature
        3. Key pathways and mechanisms
        4. Cellular and physiological effects
        5. Clinical relevance
        
        Format as markdown with clear sections.
        """
        
        topics_data = [
            {
                "topic": "mTOR inhibition in cancer therapy",
                "focus": "mechanistic effects on cell growth and metabolism"
            },
            {
                "topic": "CRISPR-Cas9 gene editing", 
                "focus": "DNA repair mechanisms and off-target effects"
            },
            {
                "topic": "Immunotherapy checkpoint inhibitors",
                "focus": "T cell activation and tumor immune evasion"
            }
        ]
        
        print("Submitting bulk report generation job...")
        prompts = chat_manager.create_prompts(
            templates=[report_template],
            data=topics_data,
            tags=["biomedical-reports", "demo"]
        )
        
        await chat_manager.submit_bulk_job(prompts, "demo-reports-v1")
        
        print("Waiting for completion...")
        completed = await chat_manager.wait_for_completion(
            "demo-reports-v1", 
            show_progress=True,
            timeout=600
        )
        
        if completed:
            result = await chat_manager.collect_results("demo-reports-v1")
            
            if result.success:
                print(f"Generated {len(result.conversations)} reports")
                
                messages = chat_manager.extract_messages(result.conversations)
                generated_reports = []
                
                for i, conv in enumerate(messages):
                    if conv['messages']:
                        report_content = conv['messages'][-1]['content']
                        generated_reports.append({
                            "topic": topics_data[i]["topic"],
                            "report": report_content
                        })
                
                print("\nNow processing generated reports with StructureExplainer...")
                
                explainer = create_structure_explainer()
                
                structured_data = []
                for item in generated_reports:
                    question = f"What are the key mechanisms described in this {item['topic']} report?"
                    
                    try:
                        parsed = explainer.generate_explanation(
                            question=question,
                            report=item['report'],
                            validate_response=True
                        )
                        
                        structured_data.append({
                            "topic": item["topic"],
                            "question": question,
                            "thinking": parsed.thinking,
                            "answer": parsed.answer,
                            "explain": parsed.explain,
                            "success": True
                        })
                        
                    except Exception as e:
                        print(f"Error processing {item['topic']}: {e}")
                        structured_data.append({
                            "topic": item["topic"], 
                            "success": False,
                            "error": str(e)
                        })
                
                final_df = pd.DataFrame(structured_data)
                final_df.to_parquet("demo_enhanced_chat_results.parquet", index=False)
                
                print(f"\nComplete workflow results:")
                print(f"- Generated reports: {len(generated_reports)}")
                print(f"- Structured explanations: {final_df['success'].sum()}")
                print(f"- Success rate: {final_df['success'].mean():.1%}")
                print("- Saved to demo_enhanced_chat_results.parquet")
                
                return final_df
                
            else:
                print(f"Enhanced chat job failed: {result.error}")
                return None
        else:
            print("Job did not complete within timeout")
            return None
            
    except ImportError:
        print("Enhanced chat client not available - showing workflow structure instead:")
        
        print("\nEnhanced Chat → StructureExplainer Workflow:")
        print("1. Use EnhancedChatManager to generate detailed reports")
        print("2. Extract report content from conversation results") 
        print("3. Create questions about the generated reports")
        print("4. Use StructureExplainer to convert reports to structured format")
        print("5. Export final structured results for analysis")
        
        print("\nExample generated report processing:")
        mock_report = """
        # mTOR Inhibition in Cancer Therapy
        
        ## Background
        mTOR (mechanistic Target of Rapamycin) is a serine/threonine kinase
        that regulates cell growth, proliferation, and metabolism...
        
        ## Mechanisms
        - mTORC1 controls protein synthesis via S6K1 and 4E-BP1
        - mTORC2 regulates AKT phosphorylation and cytoskeletal organization
        - Rapamycin selectively inhibits mTORC1
        """
        
        explainer = create_structure_explainer()
        
        try:
            result = explainer.generate_explanation(
                question="How does mTOR inhibition affect cancer cell behavior?",
                report=mock_report,
                validate_response=True
            )
            
            print("\nMock structured result:")
            print(f"Answer: {result.answer[:150]}...")
            
            return {"mock_demo": True, "success": True}
            
        except Exception as e:
            print(f"Error in mock demo: {e}")
            return None

def main():
    """Run all demonstrations"""
    print("Hooke-Explain Module Comprehensive Demo")
    print("=" * 50)
    
    demo_single_explanation()
    
    demo_batch_processing()
    
    print("\nRunning enhanced chat demo (async)...")
    result = asyncio.run(demo_enhanced_chat_integration())
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    
    print("\nKey capabilities demonstrated:")
    print("✓ Single structured explanation generation")
    print("✓ Batch processing with DataFrame output")
    print("✓ Enhanced chat integration workflow")
    print("✓ Action primitive validation and statistics")
    print("✓ Large report processing and condensation")
    
    print("\nOutput files generated:")
    print("- demo_batch_results.parquet")
    print("- demo_enhanced_chat_results.parquet (if enhanced chat available)")

if __name__ == "__main__":
    main() 