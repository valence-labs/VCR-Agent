import os
import json
import argparse
import sys
from tqdm import tqdm
import pandas as pd
import wandb

from explain.llm.data_generator import DataGenerator
from explain.eval.score.syntax_score import SyntaxEvaluator
from explain.eval.score.accuracy_score import AccuracyEvaluator
from explain.eval.score.structure_explain import StructureExplain
from explain.eval.score.structural_score import StructuralEvaluator
from explain.eval.score.test import evaluate
from explain.kg.kg_utils import get_kg_entity_doc, entity_matching_info, get_kg_info
from explain.kg.kg import KnowledgeGraph
from explain.literature.harmonizome_utils import get_harmonizome_info
from explain.util import extract_entity_from_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM generation for perturbation analysis")
    parser.add_argument('--model_type', type=str, help='Model type (e.g., gpt-4o, claude-opus-4@20250514, gemini-2.5-flash)',
                        default='claude-opus-4@20250514')
    parser.add_argument('--folder_name', type=str, help='Output folder name',
                        default='test')
    parser.add_argument('--experiment_name', type=str, help='Experiment name',
                        default='test')
    parser.add_argument('--tool_list', type=json.loads, default=['kg-ner'], help='Comma-separated list of tools (e.g., wikipedia)')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--mode', type=str, default='report-explain',
        help='Mode to run the experiment(report-explain, explain-only)')
    parser.add_argument('--kg_with_rel', action='store_true', help='Whether to use kg-entity with relation')
    parser.add_argument('--kg_num_neighbor', type=int, default=1, help='Number of neighbors to use for kg-embedding')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    wandb.init(entity='valencelabs', project="hooke-explain-datagen", 
    mode=args.wandb_mode, name=f"{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}")
    wandb.config.update(args)


    data_generator = DataGenerator(args.model_type, args.tool_list)
    report_list = []
    structure_explain_list = []
    report_file_name = f'output/report/{args.folder_name}/{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}.json'
    structure_explain_file_name = f'output/structure_explain/{args.folder_name}/{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}.json'
    perturbations = data_generator.perturbation_cell_context

    if os.path.exists(report_file_name):
        report_list = json.load(open(report_file_name, 'r'))
    if os.path.exists(structure_explain_file_name):
        structure_explain_list = json.load(open(structure_explain_file_name, 'r'))
    
    additional_info = ""
    kg_info = ""
    harmonizome_info = ""
    extracted_graph_info = {}
    if any('kg' in tool for tool in args.tool_list):
        kg = KnowledgeGraph()
        
    for i, perturbation in enumerate(tqdm(perturbations)):
        if i in [structure_explain_dict['index'] for structure_explain_dict in structure_explain_list]:
            continue
        perturbation_text = json.dumps(perturbation, indent=4)
        question = "**Q: How does the following perturbation influence the cell in the described context, mechanistically and functionally?**\n\n"
        question += perturbation_text
        
        for tool in args.tool_list:
            if 'kg' in tool:
                kg_info, extracted_graph_info = get_kg_info(i, perturbation, tool, kg, args.kg_with_rel, args.kg_num_neighbor)
                if 'rephrase' in tool:
                    kg_info = data_generator.rephrase_kg_info(kg_info, perturbation)
                additional_info += '## KNOWLEDGE GRAPH INFORMATION\n' + kg_info + '\n\n'
            elif 'harmonizome' in tool:
                harmonizome_info = get_harmonizome_info(perturbation)
                additional_info += '## GENE INFORMATION\n' + harmonizome_info + '\n\n'
        # report generation
        if 'report' in args.mode:
            report = data_generator.generate_report(perturbation, additional_info)
            report_dict = {'index': i, 'perturbation': perturbation, 'report_text': report, 'question': question,
            'kg_info': kg_info, 'extracted_graph_info': extracted_graph_info, 'harmonizome_info': harmonizome_info}
            report_list.append(report_dict)
        else:
            report = report_list[i]['report_text']

        # structure explain generation
        structure_explain = data_generator.generate_structure_explain(report, question)
        thinking, answer, explain, dag = data_generator.process_structure_explain(structure_explain)
        structure_explain_dict = {'index': i, 'input_perturbation': perturbation, 'thinking': thinking,
        'answer': answer, 'explain': explain, 'dag': dag, 'raw_response': structure_explain,
        'question': question, 'input_report_text': report}


        structure_explain_list.append(structure_explain_dict)

        if i % 5 == 0 or i == len(perturbations) - 1:
            # Sort report_list by index
            report_list.sort(key=lambda x: x['index'])
            structure_explain_list.sort(key=lambda x: x['index'])
            print(i)
            with open(report_file_name, 'w') as f:
                json.dump(report_list, f)
            with open(structure_explain_file_name, 'w') as f:
                json.dump(structure_explain_list, f)
    gen_data = structure_explain_list
    gt_data = pd.read_csv('data/curation_v1/results/structure-explain-results-v3-claude4.csv')[:len(gen_data)]
    gt_data = gt_data.to_dict(orient='records')
    

    evaluate(gen_data, gt_data)