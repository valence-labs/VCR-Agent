import os
import json
import argparse
import sys
from tqdm import tqdm


from explain.util import load_data
from explain.llm.llm_utils import *



def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM generation for perturbation analysis")
    parser.add_argument('--model_type', type=str, help='Model type (e.g., gpt-4o, claude-opus-4@20250514, gemini-2.5-flash)',
                        default='claude-opus-4@20250514')
    parser.add_argument('--folder_name', type=str, help='Output folder name',
                        default='vanilla')
    parser.add_argument('--experiment_name', type=str, help='Experiment name',
                        default='')
    parser.add_argument('--tool_list', type=json.loads, default=[], help='Comma-separated list of tools (e.g., wikipedia)')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--metrics', type=json.loads, default=['llm_gen', 'format', 'compare_token'], help='Metrics to evaluate')
    return parser.parse_args()

DATA_DIR = '../../../emmanuel.noutahi/project/outgoing/hooke/hooke-explain/'
action_primitives, perturbation_cell_context, report_template, structre_explain_template = load_data(DATA_DIR)
if __name__ == '__main__':
    args = parse_args()
    tools = get_tools(args.tool_list)
    vanilla_llm = get_llm(args.model_type)
    llm = vanilla_llm.bind_tools(tools)
    report_list = []
    structure_explain_list = []
    report_file_name = f'output/report/{args.folder_name}/{args.experiment_name}_{args.tool_list}.json'
    structure_explain_file_name = f'output/structure_explain/{args.folder_name}/{args.experiment_name}_{args.tool_list}.json'
    perturbations = perturbation_cell_context
    
    if os.path.exists(report_file_name):
        report_list = json.load(open(report_file_name, 'r'))
    if os.path.exists(structure_explain_file_name):
        structure_explain_list = json.load(open(structure_explain_file_name, 'r'))
    for i, perturbation in enumerate(tqdm(perturbations)):
        if i in [structure_explain_dict['index'] for structure_explain_dict in structure_explain_list]:
            continue
        question = "**Q: How does the following perturbation influence the cell in the described context, mechanistically and functionally?**\n\n"
        question += json.dumps(perturbation, indent=4)

        # report generation
        report = generate_report(llm, perturbation, report_template, tools)
        print(report)
        report_dict = {'index': i, 'perturbation': perturbation, 'report_text': report, 'question': question}
        report_list.append(report_dict)

        # structure explain generation
        structure_explain = generate_structure_explain(llm, report, question, structre_explain_template, action_primitives)
        thinking, answer, explain, dag = process_structure_explain(structure_explain)
        structure_explain_dict = {'index': i, 'input_perturbation': perturbation, 'thinking': thinking,
        'answer': answer, 'explain': explain, 'dag': dag, 'raw_response': structure_explain,
        'question': question, 'input_report_text': report}
        print(structure_explain)
        structure_explain_list.append(structure_explain_dict)

        if i % 5 == 0 or i == len(perturbations) - 1:
            print(i)
            with open(report_file_name, 'w') as f:
                json.dump(report_list, f)
            with open(structure_explain_file_name, 'w') as f:
                json.dump(structure_explain_list, f)