import os
import json
import argparse
import sys
from tqdm import tqdm


from explain.llm.data_generator import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM generation for perturbation analysis")
    parser.add_argument('--model_type', type=str, help='Model type (e.g., gpt-4o, claude-opus-4@20250514, gemini-2.5-flash)',
                        default='claude-opus-4@20250514')
    parser.add_argument('--folder_name', type=str, help='Output folder name',
                        default='vanilla')
    parser.add_argument('--experiment_name', type=str, help='Experiment name',
                        default='')
    parser.add_argument('--tool_list', type=json.loads, default=['kg_neighbor'], help='Comma-separated list of tools (e.g., wikipedia)')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--metrics', type=json.loads, default=['llm_gen', 'format', 'compare_token'], help='Metrics to evaluate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataGenerator(args.model_type, args.tool_list)
    report_list = []
    structure_explain_list = []
    report_file_name = f'output/report/{args.folder_name}/{args.experiment_name}_{args.tool_list}.json'
    structure_explain_file_name = f'output/structure_explain/{args.folder_name}/{args.experiment_name}_{args.tool_list}.json'
    perturbations = data_generator.perturbation_cell_context

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
        report = data_generator.generate_report(perturbation)
        report_dict = {'index': i, 'perturbation': perturbation, 'report_text': report, 'question': question}
        report_list.append(report_dict)

        # structure explain generation
        structure_explain = data_generator.generate_structure_explain(report, question)
        thinking, answer, explain, dag = data_generator.process_structure_explain(structure_explain)
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