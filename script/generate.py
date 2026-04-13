import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import wandb
from tqdm import tqdm

from explain.eval.score.evaluate import evaluate
from explain.literature.harmonizome_utils import get_harmonizome_info
from explain.literature.paperqa_utils import get_paperqa_info
from explain.literature.pubmed_utils import get_pubmed_info
from explain.literature.wikipedia_utils import get_wikipedia_info
from explain.llm.data_generator import DataGenerator
from explain.starkprimekg.starkprimekg import StarkPrimeKG
from explain.starkprimekg.starkprimekg_utils import get_kg_info
from explain.util import set_perturbation_ner_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM generation for perturbation analysis")
    parser.add_argument('--model_type', type=str, help='Model type (e.g., openai, anthropic, gemini)',
                        default='anthropic')
    parser.add_argument('--folder_name', type=str, help='Output folder name',
                        default='test')
    parser.add_argument('--experiment_name', type=str, help='Experiment name',
                        default='test')
    parser.add_argument('--tool_list', type=json.loads, default=["kg-ner"], help='Comma-separated list of tools (e.g., wikipedia)')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--mode', type=str, default='report-explain',
        help='Mode to run the experiment (report-explain, explain-only)')
    parser.add_argument('--kg_with_rel', action='store_true', help='Whether to use kg-entity with relation')
    parser.add_argument('--kg_num_neighbor', type=int, default=1, help='Number of neighbors to use for kg-embedding')
    parser.add_argument('--wandb_id', type=str, default="", help='Wandb ID')
    parser.add_argument('--order', action='store_true', help='Whether to use order in structure explain')
    parser.add_argument('--report_file_name', type=str, default="", help='Report file name')
    parser.add_argument('--paperqa_num_papers', type=int, default=30, help='Number of papers to use for paperqa')
    parser.add_argument('--pubmed_num_papers', type=int, default=30, help='Number of papers to use for pubmed')
    parser.add_argument('--max_items', type=int, default=0, help='Limit number of perturbations to process (0 means all)')
    parser.add_argument('--pert_path', type=str, default="data/perturbation_ner_mapping.json", help='Perturbation path')

    return parser.parse_args()


def fetch_tool_info(tool_name):
    start = time.time()
    result = {"tool": tool_name, "additional": ""}
    if 'kg' in tool_name:
        kg_info, extracted_graph_info = get_kg_info(index, perturbation, tool_name, kg, args.kg_with_rel, args.kg_num_neighbor)
        kg_info = data_generator.post_process_additional_info(kg_info, tool_name, perturbation)
        result["kg_info"] = kg_info
        result["extracted_graph_info"] = extracted_graph_info
        result['additional'] = kg_info
    elif 'harmonizome' in tool_name:
        ner_flag = 'ner' in tool_name
        harmonizome_info = get_harmonizome_info(index, perturbation, is_ner=ner_flag)
        harmonizome_info = data_generator.post_process_additional_info(harmonizome_info, tool_name, perturbation)
        result["harmonizome_info"] = harmonizome_info
        result['additional'] = harmonizome_info
    elif 'wikipedia' in tool_name:
        wikipedia_info = get_wikipedia_info(index, perturbation)
        wikipedia_info = data_generator.post_process_additional_info(wikipedia_info, tool_name, perturbation)
        result["wikipedia_info"] = wikipedia_info
        result['additional'] = wikipedia_info
    elif 'paperqa' in tool_name:
        paperqa_info, papers_info = get_paperqa_info(index, perturbation, question, args.paperqa_num_papers, mode=tool_name)
        papers_info = data_generator.post_process_additional_info(papers_info, tool_name, perturbation)
        result["paperqa_info"] = paperqa_info
        result["papers_info"] = papers_info
        result['additional'] = papers_info
        if 'paperqa_list' not in tool_name:
            result['additional'] += '## PAPERQA INFORMATION\n' + paperqa_info + '\n\n'
    elif 'pubmed' in tool_name:
        pubmed_info = get_pubmed_info(index, perturbation, question, args.pubmed_num_papers, mode=tool_name)
        pubmed_info = data_generator.post_process_additional_info(pubmed_info, tool_name, perturbation)
        result["pubmed_info"] = pubmed_info
        result['additional'] = pubmed_info


    end = time.time()
    result["elapsed"] = end - start
    return result


if __name__ == '__main__':
    args = parse_args()
    # get index tag
    if 'tahoe' in args.pert_path:
        index_tag = 'tahoe_'
    elif 'rxrx' in args.pert_path:
        match = re.search(r'\d+', args.pert_path)
        number_in_pert_path = match.group(0) if match else None
        index_tag = f'rxrx_{number_in_pert_path}_'
    else:
        index_tag = ''
    perturbation_ner_mapping = json.load(open(args.pert_path))
    perturbation_ner_mapping_data_indices = [index_tag + str(item['index']) for item in perturbation_ner_mapping]
    set_perturbation_ner_mapping(perturbation_ner_mapping, perturbation_ner_mapping_data_indices)


    if args.wandb_id != "":
        wandb.init(entity='valencelabs', project="hooke-explain-datagen", mode=args.wandb_mode, name=f"{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}",
        id=args.wandb_id, resume=True)
    else:
        wandb.init(entity='valencelabs', project="hooke-explain-datagen", mode=args.wandb_mode, name=f"{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}")
    wandb.config.update(args)

    data_generator = DataGenerator(args.model_type, args.tool_list, order=args.order, pert_path=args.pert_path)
    report_list = []
    structure_explain_list = []
    if len(args.report_file_name) > 0:
        report_file_name = args.report_file_name
    else:
        report_file_name = f'output/report/{args.folder_name}/{index_tag}{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}.json'
    structure_explain_file_name = f'output/structure_explain/{args.folder_name}/{index_tag}{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}.json'
    # Ensure output directories exist
    os.makedirs(os.path.dirname(report_file_name), exist_ok=True)
    os.makedirs(os.path.dirname(structure_explain_file_name), exist_ok=True)
    perturbations = data_generator.perturbation_cell_context
    # Optional limit for quick tests
    if args.max_items and args.max_items > 0:
        perturbations = perturbations[:args.max_items]
    if os.path.exists(report_file_name):
        if report_file_name.endswith('.json'):
            report_list = json.load(open(report_file_name))
        elif report_file_name.endswith('.csv'):
            report_list = pd.read_csv(report_file_name)
            report_list = report_list.to_dict(orient='records')
    if os.path.exists(structure_explain_file_name):
        structure_explain_list = json.load(open(structure_explain_file_name))
    # load kg
    if any('kg' in tool for tool in args.tool_list):
        kg = StarkPrimeKG()


    processed_structure_indices = set([d['index'] for d in structure_explain_list]) if len(structure_explain_list) > 0 else set()
    additional_info, paperqa_info, wikipedia_info, harmonizome_info, kg_info = "", "", "", "", ""
    extracted_graph_info, papers_info, pubmed_info = {}, {}, ""
    for i, perturbation in enumerate(tqdm(perturbations)):
        index = index_tag + str(perturbation['index'])
        if index in processed_structure_indices:
            continue
        # Reset per-iteration variables to avoid cross-contamination and large prompts
        additional_info = ""
        kg_info = ""
        harmonizome_info = ""
        wikipedia_info = ""
        paperqa_info = ""
        extracted_graph_info = {}
        papers_info = {}
        perturbation_text = json.dumps(perturbation, indent=4)
        question = "**Q: How does the following perturbation influence the cell in the described context, mechanistically and functionally?**\n\n"
        question += perturbation_text

        # Search for additional information
        if args.mode == 'explain-only' and len(report_list) > i:
            pass
        else:
            if len(args.tool_list) <= 1:
                results = [fetch_tool_info(tool) for tool in args.tool_list]
            else:
                max_workers = min(4, len(args.tool_list))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(fetch_tool_info, tool): tool for tool in args.tool_list}
                    results = [f.result() for f in as_completed(futures)]
            # Combine results in the original tool order for determinism
            tool_to_result = {r["tool"]: r for r in results}
            for tool in args.tool_list:
                if tool not in tool_to_result:
                    continue
                r = tool_to_result[tool]
                additional_info += r.get("additional", "") + '\n\n'
                if "kg_info" in r:
                    kg_info = r["kg_info"]
                if "extracted_graph_info" in r:
                    extracted_graph_info = r["extracted_graph_info"]
                if "harmonizome_info" in r:
                    harmonizome_info = r["harmonizome_info"]
                if "wikipedia_info" in r:
                    wikipedia_info = r["wikipedia_info"]
                if "paperqa_info" in r:
                    paperqa_info = r["paperqa_info"]
                if "papers_info" in r:
                    papers_info = r["papers_info"]
                if "pubmed_info" in r:
                    pubmed_info = r["pubmed_info"]
                if "elapsed" in r:
                    print(f"Time taken for {tool}: {r['elapsed']} seconds")
        # report generation
        time_start = time.time()
        if 'report' in args.mode:
            report = data_generator.generate_report(perturbation, additional_info)
            report_dict = {'index': index, 'perturbation': perturbation, 'report_text': report, 'question': question,
            'kg_info': kg_info, 'extracted_graph_info': extracted_graph_info, 'harmonizome_info': harmonizome_info,
            'wikipedia_info': wikipedia_info, 'paperqa_info': paperqa_info, 'papers_info': papers_info, 'pubmed_info': pubmed_info}
            report_list.append(report_dict)
        else:
            # mode: Explain-only
            if len(report_list) > i:
                report = report_list[i]['report_text']
            else:
                report = ""
        time_end = time.time()
        print(f"Time taken for report generation: {time_end - time_start} seconds")
        time_start = time.time()
        # structure explain generation
        if len(report) == 0:
            # Explain-only (with additional info or not)
            structure_explain = data_generator.generate_one_step_structure_explain(question, additional_info)
        else:
            structure_explain = data_generator.generate_structure_explain(report, question)
        thinking, answer, explain, dag = data_generator.process_structure_explain(structure_explain)
        structure_explain_dict = {'index': index, 'input_perturbation': perturbation, 'thinking': thinking,
        'answer': answer, 'explain': explain, 'dag': dag, 'raw_response': structure_explain,
        'question': question, 'input_report_text': report, 'additional_info': additional_info}

        time_end = time.time()
        print(f"Time taken for structure explain generation: {time_end - time_start} seconds")


        structure_explain_list.append(structure_explain_dict)
        processed_structure_indices.add(index)

        if i % 20 == 0 or len(structure_explain_list) == len(perturbations):
            # Sort report_list by index
            report_list.sort(key=lambda x: x['index'])
            structure_explain_list.sort(key=lambda x: x['index'])
            with open(report_file_name, 'w') as f:
                json.dump(report_list, f)
            with open(structure_explain_file_name, 'w') as f:
                json.dump(structure_explain_list, f)

    gt_path = 'data/curation_v1/results/structure-explain-results-v4-claude4.csv'
    if index_tag == "" and os.path.exists(gt_path):
        # Do the evaluation for the dataset with GT (EC)
        gen_data = structure_explain_list
        gen_data.sort(key=lambda x: str(x['index']))
        gt_data = pd.read_csv(gt_path)[:len(gen_data)]
        gt_data = gt_data.to_dict(orient='records')
        gt_data.sort(key=lambda x: str(x['index']))

        score_file_name = f'output/score/{args.folder_name}/{args.experiment_name}_{args.tool_list}_{args.kg_num_neighbor}.json'
        os.makedirs(os.path.dirname(score_file_name), exist_ok=True)
        evaluate(gen_data, gt_data, score_file_name)
