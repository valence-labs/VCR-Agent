import os
import json


def load_data(data_dir):
    # load the action primitives
    with open(os.path.join(data_dir, "action_primitives.json")) as f:
        action_primitives = json.load(f)
    pert_path = os.path.join(data_dir, "data", "perturbations.json")
    with open(pert_path) as f:
        perturbation_cell_context = json.load(f)
    report_template = open(os.path.join(data_dir, "templates/generate-report.txt")).read()
    structre_explain_template = open(os.path.join(data_dir, "templates/structure-explain.txt")).read()
    return action_primitives, perturbation_cell_context, report_template, structre_explain_template

