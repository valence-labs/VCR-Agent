from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import json
import re
import os
import tiktoken
from anthropic import APIStatusError

from explain.util import load_data
from explain.llm import create_client
from explain.literature.harmonizome_utils import map_gene_set_to_related_genes
from explain.literature.harmonizome import Harmonizome, Entity

class DataGenerator:
    def __init__(self, model_name: str, tool_list: list[str], **kwargs):
        self.model_name = model_name
        self.location = "us-east5"
        self.project = "vertexai-sandbox-e8a925d0"

        self.llm_client_report = create_client(provider=model_name, project_id=self.project, location=self.location)
        self.llm_client_explain = create_client(provider=model_name, project_id=self.project, location=self.location)
        self.llm_client_rephrase = create_client(provider=model_name, project_id=self.project, location=self.location)
        self.tools = self.get_tools(tool_list)
        DATA_DIR = 'data/curation_v1/'
        pert_path = kwargs['pert_path']
        self.action_primitives, self.perturbation_cell_context, self.report_template, self.structre_explain_template = load_data(DATA_DIR, pert_path)
        self.one_step_explain_template = open(os.path.join(DATA_DIR, "templates/one-step.txt")).read()
        if 'order' in kwargs and kwargs['order']:
            self.structure_explain_template = open(os.path.join(DATA_DIR, "templates/structure-explain-order.txt")).read()
        if 'openai' in model_name:
            self.encoding = tiktoken.encoding_for_model("gpt-4-1")
        
        # Add caching for responses
        self.response_cache = {}
        self.max_cache_size = 1000

    def get_tools(self, tool_list: list[str]):
        """
        Get the list of tools for Langchain 
        """
        tools = []
        if 'wikipedia' in tool_list:
            wiki = WikipediaAPIWrapper(lang="en", top_k_results=3, doc_content_chars_max=2000)
            wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
            tools.append(wiki_tool)
        # if 'kg_neighbor' in tool_list:
        #     kg_tool = KGNeighborTool()
        #     tools.append(kg_tool)
            
        return tools

    def _get_cache_key(self, input_prompt):
        """Generate a cache key for the input prompt"""
        import hashlib
        return hashlib.md5(input_prompt.encode()).hexdigest()

    def _clean_cache(self):
        """Clean cache if it gets too large"""
        if len(self.response_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.response_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.response_cache[key]

    def generate_response(self, input_prompt, mode):
        # Simple cache to avoid duplicate generations on identical prompts
        # mode: report, explain, rephrase
        cache_key = self._get_cache_key(input_prompt)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            messages = [{"role": "user", "content": input_prompt}]
            if mode == 'report':
                result = self.llm_client_report.generate(messages, )
            elif mode == 'explain':
                result = self.llm_client_explain.generate(messages, )
            elif mode == 'rephrase':
                result = self.llm_client_rephrase.generate(messages, )
        except APIStatusError:
            if self.model_name in ['openai']:
                enc = tiktoken.encoding_for_model("gpt-4-1")  
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(input_prompt)
            num_tokens = 850000
            while True:
                try:
                    trimmed_tokens = tokens[:num_tokens]
                    trimmed_input_prompt = enc.decode(trimmed_tokens)
                    messages = [{"role": "user", "content": trimmed_input_prompt}]
                    result = self.llm_client.generate(messages, )
                    break
                except APIStatusError:
                    num_tokens -= 5000
                    continue

        if self.model_name in ['anthropic', 'litellm']:
            response = result.messages[-1]['content'][0]['text']
        else:
            response = result.content
        # Store in cache with simple size control
        self.response_cache[cache_key] = response
        self._clean_cache()
        return response
        


    def generate_report(self, perturbation, additional_info):
        input_prompt = self.report_template.format(treatment=json.dumps(perturbation, indent=4))
        if len(additional_info) > 0:
            # TODO: integrate with template 
            expected_output_marker = "### EXPECTED OUTPUT"
            idx = input_prompt.find(expected_output_marker)
            input_prompt = (
                    input_prompt[:idx]
                    + f"\n# ADDITIONAL INFORMATION\n{additional_info}\n"
                    +  "Please use the additional information to generate the report."
                    + input_prompt[idx:]
                )

        response = self.generate_response(input_prompt, mode='report')
        return response

    def generate_structure_explain(self, report, question):
        input_prompt = self.structre_explain_template.format(action_primitives=self.action_primitives,
                                                        report=report,
                                                        question=question)
        trial = 0
        while trial < 10:
            response = self.generate_response(input_prompt, mode='explain')
            if response is None:
                trial += 1
                continue
            if len(response) > 0:
                break
        if trial == 10:
            return ""
        return response


    def generate_one_step_structure_explain(self, question, information):
        input_prompt = self.one_step_explain_template.format(action_primitives=self.action_primitives, question=question, information=information)
        if len(information) == 0:
            input_prompt = input_prompt.replace("## Additional information\n", "")
        response = self.generate_response(input_prompt, mode='explain')
        return response

    def extract_tag(self, text, tag):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""


    def process_structure_explain(self, structure_explain):
        """
        Process the structure explain -> thinking, answer, explain, dag
        """
        if structure_explain is None:
            return "", "", "", ""
        thinking = self.extract_tag(structure_explain, "think")
        answer = self.extract_tag(structure_explain, "answer")
        explain = self.extract_tag(structure_explain, "explain")
        dag = self.extract_tag(structure_explain, "dag")

        return thinking, answer, explain, dag

    def rephrase_additional_info(self, additional_info, perturbation):
        input_prompt_template = """
        You are a biomedical assistant that rephrase the following information.
        Your task is to generate a **concise** and **informative** information given the additional information.
        The information will later be used to generate a report, which will be used to describe how the specified perturbation affects celluar biology in the given context.
        The future question is:

        ## QUESTION
        **Q: How does the following perturbation influence the cell in the described context, mechanistically and functionally?**  

        ## PERTURBATION
        {perturbation}

        ## ADDITIONAL INFORMATION
        {additional_info}

        ## EXPECTED OUTPUT
        Please generate a **concise** and **informative** information given the additional information.
        Do not omit any information and try to summarize all the information in ADDITIONAL INFORMATION.
        You don't need to answer the question, just rephrase the additional information that will be helpful to generate the report.
        """
        input_prompt = input_prompt_template.format(additional_info=additional_info, perturbation=json.dumps(perturbation, indent=4))
        response = self.generate_response(input_prompt, mode='rephrase')
        return response

    def schema_additional_info(self, info, tool):
        schema_info = []
        if 'kg' in tool:
            entities = info.split('\n\n')[:-1]
            for entity in entities:
                total_entity = entity.split('- ')   
                total_entity = [en.strip() for en in total_entity if len(en) > 0]
                total_entity = [{en.split(':')[0].strip(): ':'.join(en.split(':')[1:]).strip()} for en in total_entity if len(en.split(':')) > 1]
                schema_info.append(total_entity)
            
            result = {"KG nodes": schema_info}

        elif 'harmonizome' in tool:
            gene_entities = info[-1]['gene_info']
            gene_set_entities = info[-1]['gene_set_info']
            for gene_entity in gene_entities:
                gene_entity['gene_name'] = gene_entity['name']
                schema_info.append(gene_entity)
            for gene_set_entity in gene_set_entities:
                gene_set_entity['gene_set_name'] = gene_set_entity['name']
                related_genes = map_gene_set_to_related_genes(gene_set_entity)
                related_gene_info = []
                for related_gene in related_genes:
                    gene_info = Harmonizome.get(Entity.GENE, name=related_gene)
                    related_gene_info.append(gene_info)
                gene_set_entity['related_genes'] = related_gene_info
                schema_info.append(gene_set_entity)

            result = {"Gene information": schema_info}

        elif 'wikipedia' in tool:
            entities = [entity for entity in info.split('##') if len(entity) > 0]
            for entity in entities:
                key = entity.split('\n')[0].strip()
                value = '\n'.join(entity.split('\n')[1:]).strip()
                schema_info.append({'entity_name': key, 'context': value})
            result = {"wikipedia_info": schema_info}

        elif 'paperqa' in tool:
            result = {'paper_info': info}

            # result = {"graph_info": graph_info}
        return json.dumps(result)


    def post_process_additional_info(self, input_info, tool, perturbation):
        
        tool_name_dict = {'kg': 'KNOWLEDGE GRAPH INFORMATION', 
        'harmonizome': 'GENE INFORMATION', 
        'wikipedia': 'WIKIPEDIA INFORMATION', 
        'paperqa_list': 'RELATED PAPER LIST',
        'pubmed': 'RELATED PAPER LIST'}
        
        if 'schema' in tool:
            result_info = self.schema_additional_info(input_info, tool)
        else:
            if 'rephrase' in tool:
                result_info = self.rephrase_additional_info(input_info, perturbation)
            else:
                result_info = input_info
            result_info = str(result_info)
            tool = tool.replace('-rephrase', '').replace('-subgraph', '').replace('-schema', '').replace('-ner', '').replace('-fast', '')
            result_info = '## ' + tool_name_dict[tool] + '\n' + result_info + '\n\n'

        return result_info
