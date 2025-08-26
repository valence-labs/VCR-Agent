from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
import re
import os
import tiktoken
from explain.util import load_data
# from explain.kg.kg_tool import KGNeighborTool
from explain.llm import LLMConfig
from explain.llm import create_client

class DataGenerator:
    def __init__(self, model_name: str, tool_list: list[str], **kwargs):
        self.model_name = model_name
        self.location = "us-east5"
        self.project = "vertexai-sandbox-e8a925d0"

        self.llm_client = create_client(provider=model_name, project_id=self.project, location=self.location)
        self.tools = self.get_tools(tool_list)
        DATA_DIR = 'data/curation_v1/'
        self.action_primitives, self.perturbation_cell_context, self.report_template, self.structre_explain_template = load_data(DATA_DIR)
        self.one_step_explain_template = open(os.path.join(DATA_DIR, "templates/one-step.txt")).read()
        if 'order' in kwargs and kwargs['order']:
            self.structure_explain_template = open(os.path.join(DATA_DIR, "templates/structure-explain-order.txt")).read()
        if 'openai' in model_name:
            self.encoding = tiktoken.encoding_for_model("gpt-4-1")

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


    def generate_response(self, input_prompt):
        messages = [
            {"role": "user", "content": input_prompt},
        ]
        # INSERT_YOUR_CODE
        import time

        max_retries = 10
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                result = self.llm_client.generate(messages, )
                if self.model_name in ['anthropic', 'litellm']:
                    response = result.messages[-1]['content'][0]['text']
                else:
                    response = result.content
                return response
            except Exception as e:
                if self.model_name in ['anthropic', 'litellm']:
                    num_tokens = self.llm_client.client.sync_client.messages.count_tokens(model=self.llm_client.client.config.model, messages=messages).input_tokens
                elif self.model_name in ['openai']:
                    num_tokens = len(self.encoding.encode(input_prompt))
                else:
                    num_tokens = self.llm_client.client.client.models.count_tokens(model=self.llm_client.client.config.model, contents=input_prompt).total_tokens
                # In case when the length of tokens is too long, we truncate the input prompt
                while num_tokens > 190000:
                    input_prompt = input_prompt[:-5000]
                    messages = [{"role": "user", "content": input_prompt}]
                    if self.model_name in ['openai', 'anthropic', 'litellm']:
                        num_tokens = self.llm_client.client.sync_client.messages.count_tokens(model=self.llm_client.client.config.model, messages=messages).input_tokens
                    else:
                        num_tokens = self.llm_client.client.client.models.count_tokens(model=self.llm_client.client.config.model, contents=input_prompt).total_tokens       
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Retrying in {retry_delay} seconds... (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Max retries reached for RateLimitError.")
                    raise

        # result = self.llm_client.generate(messages)
        # response = result.messages[-1]['content'][0]['text']

        # return response

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

        response = self.generate_response(input_prompt)
        return response

    def generate_structure_explain(self, report, question):
        input_prompt = self.structre_explain_template.format(action_primitives=self.action_primitives,
                                                        report=report,
                                                        question=question)
        trial = 0
        while trial < 10:
            response = self.generate_response(input_prompt)
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
        response = self.generate_response(input_prompt)
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
        response = self.generate_response(input_prompt)
        return response