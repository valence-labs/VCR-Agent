from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
import re

from explain.util import load_data
# from explain.kg.kg_tool import KGNeighborTool
from explain.llm import LLMConfig
from explain.llm import create_client

class DataGenerator:
    def __init__(self, model_name: str, tool_list: list[str], **kwargs):
        self.model_name = model_name
        self.location = "us-east5"
        self.project = "vertexai-sandbox-e8a925d0"

        self.llm_client = create_client(model=model_name, provider="anthropic", project_id=self.project, location=self.location)
        self.tools = self.get_tools(tool_list)
        DATA_DIR = 'data/curation_v1/'
        self.action_primitives, self.perturbation_cell_context, self.report_template, self.structre_explain_template = load_data(DATA_DIR)


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

        max_retries = 5
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                result = self.llm_client.generate(messages, )
                response = result.messages[-1]['content'][0]['text']
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit (Anthropic). Retrying in {retry_delay} seconds... (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Max retries reached for Anthropic RateLimitError.")
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
        
        thinking = self.extract_tag(structure_explain, "think")
        answer = self.extract_tag(structure_explain, "answer")
        explain = self.extract_tag(structure_explain, "explain")
        dag = self.extract_tag(structure_explain, "dag")

        return thinking, answer, explain, dag

    def rephrase_kg_info(self, kg_info, perturbation):
        input_prompt_template = """
        You are a biomedical assistant that rephrase the following information.
        Your task is to generate a **concise** and **informative** information given the knowledge graph information.
        The information will later be used to generate a report, which will be used to describe how the specified perturbation affects celluar biology in the given context.
        The future question is:

        ## QUESTION
        **Q: How does the following perturbation influence the cell in the described context, mechanistically and functionally?**  

        ## PERTURBATION
        {perturbation}

        ## KNOWLEDGE GRAPH INFORMATION
        {kg_info}

        ## EXPECTED OUTPUT
        Please generate a **concise** and **informative** information given the knowledge graph information.
        Do not omit any information and try to include all the information in KNOWLEDGE GRAPH INFORMATION.
        You don't need to answer the question, just rephrase the knowledge graph information that will be helpful to generate the report.
        """
        input_prompt = input_prompt_template.format(kg_info=kg_info, perturbation=json.dumps(perturbation, indent=4))
        response = self.generate_response(input_prompt)
        return response