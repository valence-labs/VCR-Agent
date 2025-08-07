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
from explain.kg.kg_tool import KGNeighborTool
from explain.llm import LLMConfig
from explain.llm import create_client

class DataGenerator:
    def __init__(self, model_name: str, tool_list: list[str], **kwargs):
        self.model_name = model_name
        self.location = "us-east5"
        self.project = "vertexai-sandbox-e8a925d0"

        self.llm_client = create_client(model=model_name, provider="litellm", project_id=self.project, location=self.location)
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
        if 'kg_neighbor' in tool_list:
            kg_tool = KGNeighborTool()
            tools.append(kg_tool)
            
        return tools


    def generate_response(self, input_prompt):
        messages = [
            {"role": "user", "content": input_prompt},
        ]
        result = self.llm_client.generate(messages)
        response = result.messages[-1]['content'][0]['text']

        return response

    def generate_report(self, perturbation):
        input_prompt = self.report_template.format(treatment=json.dumps(perturbation, indent=4))
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