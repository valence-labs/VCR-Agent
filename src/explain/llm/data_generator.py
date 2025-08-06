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

        self.llm_client = create_client(model=model_name, provider="anthropic", **kwargs)


        self.tools = self.get_tools(tool_list)
        self.base_llm = self.get_base_llm(model_name)
        self.llm = self.base_llm.bind_tools(self.tools)
        DATA_DIR = '../../../emmanuel.noutahi/project/outgoing/hooke/hooke-explain/'
        self.action_primitives, self.perturbation_cell_context, self.report_template, self.structre_explain_template = load_data(DATA_DIR)


    def get_base_llm(self, model_name: str):
        """
        Get the LLM model for Langchain
        """
        max_tokens = 10000
        if 'gpt'in model_name:
            llm = ChatOpenAI(model=model_name, streaming=False, max_tokens=max_tokens)
        elif 'claude' in model_name:
            llm = ChatAnthropicVertex(model_name=model_name, project=self.project,
                    location=self.location, streaming=False, max_tokens=max_tokens)
        elif 'gemini' in model_name:
            llm =  ChatVertexAI(model_name=model_name, project=self.project,
                    location=self.location, streaming=False, max_tokens=max_tokens)
        return llm

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
        messages = [("system", "You are a biomedical reasoning assistant."), ("placeholder", "{chat_history}"),
                ("human", "{input_prompt}"), ("placeholder", "{agent_scratchpad}")]

        final_prompt = ChatPromptTemplate.from_messages(messages)
        agent = create_tool_calling_agent(self.llm, self.tools, final_prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True)

        while True:
            try:
                result = executor.invoke({'input_prompt': input_prompt})
                break
            except Exception as e:
                print(f"Error during executor.invoke: {e}")
                continue

        response = result['output'][0]['text']

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