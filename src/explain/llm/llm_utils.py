import json
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import OpenAI as OPENAI_langchain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
import re

LOCATION = "us-east5"
PROJECT = "vertexai-sandbox-e8a925d0"

def get_tools(tool_list: list[str]):
    """
    Get the list of tools for Langchain
    """
    tools = []
    if 'wikipedia' in tool_list:
        wiki = WikipediaAPIWrapper(lang="en", top_k_results=3, doc_content_chars_max=2000)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
        tools.append(wiki_tool)
    return tools

def get_llm(model_name: str):
    """
    Get the LLM model for Langchain
    """
    max_tokens = 10000
    if 'gpt'in model_name:
        llm = ChatOpenAI(model=model_name, streaming=False, max_tokens=max_tokens)
    elif 'claude' in model_name:
        llm = ChatAnthropicVertex(model_name=model_name, project=PROJECT,
                location=LOCATION, streaming=False, max_tokens=max_tokens)
    elif 'gemini' in model_name:
        llm =  ChatVertexAI(model_name=model_name, project=PROJECT,
                location=LOCATION, streaming=False, max_tokens=max_tokens)
    return llm

def generate_response(input_prompt, llm, tools=[]):
    messages = [("system", "You are a biomedical reasoning assistant."), ("placeholder", "{chat_history}"),
            ("human", "{input_prompt}"), ("placeholder", "{agent_scratchpad}")]

    final_prompt = ChatPromptTemplate.from_messages(messages)
    agent = create_tool_calling_agent(llm, tools, final_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    while True:
        try:
            result = executor.invoke({'input_prompt': input_prompt})
            break
        except Exception as e:
            print(f"Error during executor.invoke: {e}")
            continue

    response = result['output'][0]['text']

    return response

def generate_report(llm, perturbation, report_template, tools):
    input_prompt = report_template.format(treatment=json.dumps(perturbation, indent=4))
    response = generate_response(input_prompt, llm, tools)
    return response

def generate_structure_explain(llm, report, question, structure_explain_template, action_primitives):
    input_prompt = structure_explain_template.format(action_primitives=action_primitives,
                                                    report=report,
                                                    question=question)
    response = generate_response(input_prompt, llm)
    return response

def extract_tag(text, tag):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def process_structure_explain(structure_explain):
    """
    Process the structure explain -> thinking, answer, explain, dag
    """
    
    thinking = extract_tag(structure_explain, "think")
    answer = extract_tag(structure_explain, "answer")
    explain = extract_tag(structure_explain, "explain")
    dag = extract_tag(structure_explain, "dag")


    return thinking, answer, explain, dag