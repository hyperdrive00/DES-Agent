# DesAgent.py
import time
# start_time = time.time()
import os
# import torch
from operator import itemgetter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

import re
import fuzzywuzzy.fuzz as fuzz
from typing import List, Dict, Any
import streamlit as st

import pandas as pd
import json
from json_repair import repair_json
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# from agent_prompt import *
from prompts import (
    CYHPER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    FEEWSHOT_EXAMPLES,
    EXAMPLE_OUTPUT_PROMPT,
    CONVERT_CYPHER_SYSTEM_PROMPT
)
# Environment variables setup
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DES Agent Deploy"


NEO4J_URI = st.secrets['neo4j_credentials']['NEO4J_URI']
NEO4J_USER = st.secrets['neo4j_credentials']['NEO4J_USER']
NEO4J_PASSWORD = st.secrets['neo4j_credentials']['NEO4J_PASSWORD']

# OpenRouter configuration for free tier
OPENROUTER_FREE_API_KEY = st.secrets.get('OPENROUTER_FREE_API_KEY', None)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter Free Tier Models (updated list as of 2025)
OPENROUTER_FREE_MODELS = {
    # DeepSeek Models
    "deepseek/deepseek-chat-v3-0324:free": "DeepSeek Chat V3 - Excellent for coding and general tasks",
    "deepseek/deepseek-r1:free": "DeepSeek R1 - Advanced reasoning model with open reasoning tokens",
    "deepseek/deepseek-r1-zero:free": "DeepSeek R1 Zero - RL-trained model without SFT",
    "deepseek/deepseek-r1-0528-qwen3-8b:free": "DeepSeek R1 0528 Qwen3 8B - Latest reasoning model",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B - High performance distilled model",
    "deepseek/deepseek-r1-distill-qwen-32b:free": "DeepSeek R1 Distill Qwen 32B - Strong reasoning capabilities",
    "deepseek/deepseek-r1-distill-qwen-14b:free": "DeepSeek R1 Distill Qwen 14B - Balanced performance",
    "deepseek/deepseek-r1-distill-qwen-7b:free": "DeepSeek R1 Distill Qwen 7B - Efficient reasoning model",
    "deepseek/deepseek-r1-distill-qwen-1.5b:free": "DeepSeek R1 Distill Qwen 1.5B - Lightweight yet capable",
    "deepseek/deepseek-r1-distill-llama-8b:free": "DeepSeek R1 Distill Llama 8B - Good balance of size and performance",
    "deepseek/deepseek-v3-base:free": "DeepSeek V3 Base - 671B parameter base model",
    
    # Meta Llama Models
    "meta-llama/llama-4-scout:free": "Llama 4 Scout - Multimodal MoE model with 200K context",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct - High-quality instruction following",
    
    # Google Models
    "google/gemini-2.5-pro-exp-03-25:free": "Gemini 2.5 Pro Experimental - Latest Google model",
    "google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking - Fast reasoning model",
    "google/gemini-2.0-flash-exp:free": "Gemini 2.0 Flash Experimental - Quick responses",
    "google/gemma-3-27b-it:free": "Gemma 3 27B IT - Google's open model",
    
    # Qwen Models
    "qwen/qwen3-4b:free": "Qwen3 4B - Dual-mode architecture with 128K context",
    "qwen/qwen3-0.6b-04-28:free": "Qwen3 0.6B - Lightweight model with 32K context",
    "qwen/qwq-32b:free": "QwQ 32B - Specialized reasoning model",
    
    # NVIDIA Models
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "Llama 3.1 Nemotron Ultra 253B - NVIDIA's flagship model"
}

# Default fallback model
DEFAULT_FREE_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Default configuration for user API
DEFAULT_BASE_URL = None
DEFAULT_MODEL = "gpt-4o"

CYPHER_CLAUSE_KEYWORDS = [
    'MATCH',
    'OPTIONAL MATCH',
    'WHERE',
    'RETURN',
    'WITH',
    'CREATE',
    'MERGE',
    'DELETE',
    'SET',
    'UNWIND',
    'ORDER BY',
    'SKIP',
    'LIMIT',
    'FOREACH',
    'CALL',
    'DETACH DELETE',
    'REMOVE'
]

CLAUSE_PATTERN = '|'.join([re.escape(keyword) for keyword in CYPHER_CLAUSE_KEYWORDS])

# end_time = time.time()
# print(f"Time taken for imports: {end_time - start_time:.2f}s")

class FlexibleJsonOutputParser(BaseOutputParser):
    """
    A flexible JSON parser that can handle responses from models that don't strictly 
    follow JSON format requirements using json_repair.
    """
    
    def parse(self, text: str) -> dict:
        """Parse the output of an LLM call to a JSON object."""
        try:
            # First try standard JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Use json_repair to fix malformed JSON
                repaired = repair_json(text)
                return json.loads(repaired)
            except Exception as e:
                print(f"JSON repair failed: {e}")
                
                # Extract JSON-like content from text using regex as last resort
                import re
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, text, re.DOTALL)
                
                for match in matches:
                    try:
                        # Try to repair each potential JSON block
                        repaired_match = repair_json(match)
                        result = json.loads(repaired_match)
                        # Validate that it has expected structure
                        if isinstance(result, dict) and ('use_cypher' in result or 'cypher_query' in result):
                            return result
                    except:
                        continue
                
                # If all JSON extraction fails, construct a fallback response
                print(f"Falling back to text analysis for: {text[:200]}...")
                
                if "use_cypher" in text.lower():
                    if any(word in text.lower() for word in ["yes", "true", "match", "return"]):
                        # Try to extract cypher query
                        cypher_match = re.search(r'(MATCH.*?RETURN[^.]*)', text, re.IGNORECASE | re.DOTALL)
                        cypher_query = cypher_match.group(1).strip() if cypher_match else "MATCH (n) RETURN n LIMIT 5"
                        return {
                            "use_cypher": "yes",
                            "thought_process": "Extracted from free-form response due to JSON parsing issues",
                            "cypher_query": cypher_query
                        }
                    else:
                        return {
                            "use_cypher": "no",
                            "thought_process": text[:500] if len(text) > 500 else text
                        }
                else:
                    # Default fallback
                    return {
                        "use_cypher": "no",
                        "thought_process": text[:500] if len(text) > 500 else text
                    }

string_list = json.load(open('substance_string_list.json', 'r'))

def levenshtein_search(query, threshold=80):
    query_lower = query.lower()
    matches = []
    for s in string_list:
        string = s['string'].lower()
        # Calculate similarity score using Levenshtein distance
        similarity = fuzz.token_sort_ratio(query_lower, string)
        if similarity >= threshold:
            matches.append({
                'string': s['string'],
                'pubchem_cid': s['pubchem_cid'],
                'similarity': similarity
            })
    
    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x['similarity'], reverse=True)

    top_score = matches[0]['similarity']
    print(f"Top score: {top_score}")
    
    # Keep only top 5 unique IDs
    unique_ids = set()
    unique_matches = []
    for match in matches:
        if match['pubchem_cid'] not in unique_ids and len(unique_ids) < 5:
            unique_ids.add(match['pubchem_cid'])
            unique_matches.append(match)
    
    top_matches = [match for match in unique_matches if match['similarity'] == top_score]
            

    # cids = [match['pubchem_cid'] for match in unique_matches if match['similarity'] == top_score]
    
    # # Yield information about all the top-scoring matches being returned
    # for match in unique_matches:
    #     if match['similarity'] == top_score:
    #         yield(f"CID: {match['pubchem_cid']}, String: {match['string']}, Score: {match['similarity']}%")
    
    return top_matches

# --- Updated Transformation Function ----------------------------------
def transform_cypher_query(query_text):
    """
    Transforms a generated Cypher query so that:
      1. Literal substance names in node maps and WHERE conditions 
         (using pubchem_name) are replaced by their fuzzy–matched pubchem_cid values.
      2. If a node map in a MATCH clause (e.g. {pubchem_name:"X"}) yields exactly one CID,
         it is inlined as (u:Substance{pubchem_cid: <CID>}); if multiple CIDs are found, an extra
         condition for that alias is recorded.
      3. Conditions using equality (pubchem_name = "X"), inequality (pubchem_name <> "X")
         and literal IN clauses (pubchem_name IN ["X", ...]) are replaced accordingly.
      4. WITH clauses that define a literal list of substance names are transformed: 
         the list literal is replaced by the union of PubChem CIDs.
      5. Occurrences of a variable (from such a WITH clause) in conditions and map filters are updated:
         * "pubchem_name IN <variable>" becomes "pubchem_cid IN <variable>"
         * In map filters, {pubchem_name: substance} becomes {pubchem_cid: substance}
      6. Extra conditions (from node maps that couldn't be inlined) are merged into an existing WHERE clause,
         using AND.
      
    Returns:
      A tuple of (transformed_query, top_matches) where top_matches is a list of all substance matches found
    """
    # Dictionary to store all top matches found during transformation
    all_top_matches = []
    
    # --- Extra Step A: Process WITH Clauses that assign a literal list.
    # For example: WITH ["Sodium Chloride", "Calcium Chloride"] AS targetSubstances
    # We'll replace it with a list of corresponding CIDs.
    with_list_pattern = r'WITH\s*\[([^\]]+)\]\s+AS\s+(\w+)'
    def replace_with_list(match):
        list_contents = match.group(1)  # e.g. '"Sodium Chloride", "Calcium Chloride"'
        var_name = match.group(2)       # e.g. targetSubstances
        # Extract the quoted substance names using a pattern that accepts " ' or `
        names = [m[1] for m in re.findall(r'(["\'`])([^"\'`]+)\1', list_contents)]
        cid_set = set()
        for name in names:
            top_matches = levenshtein_search(name)
            all_top_matches.extend(top_matches)
            for match in top_matches:
                cid_set.add(match['pubchem_cid'])
        cid_list = sorted(cid_set)
        return f'WITH {cid_list} AS {var_name}'
    query_text = re.sub(with_list_pattern, replace_with_list, query_text)
    
    extra_conditions = {}  # For node maps that cannot be inlined
    
    # --- Step 1: Process Node Maps in MATCH clauses ---
    # Pattern: (alias:Substance { pubchem_name: "X" })
    node_map_pattern = r'(\w+):Substance\s*\{\s*pubchem_name\s*:\s*(["\'`])([^"\'`]+)\2\s*\}'
    def replace_node_map(match):
        alias = match.group(1)
        name = match.group(3)
        top_matches = levenshtein_search(name)
        all_top_matches.extend(top_matches)
        if len(top_matches) == 1:
            return f'{alias}:Substance{{pubchem_cid: {top_matches[0]["pubchem_cid"]}}}'
        else:
            cids = [match['pubchem_cid'] for match in top_matches]
            extra_conditions[alias] = f'{alias}.pubchem_cid IN [{", ".join(str(cid) for cid in cids)}]'
            return f'{alias}:Substance'
    query_text = re.sub(node_map_pattern, replace_node_map, query_text)
    
    # --- Step 2: Process Equality Conditions in WHERE clauses ---
    eq_pattern = r'((?:\w+\.)?)pubchem_name\s*=\s*(["\'`])([^"\'`]+)\2'
    def replace_eq(match):
        prefix = match.group(1)
        name = match.group(3)
        top_matches = levenshtein_search(name)
        all_top_matches.extend(top_matches)
        cids = [match['pubchem_cid'] for match in top_matches]
        return f'{prefix}pubchem_cid IN [{", ".join(str(cid) for cid in cids)}]'
    query_text = re.sub(eq_pattern, replace_eq, query_text)
    
    # --- Step 3: Process Inequality Conditions in WHERE clauses ---
    neq_pattern = r'((?:\w+\.)?)pubchem_name\s*<>\s*(["\'`])([^"\'`]+)\2'
    def replace_neq(match):
        prefix = match.group(1)
        name = match.group(3)
        top_matches = levenshtein_search(name)
        all_top_matches.extend(top_matches)
        cids = [match['pubchem_cid'] for match in top_matches]
        return f'NOT {prefix}pubchem_cid IN [{", ".join(str(cid) for cid in cids)}]'
    query_text = re.sub(neq_pattern, replace_neq, query_text)
    
    # --- Step 4: Process literal IN Clauses (when list is written inline) ---
    in_pattern = r'pubchem_name\s+IN\s+\[([^\]]+)\]'
    def replace_in(match):
        content = match.group(1)
        names = [n.strip().strip('"').strip("'").strip("`") for n in content.split(",")]
        cids_set = set()
        for name in names:
            top_matches = levenshtein_search(name)
            all_top_matches.extend(top_matches)
            for match in top_matches:
                cids_set.add(match['pubchem_cid'])
        cids_list = sorted(cids_set)
        return f'pubchem_cid IN [{", ".join(str(cid) for cid in cids_list)}]'
    query_text = re.sub(in_pattern, replace_in, query_text)
    
    # --- Step 5: Merge Extra Conditions into an existing WHERE clause using AND ---
    if extra_conditions:
        extra_str = " AND ".join(extra_conditions.values())
        pattern = r'(MATCH\s+.*?\))(\s+WHERE\s+)(.*?)(\s+(RETURN|WITH|MATCH)|\s*$)'
        def append_and(match):
            return f'{match.group(1)}{match.group(2)}{match.group(3).rstrip()} AND {extra_str}{match.group(4)}'
        (query_text, count) = re.subn(pattern, append_and, query_text, count=1, flags=re.DOTALL)
        if count == 0:
            query_text = re.sub(r'(\s+(RETURN|WITH|MATCH))', f' WHERE {extra_str} \\1', query_text, count=1)
    
    # --- Step 6: Replace Variable References in IN Clauses ---
    # Replace occurrences like "pubchem_name IN targetSubstances" with "pubchem_cid IN targetSubstances"
    variable_in_pattern = r'((?:\w+\.)?)pubchem_name\s+IN\s+(\w+)'
    def replace_variable_in(match):
        prefix = match.group(1)
        var = match.group(2)
        return f'{prefix}pubchem_cid IN {var}'
    query_text = re.sub(variable_in_pattern, replace_variable_in, query_text)
    
    # --- Step 7: Replace Map Filters that Use a Variable (unquoted) ---
    # For example, in EXIST( ... (:Substance {pubchem_name: substance}) ... )
    map_var_pattern = r'\{\s*pubchem_name\s*:\s*([a-zA-Z_]\w*)\s*\}'
    query_text = re.sub(map_var_pattern, r'{pubchem_cid: \1}', query_text)
    
    return query_text, all_top_matches


class DesAgent:
    """Description Agent for querying Neo4j graph using LangChain."""
    
    def __init__(self, llm_model_name=None, session_id=None, api_mode="free", user_api_key=None, user_base_url=None):
        """
        Initialize the DesAgent with LLM model and session details.

        Args:
            llm_model_name (str, optional): Name of the language model to use. 
                                          For free mode: choose from OPENROUTER_FREE_MODELS.keys() or None for default.
                                          For user mode: any model supported by the API.
            session_id (str, optional): Session identifier. Defaults to "global".
            api_mode (str): Either "free" for OpenRouter free tier or "user" for user-provided API key.
            user_api_key (str, optional): User's API key when api_mode is "user".
            user_base_url (str, optional): User's base URL when api_mode is "user".
        """
        self.api_mode = api_mode
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url
        
        # Configure API settings based on mode
        if api_mode == "free":
            # Validate model selection for free mode
            if llm_model_name:
                # Check if model is in our known free models list
                if llm_model_name not in OPENROUTER_FREE_MODELS:
                    # If not in our list, check if it's at least formatted correctly as a free model
                    if not llm_model_name.endswith(":free"):
                        raise ValueError(f"Invalid free model '{llm_model_name}'. Model must end with ':free' for free tier usage.")
                    elif "/" not in llm_model_name:
                        raise ValueError(f"Invalid model format '{llm_model_name}'. Expected format: 'provider/model-name:free'")
                    else:
                        # Model format looks correct but not in our list - warn but allow
                        print(f"Warning: Model '{llm_model_name}' not in known free models list. Attempting to use anyway...")
            
            self.llm_model_name = llm_model_name or DEFAULT_FREE_MODEL
            self.base_url = OPENROUTER_BASE_URL
            self.api_key = OPENROUTER_FREE_API_KEY
            if not self.api_key:
                raise ValueError("OpenRouter free API key not found in secrets. Please configure OPENROUTER_FREE_API_KEY.")
        elif api_mode == "user":
            self.llm_model_name = llm_model_name or DEFAULT_MODEL
            self.base_url = user_base_url or DEFAULT_BASE_URL
            self.api_key = user_api_key
            if not self.api_key:
                raise ValueError("User API key is required when api_mode is 'user'.")
        else:
            raise ValueError("api_mode must be either 'free' or 'user'")
            
        self.session_id = session_id or "global"  # fallback
        self.log_dir = "chat_logs"
        self.log_file = f"./{self.log_dir}/chat_log_{self.session_id}.txt"
        
        # Make directory if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.CHAT_HISTORY = ChatMessageHistory()
        self.CHAT_HISTORY_FILE_PATH = "chat_history/chat_history.txt"
        for attempt in range(3):
            try:
                self.graph = Neo4jGraph(
                    url=NEO4J_URI,
                    username=NEO4J_USER,
                    password=NEO4J_PASSWORD,
                    # driver_config={"notifications_min_severity":"WARNING","notifications_disabled_classifications": ["UNRECOGNIZED","DEPRECATED"]},
                    enhanced_schema=True
                )
                self.schema = self.graph.schema
                break
            except Exception as e:
                print(f"[Error initializing Neo4j connection, attempt {attempt + 1}]: {e}")
                self.graph = None
                self.schema = "No schema available due to connection error."

        self.fewshot_examples = FEEWSHOT_EXAMPLES
        self.example_output_prompt = EXAMPLE_OUTPUT_PROMPT
        self.cypher_system_prompt = CYHPER_SYSTEM_PROMPT
        self.answer_system_prompt = ANSWER_SYSTEM_PROMPT
        self.convert_cypher_system_prompt = CONVERT_CYPHER_SYSTEM_PROMPT
        self.schema = f"Schema: {self.schema}"
        
        self.cypher_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.cypher_system_prompt),
                ("system", "{fewshot_examples}"),
                ("system", self.example_output_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", self.schema),
                ("human", "{question}"),
            ]
        )
        
        # Configure LLM clients based on API mode
        llm_kwargs = {
            "model": self.llm_model_name,
            "temperature": 0,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            
        # For OpenRouter, we need to handle JSON mode differently
        if api_mode == "free":
            # OpenRouter free tier may not support JSON mode reliably
            try:
                self.cypher_llm = ChatOpenAI(**llm_kwargs)
                self.answer_llm = ChatOpenAI(**llm_kwargs)
                json_parser = JsonOutputParser()
            except Exception as e:
                if "model" in str(e).lower() and "not found" in str(e).lower():
                    raise ValueError(f"Model '{self.llm_model_name}' not found on OpenRouter. Please check the model name and try again.")
                else:
                    raise e
        else:
            # User's API (likely OpenAI) supports JSON mode
            cypher_kwargs = llm_kwargs.copy()
            cypher_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
            try:
                self.cypher_llm = ChatOpenAI(**cypher_kwargs)
                self.answer_llm = ChatOpenAI(**llm_kwargs)
                json_parser = JsonOutputParser()
            except Exception as e:
                if "model" in str(e).lower() and "not found" in str(e).lower():
                    raise ValueError(f"Model '{self.llm_model_name}' not found. Please check the model name and try again.")
                else:
                    raise e
            
        self.cypher_chain = (
            {"question": itemgetter("question"), "chat_history": itemgetter("chat_history"), "fewshot_examples": itemgetter("fewshot_examples")}
            | self.cypher_agent_prompt
            | self.cypher_llm 
            | json_parser)
        
        self.fix_cypher_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Fix the cypher query based on the error message. Only return the fixed cypher query."),
                ("human", "Question: {question}"),
                ("human", "Schema: {schema}"),
                ("human", "Error message: {error_message}"),
                ("human", "Wrong cypher query: {cypher_query}"),
            ]
        )
        self.fix_cypher_query_chain = (
            {"question": itemgetter("question"), "schema": itemgetter("schema"), "error_message": itemgetter("error_message"), "cypher_query": itemgetter("cypher_query")}
            | self.fix_cypher_query_prompt
            | self.answer_llm  # Use answer_llm for string output
            | StrOutputParser()
        )

        self.convert_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.convert_cypher_system_prompt),
                ("system", "{schema}"),
                ("human", "Cypher Query: {cypher_query}, \nResult: {result}"),
            ]
        )
        self.convert_cypher_chain = (
            {"cypher_query": itemgetter("cypher_query"), "result": itemgetter("result"),"schema": itemgetter("schema")}
            | self.convert_cypher_prompt
            | self.cypher_llm
            | json_parser
        )
        self.cypher_clause_keywords = CYPHER_CLAUSE_KEYWORDS
        self.clause_pattern = CLAUSE_PATTERN

        self.start_session_log()
        
        # Initialize a list to store processed results for each query
        self.processed_results = []

    @classmethod
    def get_available_free_models(cls):
        """
        Get a dictionary of all available free models and their descriptions.
        
        Returns:
            dict: Dictionary mapping model names to descriptions.
        """
        return OPENROUTER_FREE_MODELS.copy()
    
    @classmethod
    def list_free_models(cls):
        """
        Print a formatted list of all available free models.
        """
        print("Available OpenRouter Free Tier Models:")
        print("=" * 50)
        for model, description in OPENROUTER_FREE_MODELS.items():
            print(f"Model: {model}")
            print(f"Description: {description}")
            print("-" * 50)
    
    @classmethod
    def get_models_by_provider(cls, provider=None):
        """
        Get models filtered by provider.
        
        Args:
            provider (str, optional): Provider name (e.g., 'deepseek', 'meta-llama', 'google', 'qwen', 'nvidia').
                                    If None, returns all models grouped by provider.
        
        Returns:
            dict: Models filtered by provider or all models grouped by provider.
        """
        if provider is None:
            # Group all models by provider
            grouped = {}
            for model, desc in OPENROUTER_FREE_MODELS.items():
                model_provider = model.split('/')[0]
                if model_provider not in grouped:
                    grouped[model_provider] = {}
                grouped[model_provider][model] = desc
            return grouped
        else:
            # Filter by specific provider
            return {k: v for k, v in OPENROUTER_FREE_MODELS.items() if k.startswith(f"{provider}/")}
    
    @classmethod
    def get_recommended_models(cls):
        """
        Get a list of recommended models for different use cases.
        
        Returns:
            dict: Dictionary with use cases as keys and recommended models as values.
        """
        return {
            "coding": [
                "deepseek/deepseek-chat-v3-0324:free",
                "deepseek/deepseek-r1:free",
                "deepseek/deepseek-r1-distill-llama-70b:free"
            ],
            "reasoning": [
                "deepseek/deepseek-r1:free",
                "deepseek/deepseek-r1-0528-qwen3-8b:free",
                "qwen/qwq-32b:free"
            ],
            "general_chat": [
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemini-2.0-flash-exp:free",
                "deepseek/deepseek-chat-v3-0324:free"
            ],
            "lightweight": [
                "deepseek/deepseek-r1-distill-qwen-1.5b:free",
                "qwen/qwen3-0.6b-04-28:free",
                "deepseek/deepseek-r1-distill-llama-8b:free"
            ],
            "multimodal": [
                "meta-llama/llama-4-scout:free"
            ]
        }

    def start_session_log(self):
        """
        Start logging the session by recording the start time.
        """
        self.session_log = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "end_time": None,
            "messages": []
        }
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"Session start time: {self.session_log['start_time']}\n")
        except Exception as e:
            print(f"[Error writing to log file]: {e}")

    def log_message(self, role, content):
        """
        Log a message from the user or AI.

        Args:
            role (str): The role of the message sender ('user' or 'ai').
            content (str): The content of the message.
        """
        self.session_log["messages"].append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "role": role,
            "content": content
        })
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"{role.capitalize()}: {content}\n")
        except FileNotFoundError:
            with open(self.log_file, "w", encoding='utf-8') as f:
                f.write(f"Session start time: {self.session_log['start_time']}\n")
                f.write(f"{role.capitalize()}: {content}\n")
        except Exception as e:
            print(f"[Error writing message to log file]: {e}")

    def save_session_log(self, log_filepath=None):
        """
        Save the session log by recording the end time.

        Args:
            log_filepath (str, optional): Path to save the log file. Defaults to self.log_file.
        """
        if not self.session_log["end_time"]:
            self.session_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            final_log_file = log_filepath if log_filepath else self.log_file
            try:
                with open(final_log_file, "a", encoding='utf-8') as f:
                    f.write(f"Session end time: {self.session_log['end_time']}\n\n")
            except Exception as e:
                print(f"[Error writing end time to log file]: {e}")

    def fix_cypher_query(self, cypher_query, error_message, question):
        """
        Attempt to fix a faulty Cypher query based on the error message.

        Args:
            cypher_query (str): The original Cypher query.
            error_message (str): The error message returned from Neo4j.
            question (str): The user's original question.

        Returns:
            str or None: The fixed Cypher query or None if unable to fix.
        """
        try:
            return self.fix_cypher_query_chain.invoke({
                "cypher_query": cypher_query, 
                "error_message": error_message, 
                "question": question, 
                "schema": self.schema
            })
        except Exception as e:
            self.log_message("ai", f"Error while fixing cypher query: {e}")
            return None
    
    def convert_query_result(self, result, result_type="json"):
        """
        Convert the query result to a pandas DataFrame or markdown.
        """
        if result_type == "json":
            return result
        elif result_type == "df":
            df = pd.json_normalize(result)
            # stringify all cells which are not either int, float, or str
            df = df.map(lambda x: str(x) if not isinstance(x, (int, float, str)) else x)
            return df
        elif result_type == "md":
            df = pd.json_normalize(result)
            markdown_table = df.to_markdown(index=False)
            return markdown_table
    def query_graph_with_retry(self, cypher_query, retry_count=3, question=None,result_type="json"):
        """
        Query the Neo4j graph with a retry mechanism.

        Args:
            cypher_query (str): The Cypher query to execute.
            retry_count (int, optional): Number of retry attempts. Defaults to 3.
            question (str, optional): The user's original question.

        Returns:
            Any or None: The query result or None if unsuccessful.
        """
        if not self.graph:
            self.log_message("ai", "No graph connection available.")
            return None
        for i in range(retry_count):
            try:
                self.log_message("ai", f"Cypher query: {cypher_query}")
                self.CHAT_HISTORY.add_ai_message(f"Cypher query: {cypher_query}")
                result = self.graph.query(cypher_query)
                result = self.convert_query_result(result, result_type=result_type)
                return result
            except Exception as e:
                print(f"Error: {e}, please fix the cypher query and try again.")
                self.log_message("ai", f"Error: {e}, please fix the cypher query and try again.")
                self.CHAT_HISTORY.add_ai_message(f"Error: {e}, please fix the cypher query and try again.")
                cypher_query = self.fix_cypher_query(cypher_query, error_message=e, question=question)
                if i == retry_count - 1:
                    return None
        return None
    
    def summarize_dataframe(self, result: pd.DataFrame)->str:
        """
        Summarize the result of a Cypher query.
        """
        if isinstance(result, pd.DataFrame):
            # analyze the size, column names, data type of each column, randomly sample 10 rows
            summary = f"Size: {result.shape}\n"
            summary += f"Columns: {result.columns.tolist()}\n"
            summary += f"Data types: {result.dtypes.to_dict()}\n"
            # Handle case where dataframe has fewer than 10 rows
            sample_size = min(10, len(result))
            summary += f"Sample rows: {result.sample(sample_size).to_markdown() if sample_size > 0 else 'No rows available'}\n"
            return summary
        else:
            return str(result)[:20000] # limit the length of the result to 20000 characters

    
    def cypher_query_to_path(self, cypher_query,result,question):
        """
        Convert the cypher query to focus on showing the graph paths.

        Args:
            cypher_query (str): The original Cypher query.
            result (Any): The result of the cypher query.
            question (str): The user's original question.

        Returns:
            str: The converted Cypher query.
        """
        converted_cypher_query = self.convert_cypher_chain.invoke({"cypher_query": cypher_query, "result": result, "schema": self.schema})
        converted_cypher_query = converted_cypher_query["cypher_query"]
        result = self.query_graph_with_retry(converted_cypher_query, retry_count=3, question=question,result_type="json")

        if result is None:
            msg = "Error: No results found. Please try with another cypher query."
            self.log_message("ai", msg)
            self.CHAT_HISTORY.add_ai_message(msg)
            return msg
        else:
            return result,converted_cypher_query
    
    def parse_query_paths(self,query: str) -> Dict[str, List[str]]:
        """
        Parses the Cypher query to extract the structure of each path in the RETURN clause.
        Returns a dictionary mapping each path variable to a list representing the sequence
        of node labels and relationship types.
        """
        paths = {}
        variable_label_map = {}  # Tracks variable to label mappings

        # Normalize whitespace for consistent regex matching
        normalized_query = ' '.join(query.strip().split())

        # Extract RETURN clause
        return_clause_match = re.search(r'\bRETURN\b\s+(.+)', normalized_query, re.IGNORECASE)
        if not return_clause_match:
            raise ValueError("No RETURN clause found in the query.")

        return_vars = return_clause_match.group(1).split(',')
        return_vars = [var.strip() for var in return_vars]

        # Extract all MATCH clauses
        # This regex captures 'MATCH' or 'OPTIONAL MATCH' followed by anything until the next clause keyword or end of string
        match_clauses = re.findall(
            rf'(?:MATCH|OPTIONAL MATCH)\s+(.*?)(?=\b(?:{CLAUSE_PATTERN})\b|$)',
            normalized_query,
            re.IGNORECASE
        )

        if not match_clauses:
            raise ValueError("No MATCH clauses found in the query.")

        for match_clause in match_clauses:
            # Remove any trailing WHERE clauses or other filters within the MATCH clause
            match_clause_clean = re.split(r'\bWHERE\b', match_clause, flags=re.IGNORECASE)[0].strip()

            # Extract the path variable name and the path pattern
            # Pattern: path_var=pattern or just pattern without assignment
            path_var_match = re.match(r'(\w+)\s*=\s*(.+)', match_clause_clean)
            if not path_var_match:
                # Handle MATCH without assignment, e.g., MATCH (n:Label)-[:REL]->(m:Label)
                # Assign a default variable name
                path_var = f"path_{len(paths)+1}"
                path_pattern = match_clause_clean
            else:
                path_var = path_var_match.group(1)
                path_pattern = path_var_match.group(2)

            # Extract nodes and relationships using regex
            # Nodes are within round brackets ()
            # Relationships are within square brackets []
            node_pattern = r'\(([^()]+)\)'
            relationship_pattern = r'\[([^()\[\]]+)\]'

            # Find all nodes and relationships in order
            nodes = re.findall(node_pattern, path_pattern)
            relationships = re.findall(relationship_pattern, path_pattern)

            if not nodes:
                print(f"Warning: No nodes found in MATCH clause: {match_clause_clean}")
                continue
            if not relationships:
                print(f"Warning: No relationships found in MATCH clause: {match_clause_clean}")
                # It's possible to have MATCH clauses without relationships
                # Handle accordingly if needed

            # Extract node labels by splitting on ':' and taking the second part
            node_labels = []
            for node in nodes:
                parts = node.split(':')
                if len(parts) >= 2:
                    variable = parts[0].strip()
                    label_and_props = parts[1].strip()
                    # Extract label before any space or property
                    label = re.split(r'\s|\{', label_and_props)[0]
                    node_labels.append(label)
                    # Update variable_label_map if variable is present
                    if variable:
                        variable_label_map[variable] = label
                elif len(parts) == 1:
                    # Node without label, possibly just a variable
                    variable = parts[0].strip()
                    label = variable_label_map.get(variable, 'Unknown')  # Retrieve label if exists
                    node_labels.append(label)
                    if variable:
                        variable_label_map[variable] = label
                else:
                    node_labels.append('Unknown')

            # Extract relationship types by splitting on ':' and taking the second part
            rel_types = []
            for rel in relationships:
                parts = rel.split(':')
                if len(parts) >= 2:
                    rel_type = parts[1].strip().split(']')[0]  # Removes any trailing characters if present
                    rel_types.append(rel_type)
                else:
                    rel_types.append('UNKNOWN_RELATIONSHIP')  # Default type if not specified

            # Reconstruct the sequence: node, relationship, node, relationship, ...
            sequence = []
            for i in range(len(rel_types)):
                # Append node label
                if i < len(node_labels):
                    sequence.append(node_labels[i])
                else:
                    sequence.append('Unknown')
                # Append relationship type
                sequence.append(rel_types[i])
            # Append the last node label if exists
            if len(node_labels) > len(rel_types):
                sequence.append(node_labels[-1])

            # Assign the sequence to the path variable
            paths[path_var] = sequence

        # # Debug: Print parsed paths
        # print("\n[DEBUG] Parsed Paths:")
        # for var, seq in paths.items():
        #     print(f"  {var}: {seq}")

        return paths
    # Function to process the Neo4j query results based on the parsed paths
    def process_results(self, paths: Dict[str, List[str]], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes the Neo4j query results dynamically based on the provided path structures.
        Returns a list of processed results with labeled nodes and relationships.
        """
        processed = []
        
        for record in results:
            processed_record = {}
            for path_var, path_structure in paths.items():
                if path_var not in record:
                    continue
                path_data = record[path_var]
                sequence = path_structure
                nodes = []
                relationships = []
                
                # Iterate through the path data
                for index, element in enumerate(path_data):
                    if index % 2 == 0:
                        # Node
                        node_label = sequence[index]
                        node = {'label': node_label, 'properties': element}
                        nodes.append(node)
                    else:
                        # Relationship
                        rel_type = sequence[index]
                        relationship = {'type': rel_type}
                        relationships.append(relationship)
                
                # Combine nodes and relationships
                path_representation = []
                for i in range(len(nodes)):
                    path_representation.append({'n':nodes[i]})
                    if i < len(relationships):
                        path_representation.append({'r':relationships[i]})
                
                processed_record[path_var] = path_representation
            
            processed.append(processed_record)
        
        return processed


    def create_final_result_prompt_template(self, use_cypher, result):
        """
        Create the final prompt template based on query results.

        Args:
            use_cypher (str): Indicates whether a Cypher query was used.
            results (Any): The results from the Cypher query.

        Returns:
            str: The final prompt template.
        """
        template_parts = []

        if use_cypher == "yes" and result:
            template_parts.append("Cypher query result_summary: {result_summary}, this is the summary of the result, the full table is shown to the user.")
        elif use_cypher == "no":
            template_parts.append("No cypher query result needed, answer the question directly.")
        elif use_cypher == "yes" and result is None:
            template_parts.append("No results found, please try with another cypher query.")
        else:
            template_parts.append("There is an error in the cypher query.")
        
        return "\n\n".join(template_parts)

    def task_execution(self, question):
        """
        Execute the task based on the user's question.

        Args:
            question (str): The user's question.

        Yields:
            str: Responses or intermediate steps.
        """
        try:
            result_summary = None
            result = None
            # Log the user's message
            self.log_message("user", question)
            self.CHAT_HISTORY.add_user_message(question)

            # Run the chain to decide if a cypher query is needed
            cypher_response = self.cypher_chain.invoke({
                "question": question,
                "chat_history": self.CHAT_HISTORY.messages,
                "fewshot_examples": self.fewshot_examples
            })
            use_cypher = cypher_response["use_cypher"]

            # Handle Cypher query execution
            if use_cypher == "yes":
                if "cypher_query" in cypher_response:
                    thought_process = cypher_response["thought_process"]
                    # Format thought process to avoid markdown parsing issues
                    msg = f"[Thought Process]\n{thought_process}\n\n"
                    yield msg
                    self.CHAT_HISTORY.add_ai_message(msg)
                    cypher_query = cypher_response["cypher_query"]
                    # cypher_query = cypher_query.replace('{', '{{').replace('}', '}}')

                    msg = f"[Generated Cypher Query]\n{cypher_query}\n\n"
                    yield msg
                    self.CHAT_HISTORY.add_ai_message(msg)
                    self.log_message("ai", f"Generated Cypher Query: {cypher_query}")


                    new_cypher_query, top_matches = transform_cypher_query(cypher_query)
                    # new_cypher_query = new_cypher_query.replace('{', '{{').replace('}', '}}')

                    # Format query display to avoid markdown parsing issues
                    if top_matches:
                        # Display unique top matches by pubchem_cid to avoid duplicates
                        unique_matches = {}
                        for match in top_matches:
                            cid = match['pubchem_cid']
                            if cid not in unique_matches or match['similarity'] > unique_matches[cid]['similarity']:
                                unique_matches[cid] = match
                        
                        # Convert to list and sort by similarity (descending)
                        unique_match_list = list(unique_matches.values())
                        unique_match_list.sort(key=lambda x: x['similarity'], reverse=True)
                        
                        # Display the matches
                        if len(unique_match_list) > 0:
                            yield f"[Found substance name]"
                            for match in unique_match_list:
                                yield f"\n- {match['string']} (CID: {match['pubchem_cid']}, Similarity: {match['similarity']}%)"
                            self.CHAT_HISTORY.add_ai_message(f"Found substance name:\n- {match['string']} (CID: {match['pubchem_cid']}, Similarity: {match['similarity']}%)")
                            self.log_message("ai", f"Found substance name:\n- {match['string']} (CID: {match['pubchem_cid']}, Similarity: {match['similarity']}%)")
                    else:
                        yield f"[No substance matches found]\n"
                        self.CHAT_HISTORY.add_ai_message(f"[No substance matches found]\n")
                        self.log_message("ai", f"[No substance matches found]\n")

                    msg = f"\n\n[New Cypher Query]\n{new_cypher_query}\n\n"
                    yield msg
                    self.log_message("ai", f"New Cypher Query: {new_cypher_query}")
                    self.CHAT_HISTORY.add_ai_message(msg)
                    # temporary fix for <= and >=, replace all ≥ ≤ with >= and <=
                    # new_cypher_query = new_cypher_query.replace("≥", ">=").replace("≤", "<=")
                    
                    result = self.query_graph_with_retry(new_cypher_query, retry_count=3, question=question,result_type="df")
                    if result is None:
                        msg = "Error: No results found. Please try another query."
                        self.log_message("ai", msg)
                        self.CHAT_HISTORY.add_ai_message(msg)
                        yield msg + "\n\n"
                    else:
                        result_summary = self.summarize_dataframe(result)
                        msg = f"[Results found]\n{result_summary}"
                        self.log_message("ai", msg)
                        self.CHAT_HISTORY.add_ai_message(msg)
                        yield result
                else:
                    msg = f"Error: No cypher query found in the response.\n{cypher_response}"
                    self.log_message("ai", msg)
                    self.CHAT_HISTORY.add_ai_message(msg)
                    yield msg + "\n\n"
                    result = None
            else:
                thought_process = cypher_response["thought_process"]
                yield f"[Thought Process]\n{thought_process}\n\n"
                self.CHAT_HISTORY.add_ai_message(f"[Thought Process]\n{thought_process}\n\n")
                self.log_message("ai", f"Thought Process: {thought_process}")
                result = None
                yield f"No cypher query result needed, just answer directly.\n\n"
                self.CHAT_HISTORY.add_ai_message("No cypher query result needed, just answer directly.")
                self.log_message("ai", "No cypher query result needed, just answer directly.")

            # Prepare final response
            final_result_prompt_template = self.create_final_result_prompt_template(use_cypher, result_summary)
            answer_agent_prompt = ChatPromptTemplate.from_messages([
                ("system", self.answer_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", self.schema),
                ("human", final_result_prompt_template),
                ("human", "{question}"),
            ])

            current_chain = {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "result_summary": itemgetter("result_summary")
            }
            answer_agent = (
                current_chain
                | answer_agent_prompt
                | self.answer_llm
                | StrOutputParser()
            )
            chain_with_message_history = RunnableWithMessageHistory(
                answer_agent,
                lambda session_id: self.CHAT_HISTORY,
                input_messages_key='question',
                history_messages_key="chat_history",
            )

            respond_string = ""
            yield f"[Answer]\n"
            for chunk in chain_with_message_history.stream(
                {"question": question, "result_summary": result_summary},
                config={"configurable": {"session_id": self.session_id}}
            ):
                respond_string += chunk
                yield chunk

            # Log AI response
            self.log_message("ai", respond_string)
            self.CHAT_HISTORY.add_ai_message(respond_string)

            # Visualize the graph
            if use_cypher == "yes" and result_summary:
                # Apply the transform_cypher_query function also to the visualization
                visualization_cypher, _ = transform_cypher_query(cypher_query)
                path_result, path_query = self.cypher_query_to_path(visualization_cypher, question, result)
                # print(f"cypher_query: {path_query}")
                yield f"\n\n[Visualize Query]\n{path_query}\n\n"
                path = self.parse_query_paths(path_query)
                processed_result = self.process_results(path, path_result)

                # Store the processed_result for visualization
                self.processed_results.append(processed_result)
            else:
                # Append None to maintain the order
                self.processed_results.append(None)

        except Exception as e:
            error_message = f"An error occurred while processing: {e}"
            self.log_message("ai", error_message)
            self.CHAT_HISTORY.add_ai_message(error_message)
            yield error_message

    def run(self):
        """
        Run the DesAgent, handling user input and responses.
        """
        print("System is ready. You can start asking questions.")
        while True:
            try:
                print("\n")
                question = input("Please enter your question (type 'q', 'quit', or 'exit' to stop): ")
                if question.lower() in ["q", "quit", "exit"]:
                    print("Exiting...")
                    self.save_session_log()
                    break
                for response in self.task_execution(question):
                    print(response, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    def get_latest_processed_result(self):
        """
        Retrieve the latest processed result.

        Returns:
            Dict[str, List[Dict[str, Any]]] or None: The latest processed result or None.
        """
        if self.processed_results:
            return self.processed_results[-1]
        return None

if __name__ == "__main__":
    # Example usage with free mode
    
    # List all available free models
    print("=== Available Free Models ===")
    DesAgent.list_free_models()
    
    # Get recommendations for different use cases
    print("\n=== Recommended Models by Use Case ===")
    recommendations = DesAgent.get_recommended_models()
    for use_case, models in recommendations.items():
        print(f"\n{use_case.title()}:")
        for model in models:
            print(f"  - {model}")
    
    # Example with default free model
    print(f"\n=== Using Default Model ({DEFAULT_FREE_MODEL}) ===")
    agent = DesAgent(api_mode="free")
    
    # Example with specific model selection
    print(f"\n=== Using Specific Model for Reasoning Tasks ===")
    reasoning_agent = DesAgent(
        llm_model_name="deepseek/deepseek-r1:free", 
        api_mode="free"
    )
    
    # Example with coding-focused model
    print(f"\n=== Using Model for Coding Tasks ===")
    coding_agent = DesAgent(
        llm_model_name="deepseek/deepseek-chat-v3-0324:free", 
        api_mode="free"
    )
    
    # Run the agent
    agent.run()
