# agent_streamlit.py
import time
start_time = time.time()
import streamlit as st
import os
import random
from datetime import datetime
from pyvis.network import Network
import streamlit.components.v1 as components
from markdown_it import MarkdownIt
# Import your DesAgent from DesAgent.py
from DesAgent import DesAgent

import re
import pandas as pd
from typing import List, Dict, Any
import traceback



def sanitize_markdown(content):
    """
    Sanitize content for safe markdown rendering in Streamlit.
    Preserves code blocks and Cypher queries with curly braces while
    removing or escaping problematic markdown syntax elsewhere.

    Args:
        content (str): The markdown content to sanitize

    Returns:
        str: Sanitized markdown content
    """
    if content is None:
        return ""

    # Ensure content is a string
    content = str(content)

    # Split content into code blocks and regular text using regex
    code_pattern = r'(```(?:cypher|sql|python|json|)?\s*[\s\S]*?```|~~~\s*[\s\S]*?~~~)'
    parts = re.split(code_pattern, content)

    result_parts = []
    for i, part in enumerate(parts):
        is_code_block = i % 2 == 1  # Code blocks are at odd indices due to re.split capturing groups
        if is_code_block:
            # Preserve code blocks entirely
            result_parts.append(part)
        else:
            # Check if this part looks like a Cypher query
            if re.search(r'^\s*(MATCH|RETURN|WHERE|CREATE|DELETE|MERGE)\b', part, re.IGNORECASE | re.MULTILINE):
                # Wrap standalone Cypher queries in a code block
                wrapped_part = f"```cypher\n{part.strip()}\n```"
                result_parts.append(wrapped_part)
            else:
                # Sanitize non-code, non-Cypher content
                # Remove lines starting with ':::' (directive syntax)
                cleaned = re.sub(r'^:::.+', '', part, flags=re.MULTILINE)
                # Remove any remaining ':::' markers
                cleaned = cleaned.replace(':::', '')
                # Remove heading attributes like '{#id}'
                cleaned = re.sub(r'\{#.*?\}', '', cleaned)
                # Escape special markdown characters
                cleaned = cleaned.replace('\\', '\\\\')
                markdown_special_chars = ['{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '|', '*', '_', '`', '>', '<']
                for char in markdown_special_chars:
                    cleaned = cleaned.replace(char, '\\' + char)
                result_parts.append(cleaned)

    # Recombine all parts into a single string
    return ''.join(result_parts)

def safe_markdown(content):
    try:
        sanitized_content = sanitize_markdown(content)
        print(f"Sanitized content:\n{sanitized_content}")  # Debug output
        st.markdown(sanitized_content)
    except Exception as e:
        st.error(f"Error rendering markdown: {e}")
        st.text(content)

@st.dialog("📚 User Manual")
def show_user_manual_popup():
    """Display the user manual in a popup dialog."""
    # Add custom CSS to make the dialog wider and handle table display
    st.markdown("""
    <style>
    /* Make dialog wider */
    .stDialog > div {
        width: 95vw !important;
        max-width: 1200px !important;
    }
    
    /* Improve table display in dialog */
    .stDialog table {
        width: 100% !important;
        table-layout: auto !important;
        word-wrap: break-word !important;
    }
    
    /* Prevent text wrapping in table cells */
    .stDialog td, .stDialog th {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-width: 100px !important;
    }
    
    /* Allow horizontal scrolling for wide tables */
    .stDialog .stMarkdown {
        overflow-x: auto !important;
    }
    
    /* Better spacing for content */
    .stDialog .stMarkdown > div {
        padding: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    manual_path = os.path.join('docs', 'des_agent_doc_en.md')
    
    if os.path.exists(manual_path):
        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_content = f.read()
            
            # Create a larger scrollable container for the manual content
            with st.container(height=700):
                # Render markdown directly without sanitization for the manual
                st.markdown(manual_content, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading user manual: {e}")
            # Fallback to plain text display
            try:
                with open(manual_path, "r", encoding="utf-8") as f:
                    manual_content = f.read()
                st.text(manual_content)
            except:
                st.error("Could not display manual content.")
    else:
        st.error("User manual file not found at: `docs/des_agent_doc_en.md`")
        st.info("Please ensure the user manual file exists in the docs directory.")
    
    # Add a close button at the bottom
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Close Manual", type="primary", use_container_width=True):
            st.rerun()

@st.dialog("⚙️ Settings")
def show_settings_dialog():
    """Display the settings configuration in a popup dialog."""
    st.markdown("""
    <style>
    .stDialog > div {
        width: 80vw !important;
        max-width: 800px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("API Configuration")
    
    # API Mode Selection
    api_mode = st.radio(
        "Choose API Mode:",
        options=["free", "user"],
        format_func=lambda x: "🆓 Free (OpenRouter)" if x == "free" else "🔑 Your API Key",
        index=0 if st.session_state.api_mode == "free" else 1,
        key="settings_api_mode_radio"
    )
    
    # User API configuration
    user_api_key = ""
    user_base_url = ""
    user_model_name = ""
    
    if api_mode == "user":
        st.markdown("### OpenAI-Compatible API Configuration")
        st.info("💡 Supports OpenAI and OpenAI SDK-compatible services like DeepSeek, Qwen (Alibaba), Together AI, etc.")
        
        user_api_key = st.text_input(
            "API Key:",
            type="password",
            value=st.session_state.user_api_key,
            help="Enter your API key from OpenAI or compatible service (DeepSeek, Qwen, etc.)"
        )
        user_base_url = st.text_input(
            "Base URL:",
            value=st.session_state.user_base_url,
            placeholder="e.g., https://api.deepseek.com",
            help="API endpoint URL. Leave empty for OpenAI (api.openai.com). Examples: DeepSeek: https://api.deepseek.com, Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        user_model_name = st.text_input(
            "Model Name:",
            value=st.session_state.get("user_model_name", ""),
            placeholder="e.g., deepseek-chat, qwen-plus",
            help="Model name to use. Examples: DeepSeek: 'deepseek-chat', Qwen: 'qwen-plus', OpenAI: 'gpt-4o-mini'"
        )
        
        # Examples section
        with st.expander("📋 Service Examples"):
            st.markdown("""
            **DeepSeek:**
            - Base URL: `https://api.deepseek.com`
            - Model: `deepseek-chat`
            
            **Qwen (Alibaba):**
            - Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
            - Model: `qwen-plus` or `qwen-turbo`
            
            **OpenAI:**
            - Base URL: (leave empty)
            - Model: `gpt-4o-mini`, `gpt-4o`, etc.
            """)
        
        if not user_api_key:
            st.warning("⚠️ Please enter your API key to use this mode")
    else:
        st.markdown("### Free OpenRouter Configuration")
        st.info("ℹ️ Using free OpenRouter API")
        
        # Model Selection for Free Mode
        st.markdown("**Choose Model:**")
        
        # Link to OpenRouter free models
        st.markdown(
            "🔗 **[Browse Free Models on OpenRouter](https://openrouter.ai/models?q=free)**"
        )
        st.markdown(
            "💡 Look for models with the `:free` suffix"
        )
        
        # Simple text input for model name
        selected_model = st.text_input(
            "Model Name:",
            value=st.session_state.selected_model or "",
            placeholder="e.g., deepseek/deepseek-chat-v3-0324:free",
            help="Enter the exact model name from OpenRouter (must end with :free)"
        )
        
        # Quick examples
        st.markdown("**Popular free models:**")
        popular_models = [
            "deepseek/deepseek-chat-v3-0324:free",
            "deepseek/deepseek-r1:free", 
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free"
        ]
        
        cols = st.columns(2)
        for i, model in enumerate(popular_models):
            with cols[i % 2]:
                model_display = model.split('/')[1].replace(':free', '')
                if st.button(f"📋 {model_display}", key=f"settings_copy_{model}", use_container_width=True):
                    selected_model = model
                    st.rerun()
        
        # Check if OpenRouter API key is configured
        try:
            openrouter_key = st.secrets.get('OPENROUTER_FREE_API_KEY', None)
            if not openrouter_key or openrouter_key == "your-openrouter-free-api-key-here":
                st.warning("⚠️ OpenRouter API key not configured in secrets.toml")
        except:
            pass
    
    # Check if configuration changed
    config_changed = (api_mode != st.session_state.api_mode or 
                     user_api_key != st.session_state.user_api_key or 
                     user_base_url != st.session_state.user_base_url or
                     (api_mode == "user" and user_model_name != st.session_state.user_model_name) or
                     (api_mode == "free" and selected_model != st.session_state.selected_model))
    
    if config_changed:
        st.markdown("---")
        st.subheader("⚠️ Configuration Changed")
        
        # Show what's changing
        if api_mode != st.session_state.api_mode:
            old_mode = "🆓 Free (OpenRouter)" if st.session_state.api_mode == "free" else "🔑 Your API Key"
            new_mode = "🆓 Free (OpenRouter)" if api_mode == "free" else "🔑 Your API Key"
            st.write(f"**Mode:** {old_mode} → {new_mode}")
        
        if user_api_key != st.session_state.user_api_key:
            if user_api_key:
                st.write(f"**API Key:** Updated")
            else:
                st.write(f"**API Key:** Removed")
        
        if user_base_url != st.session_state.user_base_url:
            st.write(f"**Base URL:** {st.session_state.user_base_url or 'Default'} → {user_base_url or 'Default'}")
        
        if api_mode == "user" and user_model_name != st.session_state.user_model_name:
            old_model = st.session_state.user_model_name or "Default"
            new_model = user_model_name or "Default"
            st.write(f"**Model:** {old_model} → {new_model}")
        
        if api_mode == "free" and selected_model != st.session_state.selected_model:
            old_model = st.session_state.selected_model or "Default"
            if old_model != "Default":
                old_display = old_model.split('/')[1].replace(':free', '')
            else:
                old_display = "Default"
            new_display = selected_model.split('/')[1].replace(':free', '') if selected_model else "Default"
            st.write(f"**Model:** {old_display} → {new_display}")
        
        # Confirmation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("✅ Apply Changes", key="settings_apply_config", type="primary", use_container_width=True):
                # Apply the changes
                st.session_state.api_mode = api_mode
                st.session_state.user_api_key = user_api_key
                st.session_state.user_base_url = user_base_url
                st.session_state.user_model_name = user_model_name if api_mode == "user" else ""
                st.session_state.selected_model = selected_model if api_mode == "free" else None
                st.session_state.api_config_changed = True
                st.rerun()
        
        st.info("Click 'Apply Changes' to update the agent configuration.")
    else:
        # Show current configuration status
        st.markdown("---")
        st.subheader("✅ Current Configuration")
        
        if api_mode == "free":
            st.success(f"**Mode:** 🆓 Free (OpenRouter)")
            if st.session_state.selected_model:
                model_display = st.session_state.selected_model.split('/')[1].replace(':free', '')
                st.info(f"**Model:** {model_display}")
            else:
                st.info("**Model:** Default")
        else:
            st.success(f"**Mode:** 🔑 Your API Key")
            if st.session_state.user_base_url:
                if "deepseek" in st.session_state.user_base_url.lower():
                    service_name = "DeepSeek"
                elif "aliyuncs" in st.session_state.user_base_url.lower():
                    service_name = "Qwen (Alibaba)"
                else:
                    service_name = "Custom API"
                st.info(f"**Service:** {service_name}")
            else:
                st.info("**Service:** OpenAI")
            
            model_name = st.session_state.user_model_name or "Default"
            st.info(f"**Model:** {model_name}")
    
    # Close button
    st.markdown("---")
    if st.button("Close Settings", type="secondary", use_container_width=True):
        st.rerun()

def build_graph_html(processed_records: List[Dict[str, List[Dict[str, Any]]]]) -> str:
    """
    Builds an interactive graph using PyVis based on the processed Neo4j query results.
    This version accepts a list of records, where each record is a dictionary
    (e.g., [{'p1': [...], 'p2': [...]}, {...}, ... ]).

    Args:
        processed_records (List[Dict[str, List[Dict[str, Any]]]]):
            A list of dictionaries, each containing path variables (e.g., 'p1', 'p2') mapped
            to their respective sequences of nodes and relationships.

    Returns:
        str: HTML representation of the interactive graph that combines all records.
    """
    # Initialize a PyVis Network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        filter_menu=False,
        select_menu=False,
        neighborhood_highlight=False
    )

    group_label_map = {
        "Mixture":   {"group": 1, "color": "#FFB3A7"},
        "Substance": {"group": 2, "color": "#B3FFB3"},
        "Article":   {"group": 3, "color": "#B3FFB3"},
        "Unknown":   {"group": 4, "color": "#E0E0E0"}
    }

    added_nodes = set()
    added_edges = set()
    node_id_map = {}

    # ------------------ PASS 1: ADD ALL NODES ------------------
    for record in processed_records:
        for path_var, elements in record.items():
            for element in elements:
                if 'n' in element:
                    node = element['n']
                    label = node.get('label', 'Unknown')
                    properties = node.get('properties', {})

                    # Determine a unique, stable signature for the node
                    # We'll combine the label + key property to form a key in node_id_map.
                    if label == 'Mixture':
                        key_property = properties.get('mixture_id', f"Mixture_{len(added_nodes)+1}")
                    elif label == 'Substance':
                        key_property = properties.get('pubchem_name', f"Substance_{len(added_nodes)+1}")
                    else:
                        # fallback if no recognized label
                        key_property = f"Unknown_{len(added_nodes)+1}"

                    # This signature helps us consistently retrieve the same node ID
                    node_signature = (label, key_property)

                    if node_signature not in node_id_map:
                        # Create a tooltip with properties
                        tooltip = "<br>".join([f"{k}: {v}" for k, v in properties.items()])
                        # Choose a color
                        # color = label_color_map.get(label, label_color_map['Unknown'])

                        # Add the node to the PyVis network
                        net.add_node(
                            key_property,    # node ID
                            label=key_property,
                            title=tooltip,
                            # group=group_label_map.get(label, group_label_map['Unknown'])['group'],
                            color=group_label_map.get(label, group_label_map['Unknown'])['color']
                        )

                        # Mark it as added and remember the signature -> ID mapping
                        node_id_map[node_signature] = key_property
                        added_nodes.add(key_property)

    # ------------------ PASS 2: ADD ALL RELATIONSHIPS ------------------
    for record in processed_records:
        for path_var, elements in record.items():
            for idx, element in enumerate(elements):
                if 'r' in element:
                    relationship = element['r']
                    rel_type = relationship.get('type', 'UNKNOWN_RELATIONSHIP')

                    # We assume the order is: [node, relationship, node, relationship, node, ...]
                    # So if we're at index idx with a relationship, we look at the node before and the node after.
                    if 0 < idx < len(elements) - 1:
                        prev_element = elements[idx - 1]
                        next_element = elements[idx + 1]

                        if 'n' in prev_element and 'n' in next_element:
                            # Extract the label and key property for the source node
                            prev_label = prev_element['n'].get('label', 'Unknown')
                            prev_props = prev_element['n'].get('properties', {})
                            if prev_label == 'Mixture':
                                prev_key = prev_props.get('mixture_id', 'UnknownMixture')
                            elif prev_label == 'Substance':
                                prev_key = prev_props.get('pubchem_name', 'UnknownSubstance')
                            else:
                                prev_key = f"Unknown_{idx-1}"

                            # Extract the label and key property for the target node
                            next_label = next_element['n'].get('label', 'Unknown')
                            next_props = next_element['n'].get('properties', {})
                            if next_label == 'Mixture':
                                next_key = next_props.get('mixture_id', 'UnknownMixture')
                            elif next_label == 'Substance':
                                next_key = next_props.get('pubchem_name', 'UnknownSubstance')
                            else:
                                next_key = f"Unknown_{idx+1}"

                            edge_id = (prev_key, next_key, rel_type)

                            if edge_id not in added_edges:
                                net.add_edge(prev_key, next_key, title=rel_type) # dont show relationship type
                                # net.add_edge(prev_key, next_key, label=rel_type, title=rel_type) # show relationship type
                                added_edges.add(edge_id)

    # Enable physics for a better layout (optional)
    net.toggle_physics(False)
    net.show_buttons(filter_=['physics'])
    net.set_edge_smooth('dynamic')

    # Generate and return the HTML for embedding
    return net.generate_html()


def process_streaming_response(agent_generator, message_placeholder, current_message, response_chunks):
    """
    Helper function to process streaming response with real-time updates.
    Returns tuple of (updated_current_message, collected_dataframes).
    """
    collected_dataframes = []
    
    for chunk in agent_generator:
        if type(chunk) == pd.DataFrame:
            # Collect DataFrames but don't add to session state yet
            collected_dataframes.append(chunk)
            # Display the DataFrame immediately
            st.dataframe(chunk)
        else:
            # Accumulate text chunks
            current_message += str(chunk)
            try:
                # Use the sanitize_markdown function first
                sanitized = sanitize_markdown(current_message)
                message_placeholder.markdown(sanitized)
            except Exception as e:
                try:
                    # First fallback: Try plain text with minimal formatting
                    print(f"Markdown rendering error: {e}")
                    # Strip out all potential problematic markdown syntax
                    strict_content = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\!\?\-\_\*\#\(\)\[\]\>\~\`\=\+\/\\]', '', current_message)
                    # Force plain text display
                    message_placeholder.text(current_message)
                except Exception as e2:
                    # Last resort: Just display raw text
                    print(f"Second markdown rendering error: {e2}")
                    message_placeholder.code(current_message)
            response_chunks.append(chunk)
    
    return current_message, collected_dataframes


def main():

    md = MarkdownIt()
    
    # Header with title and controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("DES Agent")
        st.subheader("Interactive Deep Eutectic Solvent Question Answering Chatbot")
    
    with col2:
        if st.button("📚 User Manual", use_container_width=True):
            show_user_manual_popup()
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            show_settings_dialog()

    # Add custom CSS for text wrapping
    st.markdown("""
    <style>
    .element-container div.markdown-text-container p {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    .element-container div.markdown-text-container code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    .stMarkdown {
        word-wrap: break-word !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    /* Additional selectors to catch all markdown elements */
    .streamlit-expanderContent, .stMarkdown div, .streamlit-expanderContent div {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }
    /* Force all text to be wrappable */
    p, span, div, li, pre, code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }
    /* Handle overflow for all containers */
    .main .block-container, .stTextInput, .stMarkdown, .css-1kyxreq {
        max-width: 100% !important;
        overflow-x: auto !important;
    }
    
    /* Sidebar styling - responsive to theme */
    .css-1d391kg {
        background-color: var(--background-color);
    }
    
    /* Sidebar button styling - theme aware */
    .stSidebar .stButton > button {
        width: 100%;
        margin-bottom: 5px;
        text-align: left;
        border-radius: 8px;
        border: 1px solid var(--secondary-background-color);
        background-color: var(--background-color);
        color: var(--text-color);
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stSidebar .stButton > button:hover {
        background-color: var(--secondary-background-color);
        border-color: var(--primary-color);
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    
    .stSidebar .stButton > button:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* Question button text wrapping */
    .stSidebar .stButton > button div {
        white-space: normal !important;
        text-align: left !important;
        font-size: 14px !important;
        line-height: 1.3 !important;
        padding: 5px !important;
        color: inherit !important;
    }
    
    /* Dark mode specific adjustments */
    @media (prefers-color-scheme: dark) {
        .stSidebar .stButton > button {
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        
        .stSidebar .stButton > button:hover {
            border-color: var(--primary-color);
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        }
    }
    
    /* Streamlit dark theme override */
    [data-testid="stSidebar"] .stButton > button {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--secondary-background-color) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--secondary-background-color) !important;
        border-color: var(--primary-color) !important;
    }
    
    /* Additional dark mode compatibility */
    .stSidebar [data-testid="stExpander"] {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    .stSidebar [data-testid="stExpander"] summary {
        color: var(--text-color) !important;
        font-weight: 600;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Make sure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # --- 1. Initialize Session State Variables ---
    if "session_id" not in st.session_state:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        random_number = random.randint(1000, 9999)
        st.session_state.session_id = f"{current_time}_{random_number}"

    # Initialize API configuration state variables
    if "api_mode" not in st.session_state:
        st.session_state.api_mode = "free"
    if "user_api_key" not in st.session_state:
        st.session_state.user_api_key = ""
    if "user_base_url" not in st.session_state:
        st.session_state.user_base_url = ""
    if "user_model_name" not in st.session_state:
        st.session_state.user_model_name = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "api_config_changed" not in st.session_state:
        st.session_state.api_config_changed = False

    # Initialize or reinitialize agent if configuration changed
    if "agent" not in st.session_state or st.session_state.api_config_changed:
        try:
            if st.session_state.api_mode == "user" and not st.session_state.user_api_key:
                # Don't initialize agent without API key
                if "agent" in st.session_state:
                    del st.session_state.agent
            else:
                # Show initialization message
                if st.session_state.api_config_changed:
                    with st.spinner("🔄 Updating agent configuration..."):
                        time.sleep(0.5)  # Brief pause for UX
                
                # Prepare model parameter based on mode
                model_name = None
                if st.session_state.api_mode == "free" and st.session_state.selected_model:
                    # Strip whitespace and validate for free mode
                    model_name = st.session_state.selected_model.strip()
                    
                    # Only validate if there's actually a model name provided
                    if model_name:
                        # Basic validation for free models
                        if not model_name.endswith(":free"):
                            raise ValueError(f"Model '{model_name}' is not a free model. Free models must end with ':free'")
                        
                        if "/" not in model_name:
                            raise ValueError(f"Invalid model format '{model_name}'. Expected format: 'provider/model-name:free'")
                    else:
                        # If empty string after strip, treat as None
                        model_name = None
                elif st.session_state.api_mode == "user" and st.session_state.user_model_name:
                    # Use user-specified model name for user mode
                    model_name = st.session_state.user_model_name.strip() or None
                
                # Initialize agent with current configuration
                st.session_state.agent = DesAgent(
                    llm_model_name=model_name,
                    session_id=st.session_state.session_id,
                    api_mode=st.session_state.api_mode,
                    user_api_key=st.session_state.user_api_key if st.session_state.api_mode == "user" else None,
                    user_base_url=st.session_state.user_base_url if st.session_state.api_mode == "user" else None
                )
                
                # Show success message and reset flag
                if st.session_state.api_config_changed:
                    st.success("✅ Agent configuration updated successfully!")
                    st.session_state.api_config_changed = False
                    
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "Invalid free model" in error_msg:
                st.error(f"❌ {error_msg}")
                st.info("💡 Please visit the OpenRouter models page and select a model that ends with ':free'")
            elif "is not a free model" in error_msg:
                st.error(f"❌ {error_msg}")
                st.info("💡 Make sure the model name ends with ':free'. Paid models are not supported in free mode.")
            elif "Invalid model format" in error_msg:
                st.error(f"❌ {error_msg}")
                st.info("💡 Example of correct format: 'deepseek/deepseek-chat-v3-0324:free'")
            else:
                st.error(f"❌ Failed to initialize agent: {error_msg}")
            
            if "agent" in st.session_state:
                del st.session_state.agent
            st.session_state.api_config_changed = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize graph visibility state
    if "graph_visibility" not in st.session_state:
        st.session_state.graph_visibility = {}

    # ---- SIDEBAR SECTION - Only Predefined Questions ----
    st.sidebar.title("💡 Quick Questions")
    st.sidebar.markdown("Choose a question to get started:")
    
    # Predefined questions organized by category
    question_categories = {
        "🧪 Basic Concepts": [
            "What is a deep eutectic solvent?",
            "How are deep eutectic solvents formed?",
            "What are the applications of deep eutectic solvents?",
            "What are the properties of deep eutectic solvents?"
        ],
        "🔍 Examples & Components": [
            "Can you give an example of a deep eutectic solvent?",
            "Which substances can form a DES together with urea?",
            "Which substances can form DES with Choline Chloride?"
        ],
        "📊 Database Queries": [
            "What is the DES with the lowest melting point in the database and what are its components and ratios?",
            "Find formulations with melting point in the range [300, 400] K",
            "Find formulations containing Glycerin with melting point in the range [290, 350] K",
            "In the binary system of Sodium Chloride and Calcium Chloride, which formulation has the lowest melting point?"
        ],
        "📚 Research": [
            "Which articles have researched Glycerin?"
        ]
    }
    
    # Initialize the predefined question value in session state if not exists
    if "predefined_question" not in st.session_state:
        st.session_state.predefined_question = ""
    
    if "should_process_predefined" not in st.session_state:
        st.session_state.should_process_predefined = False
    
    # Display questions by category
    for category, questions in question_categories.items():
        with st.sidebar.expander(category, expanded=True):
            for question in questions:
                if st.button(question, key=f"predef_{question}"):
                    st.session_state.predefined_question = question
                    st.session_state.should_process_predefined = True
                    st.rerun()
        
    # ----- MAIN SECTION -----
    end_time = time.time()
    print(f"Time taken for initialization: {end_time - start_time:.2f}s")

    # --- 2. Render existing chat messages from session_state. ---
    # Display messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Display the content
            safe_markdown(message["content"])
            
            # If there's a DataFrame, display it
            if "dataframe" in message:
                st.dataframe(message["dataframe"])

            # Handle graph visualization if present
            if message["role"] == "assistant" and "graph" in message and message["graph"]:
                # Create two columns for Show and Hide Graph buttons
                col1, col2 = st.columns(2)

                with col1:
                    # Show Graph Button
                    show_button_key = f"show_graph_{idx}"
                    if st.button("Show Graph", key=show_button_key):
                        st.session_state.graph_visibility[idx] = True

                with col2:
                    # Hide Graph Button
                    hide_button_key = f"hide_graph_{idx}"
                    if st.button("Hide Graph", key=hide_button_key):
                        st.session_state.graph_visibility[idx] = False

                # Display the graph if the visibility is set to True
                if st.session_state.graph_visibility.get(idx, False):
                    html_content = build_graph_html(message["graph"])
                    components.html(html_content, height=600, width=800, scrolling=True)
                else:
                    # Only add this note when there's a graph capability but it's not shown
                    st.text("Graph is available but hidden. Click 'Show Graph' to view.")

    # --- 3. Chat input box at the bottom for the user to ask a question. ---
    # Regular chat input
    user_input = st.chat_input("Ask the DESAgent something...", key="chat_input")
    
    # Display current configuration status
    if "agent" in st.session_state:
        if st.session_state.api_mode == "free":
            model_display = st.session_state.agent.llm_model_name
            if model_display:
                # Extract just the model name part for cleaner display
                clean_name = model_display.split('/')[-1].replace(':free', '') if '/' in model_display else model_display
                st.info(f"🤖 Using OpenRouter Free Tier: **{clean_name}** (`{model_display}`)")
            else:
                st.info("🤖 Using OpenRouter Free Tier: **Default Model**")
        else:
            # User mode - show more detailed info
            model_name = st.session_state.agent.llm_model_name or "Default"
            base_url = st.session_state.user_base_url
            if base_url:
                # Extract service name from base_url for cleaner display
                if "deepseek" in base_url.lower():
                    service_name = "DeepSeek"
                elif "aliyuncs" in base_url.lower() or "dashscope" in base_url.lower():
                    service_name = "Qwen (Alibaba)"
                elif "together" in base_url.lower():
                    service_name = "Together AI"
                else:
                    service_name = "Custom API"
                st.info(f"🤖 Using {service_name}: **{model_name}**")
            else:
                st.info(f"🤖 Using OpenAI: **{model_name}**")

    # Check if agent is available before processing
    if "agent" not in st.session_state:
        if st.session_state.api_mode == "user" and not st.session_state.user_api_key:
            st.warning("⚠️ Please configure your API key in the settings to start chatting.")
        else:
            st.error("❌ Agent initialization failed. Please check your configuration in settings.")
        return

    # Process predefined question if selected
    if st.session_state.should_process_predefined and st.session_state.predefined_question:
        input_to_process = st.session_state.predefined_question
        st.session_state.predefined_question = ""
        st.session_state.should_process_predefined = False
        
        # Log the user's message to the session
        st.session_state.messages.append({"role": "user", "content": input_to_process})

        # Save user input to file
        user_log_path = f"logs/{st.session_state.session_id}.log"
        os.makedirs("logs", exist_ok=True)
        with open(user_log_path, "a", encoding="utf-8") as f:
            f.write(f"[USER] {input_to_process}\n")

        # Show it in the UI
        with st.chat_message("user"):
            safe_markdown(input_to_process)

        # Stream the assistant's response
        response_chunks = []
        current_message = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show loading animation while waiting for first response
            agent_generator = st.session_state.agent.task_execution(input_to_process)
            
            # Get the first chunk with spinner
            with st.spinner("🤖 Thinking and generating response..."):
                try:
                    first_chunk = next(agent_generator)
                except StopIteration:
                    first_chunk = None
                except Exception as e:
                    error_msg = str(e)
                    if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "invalid" in error_msg.lower()):
                        st.error(f"❌ Model Error: {error_msg}")
                        st.info("💡 Please check your model name and try again. Visit the OpenRouter models page to find valid free models.")
                        return
                    else:
                        st.error(f"❌ Error: {error_msg}")
                        return
            
            # Collect all chunks and DataFrames
            all_dataframes = []
            
            # Process first chunk immediately (spinner ends here)
            if first_chunk is not None:
                if type(first_chunk) == pd.DataFrame:
                    # Collect DataFrame but don't add to session state yet
                    all_dataframes.append(first_chunk)
                    # Display the DataFrame immediately
                    st.dataframe(first_chunk)
                else:
                    # Accumulate text chunks
                    current_message += str(first_chunk)
                    try:
                        # Use the sanitize_markdown function first
                        sanitized = sanitize_markdown(current_message)
                        message_placeholder.markdown(sanitized)
                    except Exception as e:
                        try:
                            # First fallback: Try plain text with minimal formatting
                            print(f"Markdown rendering error: {e}")
                            # Strip out all potential problematic markdown syntax
                            strict_content = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\!\?\-\_\*\#\(\)\[\]\>\~\`\=\+\/\\]', '', current_message)
                            # Force plain text display
                            message_placeholder.text(current_message)
                        except Exception as e2:
                            # Last resort: Just display raw text
                            print(f"Second markdown rendering error: {e2}")
                            message_placeholder.code(current_message)
                    response_chunks.append(first_chunk)
            
            # Now continue streaming remaining chunks in real-time
            current_message, streaming_dataframes = process_streaming_response(agent_generator, message_placeholder, current_message, response_chunks)
            all_dataframes.extend(streaming_dataframes)

            # Save all messages to session state AFTER streaming is complete
            # First save any text content
            if current_message:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": current_message
                })
            
            # Then save any DataFrames as separate messages
            for df in all_dataframes:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "[DataFrame Results]",
                    "dataframe": df
                })

        # Handle graph visualization if present
        processed_result = st.session_state.agent.get_latest_processed_result()
        if processed_result and st.session_state.messages:
            # Find the last assistant message with text content to add graph to
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if (st.session_state.messages[i]["role"] == "assistant" and 
                    "dataframe" not in st.session_state.messages[i]):
                    st.session_state.messages[i]["graph"] = processed_result
                    break

        # Render "Show Graph / Hide Graph" if needed, for the newly generated message
        # Find the message with graph data
        graph_message_idx = None
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if (st.session_state.messages[i]["role"] == "assistant" and 
                "graph" in st.session_state.messages[i]):
                graph_message_idx = i
                break
        
        if graph_message_idx is not None:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Graph", key=f"show_graph_{graph_message_idx}"):
                    st.session_state.graph_visibility[graph_message_idx] = True
            with col2:
                if st.button("Hide Graph", key=f"hide_graph_{graph_message_idx}"):
                    st.session_state.graph_visibility[graph_message_idx] = False

            if st.session_state.graph_visibility.get(graph_message_idx, False):
                html_content = build_graph_html(processed_result)
                components.html(html_content, height=600, width=800, scrolling=True)
            else:
                # Use st.text instead of st.markdown to avoid potential parsing issues
                st.text("Graph is available but hidden. Click 'Show Graph' to view.")
    
    # If there's input from the user
    elif user_input:
        # Process user's own input
        input_to_process = user_input
        
        # Log the user's message to the session
        st.session_state.messages.append({"role": "user", "content": input_to_process})

        # Save user input to file
        user_log_path = f"logs/{st.session_state.session_id}.log"
        os.makedirs("logs", exist_ok=True)
        with open(user_log_path, "a", encoding="utf-8") as f:
            f.write(f"[USER] {input_to_process}\n")

        # Show it in the UI
        with st.chat_message("user"):
            safe_markdown(input_to_process)

        # Stream the assistant's response
        response_chunks = []
        current_message = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show loading animation while waiting for first response
            agent_generator = st.session_state.agent.task_execution(input_to_process)
            
            # Get the first chunk with spinner
            with st.spinner("🤖 Thinking and generating response..."):
                try:
                    first_chunk = next(agent_generator)
                except StopIteration:
                    first_chunk = None
                except Exception as e:
                    error_msg = str(e)
                    if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "invalid" in error_msg.lower()):
                        st.error(f"❌ Model Error: {error_msg}")
                        st.info("💡 Please check your model name and try again. Visit the OpenRouter models page to find valid free models.")
                        return
                    else:
                        st.error(f"❌ Error: {error_msg}")
                        return
            
            # Collect all chunks and DataFrames
            all_dataframes = []
            
            # Process first chunk immediately (spinner ends here)
            if first_chunk is not None:
                if type(first_chunk) == pd.DataFrame:
                    # Collect DataFrame but don't add to session state yet
                    all_dataframes.append(first_chunk)
                    # Display the DataFrame immediately
                    st.dataframe(first_chunk)
                else:
                    # Accumulate text chunks
                    current_message += str(first_chunk)
                    try:
                        # Use the sanitize_markdown function first
                        sanitized = sanitize_markdown(current_message)
                        message_placeholder.markdown(sanitized)
                    except Exception as e:
                        try:
                            # First fallback: Try plain text with minimal formatting
                            print(f"Markdown rendering error: {e}")
                            # Strip out all potential problematic markdown syntax
                            strict_content = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\!\?\-\_\*\#\(\)\[\]\>\~\`\=\+\/\\]', '', current_message)
                            # Force plain text display
                            message_placeholder.text(current_message)
                        except Exception as e2:
                            # Last resort: Just display raw text
                            print(f"Second markdown rendering error: {e2}")
                            message_placeholder.code(current_message)
                    response_chunks.append(first_chunk)
            
            # Now continue streaming remaining chunks in real-time
            current_message, streaming_dataframes = process_streaming_response(agent_generator, message_placeholder, current_message, response_chunks)
            all_dataframes.extend(streaming_dataframes)

            # Save all messages to session state AFTER streaming is complete
            # First save any text content
            if current_message:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": current_message
                })
            
            # Then save any DataFrames as separate messages
            for df in all_dataframes:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "[DataFrame Results]",
                    "dataframe": df
                })

        # Handle graph visualization if present
        processed_result = st.session_state.agent.get_latest_processed_result()
        
        # Save any final response chunks to file
        final_response = "".join(str(chunk) for chunk in response_chunks)
        with open(user_log_path, "a", encoding="utf-8") as f:
            f.write(f"[ASSISTANT] \n{final_response}\n\n")
        
        # Update the last text message with graph if available
        if processed_result and st.session_state.messages:
            # Find the last assistant message with text content to add graph to
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if (st.session_state.messages[i]["role"] == "assistant" and 
                    "dataframe" not in st.session_state.messages[i]):
                    st.session_state.messages[i]["graph"] = processed_result
                    break

        # Render "Show Graph / Hide Graph" if needed, for the newly generated message
        # Find the message with graph data
        graph_message_idx = None
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if (st.session_state.messages[i]["role"] == "assistant" and 
                "graph" in st.session_state.messages[i]):
                graph_message_idx = i
                break
        
        if graph_message_idx is not None:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Graph", key=f"show_graph_{graph_message_idx}"):
                    st.session_state.graph_visibility[graph_message_idx] = True
            with col2:
                if st.button("Hide Graph", key=f"hide_graph_{graph_message_idx}"):
                    st.session_state.graph_visibility[graph_message_idx] = False

            if st.session_state.graph_visibility.get(graph_message_idx, False):
                html_content = build_graph_html(processed_result)
                components.html(html_content, height=600, width=800, scrolling=True)
            else:
                # Use st.text instead of st.markdown to avoid potential parsing issues
                st.text("Graph is available but hidden. Click 'Show Graph' to view.")

    # Note: The graph visualization is handled in the messages loop above.
    

if __name__ == "__main__":
    main()
