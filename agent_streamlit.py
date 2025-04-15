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


def main():

    md = MarkdownIt()
    st.title("DES Agent")
    st.subheader("Interactive Deep Eutectic Solvent Question Answering Chatbot")

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
    </style>
    """, unsafe_allow_html=True)

    # Make sure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # --- 1. Handle Session State for session_id, DESAgent, and Messages. ---
    if "session_id" not in st.session_state:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        random_number = random.randint(1000, 9999)
        st.session_state.session_id = f"{current_time}_{random_number}"

    if "agent" not in st.session_state:
        st.session_state.agent = DesAgent(session_id=st.session_state.session_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize graph visibility state
    if "graph_visibility" not in st.session_state:
        st.session_state.graph_visibility = {}

    # 1. Initialize the user manual visibility in session_state if not set
    if "show_user_manual" not in st.session_state:
        st.session_state.show_user_manual = False

    # ---- SIDEBAR SECTION ----
    # Move manual controls to sidebar
    st.sidebar.title("Options")
    
    # Add manual controls to sidebar
    if st.sidebar.button("Show Manual"):
        st.session_state.show_user_manual = True
    if st.sidebar.button("Hide Manual"):
        st.session_state.show_user_manual = False
    
    # Add predefined questions to sidebar
    st.sidebar.subheader("Predefined Questions")
    predefined_questions = [
        "What is a deep eutectic solvent?",
        "How are deep eutectic solvents formed?",
        "What are the applications of deep eutectic solvents?",
        "Can you give an example of a deep eutectic solvent?",
        "What are the properties of deep eutectic solvents?",
        "Which substances can form a DES together with urea?",
        "Which substances can form DES with Choline Chloride?",
        "What is the DES with the lowest melting point in the database and what are its components and ratios?",
        "Find formulations with melting point in the range [300, 400] K",
        "Find formulations containing Glycerin with melting point in the range [290, 350] K",
#        "Find formulations containing Sodium Chloride and Calcium Chloride with melting point in range [400, 600] K",
        # "What is the formulation with the lowest melting point in the database?",
        "Which articles have researched Glycerin?",
        "In the binary system of Sodium Chloride and Calcium Chloride, which formulation has the lowest melting point?"
    ]
    
    # Initialize the predefined question value in session state if not exists
    if "predefined_question" not in st.session_state:
        st.session_state.predefined_question = ""
    
    if "should_process_predefined" not in st.session_state:
        st.session_state.should_process_predefined = False
    
    # Predefined question buttons in sidebar
    for question in predefined_questions:
        if st.sidebar.button(question, key=f"predef_{question}"):
            st.session_state.predefined_question = question
            st.session_state.should_process_predefined = True
            st.rerun()
        
    # ----- MAIN SECTION -----
    # 3. Conditionally render the user manual
    if st.session_state.show_user_manual:
        # Load user manual content from an external Markdown file
        manual_path = os.path.join('docs', 'des_agent_doc_en.md')
        if os.path.exists(manual_path):
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_content = f.read()
            # Safely render the manual content
            safe_markdown(manual_content)
        else:
            st.error("User manual file not found.")

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
            for chunk in st.session_state.agent.task_execution(input_to_process):
                if type(chunk) == pd.DataFrame:
                    # If we have accumulated any text, save it as a message
                    if current_message:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": current_message
                        })
                    # Save and display the DataFrame
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "[DataFrame Results]",
                        "dataframe": chunk
                    })
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

            # Save any remaining text as a message AFTER the loop completes
            if current_message:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": current_message
                })

        # Handle graph visualization if present
        processed_result = st.session_state.agent.get_latest_processed_result()
        if processed_result:
            st.session_state.messages[-1]["graph"] = processed_result

        # Render "Show Graph / Hide Graph" if needed, for the newly generated message
        if st.session_state.messages[-1]["role"] == "assistant" and "graph" in st.session_state.messages[-1]:
            idx = len(st.session_state.messages) - 1
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Graph", key=f"show_graph_{idx}"):
                    st.session_state.graph_visibility[idx] = True
            with col2:
                if st.button("Hide Graph", key=f"hide_graph_{idx}"):
                    st.session_state.graph_visibility[idx] = False

            if st.session_state.graph_visibility.get(idx, False):
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
            for chunk in st.session_state.agent.task_execution(input_to_process):
                if type(chunk) == pd.DataFrame:
                    # If we have accumulated any text, save it as a message
                    if current_message:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": current_message
                        })
                        current_message = ""
                    # Save and display the DataFrame
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "[DataFrame Results]",
                        "dataframe": chunk
                    })
                    st.dataframe(chunk)
                else:
                    # Accumulate text chunks
                    current_message += str(chunk)
                    try:
                        # Use the sanitize_markdown function f  irst
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

            # Save any remaining text as a message AFTER the loop completes
            if current_message:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": current_message
                })

        # Handle graph visualization if present
        processed_result = st.session_state.agent.get_latest_processed_result()
        
        # Save any final response chunks to file
        final_response = "".join(response_chunks)
        with open(user_log_path, "a", encoding="utf-8") as f:
            f.write(f"[ASSISTANT] \n{final_response}\n\n")
        
        # Update the last message with graph if available
        if processed_result and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages[-1]["graph"] = processed_result

        # Render "Show Graph / Hide Graph" if needed, for the newly generated message
        if st.session_state.messages[-1]["role"] == "assistant" and "graph" in st.session_state.messages[-1]:
            idx = len(st.session_state.messages) - 1
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Graph", key=f"show_graph_{idx}"):
                    st.session_state.graph_visibility[idx] = True
            with col2:
                if st.button("Hide Graph", key=f"hide_graph_{idx}"):
                    st.session_state.graph_visibility[idx] = False

            if st.session_state.graph_visibility.get(idx, False):
                html_content = build_graph_html(processed_result)
                components.html(html_content, height=600, width=800, scrolling=True)
            else:
                # Use st.text instead of st.markdown to avoid potential parsing issues
                st.text("Graph is available but hidden. Click 'Show Graph' to view.")

    # Note: The graph visualization is handled in the messages loop above.
    

if __name__ == "__main__":
    main()
