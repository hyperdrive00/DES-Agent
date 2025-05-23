# DES-Agent

Interactive Deep Eutectic Solvent Question Answering Chatbot

DES-Agent is built with LangChain and Neo4j. It allows you to ask questions about
Deep Eutectic Solvents (DES) using natural language. The agent automatically
creates Cypher queries, executes them on your Neo4j database and streams the
results. Typical tasks include finding which substances form a DES, querying
melting points of mixtures or retrieving articles that mention a compound.

## Features
- Natural language querying without writing Cypher
- Structured storage and retrieval of articles, mixtures and substances
- Multi-turn conversations with context
- Example questions include "Which substances can form a DES with urea?" or
  "What is the DES with the lowest melting point?"

See the `docs/des_agent_doc_en.md` manual for more detail on the schema and
examples.

## Installation
1. Create a Python environment (Python 3.9 or later).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide credentials for your Neo4j instance by creating a file
   `.streamlit/secrets.toml` in the project root:
   ```toml
   [neo4j_credentials]
   NEO4J_URI="bolt://localhost:7687"
   NEO4J_USER="neo4j"
   NEO4J_PASSWORD="password"
   ```
4. Set your `OPENAI_API_KEY` environment variable so LangChain can access the
   OpenAI model.

## Usage
### Command line
Run the agent in your terminal:
```bash
python DesAgent.py
```
Enter a question at the prompt or type `quit` to exit.

### Streamlit demo
Launch a web interface with predefined questions:
```bash
streamlit run agent_streamlit.py
```
Open the displayed URL in your browser to chat with the agent and view graph
visualisations of query results.

## Documentation
A detailed user manual with schema description, typical Q&A and workflow
information is available in `docs/des_agent_doc_en.md`.

