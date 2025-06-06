# DES Agent

Interactive Deep Eutectic Solvent Question Answering Chatbot

## API Configuration

The DES Agent now supports two API modes:

### 1. Free Mode (OpenRouter)
- Uses OpenRouter's free tier with Llama 3.1 8B model
- Limited usage but no cost
- Requires OpenRouter API key in `.streamlit/secrets.toml`

### 2. User API Mode
- Use your own OpenAI API key or compatible API
- Full access to premium models like GPT-4
- Enter your API key in the sidebar

## Setup

1. Configure your API keys in `.streamlit/secrets.toml`:
```toml
# For free mode
OPENROUTER_FREE_API_KEY = "your-openrouter-api-key"

# Neo4j configuration (required)
[neo4j_credentials]
NEO4J_URI = "your-neo4j-uri"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-password"
```

2. Run the application:
```bash
streamlit run agent_streamlit.py
```

## Features

- Interactive chat interface
- Graph database querying with Cypher
- Visual graph exploration
- Support for multiple API providers
- Session management and logging
