# DES Agent

Interactive Deep Eutectic Solvent Question Answering Chatbot

## API Configuration

The DES Agent now supports two API modes with extensive model selection options:

### 1. Free Mode (OpenRouter)
- Uses OpenRouter's free tier with multiple high-quality models
- Limited usage but no cost
- Requires OpenRouter API key in `.streamlit/secrets.toml`
- **Choose from 17+ free models** including DeepSeek, Llama, Gemini, and Qwen models

### 2. User API Mode
- Use your own OpenAI API key or compatible API
- Full access to premium models like GPT-4
- Enter your API key in the sidebar

## Available Free Models

The DES Agent provides access to a comprehensive selection of free tier models:

### DeepSeek Models (Recommended for Coding & Reasoning)
- `deepseek/deepseek-chat-v3-0324:free` - Excellent for coding and general tasks
- `deepseek/deepseek-r1:free` - Advanced reasoning model with open reasoning tokens
- `deepseek/deepseek-r1-zero:free` - RL-trained model without SFT
- `deepseek/deepseek-r1-0528-qwen3-8b:free` - Latest reasoning model
- `deepseek/deepseek-r1-distill-llama-70b:free` - High performance distilled model
- `deepseek/deepseek-r1-distill-qwen-32b:free` - Strong reasoning capabilities
- `deepseek/deepseek-r1-distill-qwen-14b:free` - Balanced performance
- `deepseek/deepseek-r1-distill-qwen-7b:free` - Efficient reasoning model
- `deepseek/deepseek-r1-distill-qwen-1.5b:free` - Lightweight yet capable
- `deepseek/deepseek-r1-distill-llama-8b:free` - Good balance of size and performance
- `deepseek/deepseek-v3-base:free` - 671B parameter base model

### Meta Llama Models
- `meta-llama/llama-4-scout:free` - Multimodal MoE model with 200K context
- `meta-llama/llama-3.3-70b-instruct:free` - High-quality instruction following

### Google Models
- `google/gemini-2.5-pro-exp-03-25:free` - Latest Google model
- `google/gemini-2.0-flash-thinking-exp:free` - Fast reasoning model
- `google/gemini-2.0-flash-exp:free` - Quick responses
- `google/gemma-3-27b-it:free` - Google's open model

### Qwen Models
- `qwen/qwen3-4b:free` - Dual-mode architecture with 128K context
- `qwen/qwen3-0.6b-04-28:free` - Lightweight model with 32K context
- `qwen/qwq-32b:free` - Specialized reasoning model

### NVIDIA Models
- `nvidia/llama-3.1-nemotron-ultra-253b-v1:free` - NVIDIA's flagship model

## How to Choose a Model

### 🔗 Browse Available Models
Visit **[OpenRouter's Free Models Page](https://openrouter.ai/models?q=free)** to see all currently available free models. Look for models with the `:free` suffix.

### 📝 Model Selection in the App
1. In the sidebar, you'll find a text input box under "Model Name"
2. Enter the exact model name from OpenRouter (must end with `:free`)
3. Use the quick copy buttons for popular models, or browse the OpenRouter page for more options

### 🎯 Popular Recommendations

#### For Coding Tasks
- `deepseek/deepseek-chat-v3-0324:free` - Excellent for programming
- `deepseek/deepseek-r1:free` - Advanced reasoning for complex coding problems

#### For Reasoning Tasks  
- `deepseek/deepseek-r1:free` - Best reasoning capabilities
- `deepseek/deepseek-r1-0528-qwen3-8b:free` - Latest reasoning model

#### For General Chat
- `meta-llama/llama-3.3-70b-instruct:free` - Great instruction following
- `google/gemini-2.0-flash-exp:free` - Fast responses

#### For Lightweight/Fast Responses
- `deepseek/deepseek-r1-distill-qwen-1.5b:free` - Smallest but capable
- `qwen/qwen3-0.6b-04-28:free` - Very lightweight option

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

## Usage Examples

### Basic Usage (Default Model)
```python
from DesAgent import DesAgent

# Use default free model
agent = DesAgent(api_mode="free")
```

### Selecting a Specific Model
```python
# For coding tasks
coding_agent = DesAgent(
    llm_model_name="deepseek/deepseek-chat-v3-0324:free",
    api_mode="free"
)

# For reasoning tasks
reasoning_agent = DesAgent(
    llm_model_name="deepseek/deepseek-r1:free",
    api_mode="free"
)

# For general chat
chat_agent = DesAgent(
    llm_model_name="meta-llama/llama-3.3-70b-instruct:free",
    api_mode="free"
)
```

### Discovering Available Models
```python
# List all available models
DesAgent.list_free_models()

# Get models by provider
deepseek_models = DesAgent.get_models_by_provider("deepseek")

# Get recommendations by use case
recommendations = DesAgent.get_recommended_models()
```

## Features

- Interactive chat interface
- **17+ free tier models** from leading AI providers
- **Smart model recommendations** for different use cases
- Graph database querying with Cypher
- Visual graph exploration
- Support for multiple API providers
- Session management and logging
- **Model validation and error handling**

## Important Notes

### Privacy Considerations for Free Models
When using OpenRouter's free tier models, your data may be used for model training and improvement by the underlying providers. **Do not send sensitive or confidential information** when using free models.

### Rate Limits
- Default: 20 requests/minute, 50 requests/day
- **Recommended**: Add $10 credit to your OpenRouter account for increased limits (1,000 requests/day)
- Credits are not consumed when using free models but unlock higher rate limits

### Model Availability
Free model availability may change over time. The agent includes validation to ensure selected models are available and will provide helpful error messages if a model is unavailable.
