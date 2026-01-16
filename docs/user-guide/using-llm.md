# Using LLM Integration

Proxima integrates with Large Language Models (LLMs) to provide intelligent assistance for quantum circuit planning, result interpretation, and workflow optimization.

## Overview

Proxima's LLM integration supports:

- **Execution Planning**: Automatically generate optimal execution strategies
- **Result Summarization**: Get human-readable insights from complex results
- **Circuit Optimization**: Receive suggestions for circuit improvements
- **Configuration Assistance**: Get help with backend selection and tuning
- **Troubleshooting**: Intelligent error diagnosis and resolution

## Supported LLM Providers

| Provider | Models | Type | Setup |
|----------|--------|------|-------|
| OpenAI | GPT-4, GPT-4-turbo, GPT-3.5 | Cloud | API key required |
| Anthropic | Claude 3, Claude 2 | Cloud | API key required |
| Ollama | Llama 2, Mistral, CodeLlama | Local | Local installation |
| LM Studio | Various open models | Local | Local installation |

## Configuration

### Setting Up API Keys

#### Environment Variables (Recommended)

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# For local Ollama
export OLLAMA_HOST="http://localhost:11434"
```

#### Using Proxima Config

```bash
# Set provider
proxima config set llm.provider openai

# Set model
proxima config set llm.model gpt-4

# For local providers
proxima config set llm.provider ollama
proxima config set llm.local_endpoint http://localhost:11434

# View LLM configuration
proxima config show llm
```

### Configuration File

In your `proxima.yaml`:

```yaml
llm:
  provider: openai          # openai, anthropic, ollama, lmstudio
  model: gpt-4              # Model name
  temperature: 0.7          # Creativity (0.0-1.0)
  max_tokens: 2000          # Max response length
  require_consent: true     # Ask before LLM requests
  local_endpoint: null      # For local providers

  # Provider-specific settings
  openai:
    organization: null      # Optional org ID
  
  anthropic:
    max_retries: 3
  
  ollama:
    context_window: 4096
```

## Using LLM Commands

### Planning Execution

Generate an intelligent execution plan for complex circuits:

```bash
# Basic planning
proxima llm plan circuits/complex_vqe.json

# Plan with specific constraints
proxima llm plan circuits/complex_vqe.json --max-time 600 --prefer-gpu

# Interactive planning mode
proxima llm plan circuits/complex_vqe.json --interactive
```

**Example Output:**

```

                   Execution Plan                         

  Circuit: VQE Hydrogen Optimization                     
  Qubits: 4                                              
  Depth: 23                                              

  Recommended Strategy:                                  
                                                         
  1. Use cuQuantum backend for GPU acceleration          
     - Reason: Circuit depth and parameter count suit    
       GPU parallelization                               
                                                         
  2. Configure 10,000 shots per iteration                
     - Reason: VQE requires high statistical accuracy    
                                                         
  3. Enable checkpoint every 5 iterations                
     - Reason: Long optimization may benefit from        
       recovery points                                   
                                                         
  4. Memory estimate: ~2.1 GB                            
     - Status: Within available GPU memory               

  [y] Accept plan  [n] Modify  [i] Get more info         

```

### Summarizing Results

Get human-readable insights from execution results:

```bash
# Summarize latest result
proxima llm summarize results/latest.json

# Summarize with comparison focus
proxima llm summarize results/comparison_report.json --focus comparison

# Summarize with statistical analysis
proxima llm summarize results/vqe_run.json --focus statistics

# Export summary to file
proxima llm summarize results/latest.json --output summary.md
```

**Example Output:**

```

                  Result Summary                          

  Key Findings:                                          
                                                         
   The Bell state circuit produced expected             
    entanglement with high fidelity (F = 0.997)          
                                                         
   Distribution of |00 and |11 states is within       
    1% of theoretical 50/50 split                        
                                                         
   Backend performance comparison:                       
    - cuQuantum: 45ms (fastest)                          
    - Cirq: 152ms                                        
    - Qiskit: 168ms                                      
                                                         
  Recommendations:                                        
                                                         
   For this circuit type, cuQuantum provides 3x         
    speedup with equivalent accuracy                     
                                                         
   Consider using cuQuantum for production runs         
    if GPU is available                                  

```

### Circuit Optimization Suggestions

Get suggestions for improving your circuits:

```bash
# Analyze and suggest optimizations
proxima llm optimize circuits/my_circuit.json

# Focus on specific optimization goals
proxima llm optimize circuits/my_circuit.json --goal depth
proxima llm optimize circuits/my_circuit.json --goal gates
proxima llm optimize circuits/my_circuit.json --goal noise-resistance
```

### Configuration Assistance

Get help choosing and configuring backends:

```bash
# Get backend recommendation
proxima llm recommend-backend circuits/complex.json

# Get configuration suggestions
proxima llm configure --circuit circuits/complex.json --goal performance
```

## Consent and Safety

Proxima implements consent checks to ensure you're aware when LLM requests are made:

### Consent Modes

```yaml
# In proxima.yaml
llm:
  require_consent: true       # Ask every time (most secure)
  
consent:
  auto_approve_local_llm: false   # Auto-approve local LLM
  auto_approve_remote_llm: false  # Auto-approve cloud LLM
  remember_decisions: false        # Remember consent choices
```

### Consent Prompts

When `require_consent: true`, you'll see:

```

                   LLM Request Consent                    

  Action: Send circuit analysis to OpenAI GPT-4          
                                                         
  Data to be sent:                                       
     Circuit structure (4 qubits, 15 gates)             
     Requested analysis type                            
                                                         
   This data will be sent to external servers           

  [y] Approve  [n] Deny  [a] Always approve for session  

```

### Privacy Best Practices

1. **Use local LLMs** for sensitive data
2. **Review data** before sending to cloud providers
3. **Enable consent** in production environments
4. **Avoid sending** API keys or credentials in prompts

## Local LLM Setup

### Using Ollama

1. **Install Ollama:**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows
   # Download from https://ollama.com/download
   ```

2. **Pull a model:**
   ```bash
   ollama pull llama2
   ollama pull codellama  # For code-focused tasks
   ollama pull mistral    # Good balance of speed/quality
   ```

3. **Configure Proxima:**
   ```bash
   proxima config set llm.provider ollama
   proxima config set llm.model llama2
   proxima config set llm.local_endpoint http://localhost:11434
   ```

### Using LM Studio

1. **Install LM Studio** from https://lmstudio.ai/

2. **Download a model** (e.g., Mistral-7B, Llama-2-13B)

3. **Start the local server** in LM Studio

4. **Configure Proxima:**
   ```bash
   proxima config set llm.provider lmstudio
   proxima config set llm.local_endpoint http://localhost:1234/v1
   ```

## Programmatic Usage

### Python API

```python
from proxima.intelligence import LLMRouter, InsightEngine

# Initialize LLM router
router = LLMRouter()

# Get execution plan
plan = await router.generate_plan(
    circuit="circuits/complex.json",
    constraints={"max_time": 600, "prefer_gpu": True}
)

# Summarize results
insights = InsightEngine(router)
summary = await insights.summarize(
    results="results/latest.json",
    focus="comparison"
)
print(summary.text)
```

### Streaming Responses

```python
async for chunk in router.stream_response(prompt):
    print(chunk, end="", flush=True)
```

## Troubleshooting

### Common Issues

**API Key Errors:**
```bash
# Verify API key is set
proxima config get llm.api_key

# Test connection
proxima llm test
```

**Local LLM Not Responding:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check LM Studio server
curl http://localhost:1234/v1/models
```

**Rate Limiting:**
```yaml
# In proxima.yaml - add delays between requests
llm:
  rate_limit_delay: 1.0  # seconds between requests
```

**Token Limits:**
```yaml
# Reduce max tokens for faster responses
llm:
  max_tokens: 1000
```

## Cost Management

### Token Usage Tracking

```bash
# View LLM usage statistics
proxima llm usage

# View cost estimate
proxima llm usage --show-cost
```

### Cost Optimization Tips

1. **Use local models** for development and testing
2. **Cache results** to avoid duplicate requests
3. **Use smaller models** (gpt-3.5-turbo) for simple tasks
4. **Set token limits** appropriate to your needs
5. **Enable response caching** in config

```yaml
llm:
  cache_responses: true
  cache_ttl: 3600  # 1 hour
```

## See Also

- [Configuration Guide](../getting-started/configuration.md)
- [Agent Files](agent-files.md)
- [Advanced Topics](advanced-topics.md)
- [LLM Router API Reference](../api-reference/llm-router.md)
