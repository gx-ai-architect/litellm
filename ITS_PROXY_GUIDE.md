# ITS-Hub Integration with LiteLLM Proxy

This guide shows how to use Inference-Time Scaling (ITS) algorithms with the LiteLLM Proxy server.

## Overview

The ITS-Hub integration supports two configuration methods:
1. **Header-based** - Pass ITS parameters via HTTP headers (highest priority)
2. **YAML-based** - Configure per-deployment defaults in proxy config file

### Supported Algorithms
- **Self-Consistency** - Generate N responses and select the most common answer
- **Best-of-N** - Generate N responses and use an LLM judge to select the best

## Method 1: Header-Based Configuration

Pass ITS parameters via `X-ITS-*` headers. This works with any model and overrides YAML config.

### Available Headers

| Header | Type | Description |
|--------|------|-------------|
| `X-ITS-Algorithm` | string | Algorithm: `self-consistency` or `best-of-n` |
| `X-ITS-Budget` | integer | Number of generations |
| `X-ITS-Judge-Model` | string | Model to use for judging (best-of-n only) |
| `X-ITS-Judge-Prompt` | string | Custom judge prompt template |
| `X-ITS-Judge-Temperature` | float | Temperature for judge model |
| `X-ITS-Judge-Fallback-Score` | float | Fallback score if parsing fails |

### Examples

#### Self-Consistency
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -H "X-ITS-Algorithm: self-consistency" \
  -H "X-ITS-Budget: 5" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 15 * 24?"}],
    "temperature": 0.7
  }'
```

#### Best-of-N with Custom Judge
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -H "X-ITS-Algorithm: best-of-n" \
  -H "X-ITS-Budget: 3" \
  -H "X-ITS-Judge-Model: gpt-4o-mini" \
  -H "X-ITS-Judge-Temperature: 0.3" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a sorting function"}]
  }'
```

#### Python Client
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-1234"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_headers={
        "X-ITS-Algorithm": "self-consistency",
        "X-ITS-Budget": "5"
    },
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Method 2: YAML-Based Configuration

Configure per-deployment ITS defaults in your proxy config file.

### Config Example

```yaml
model_list:
  # Standard model without ITS
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # Model with self-consistency by default
  - model_name: gpt-4o-mini-sc
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
    its_params:
      algorithm: self-consistency
      budget: 5

  # Model with best-of-n and custom judge
  - model_name: gpt-4o-bon
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    its_params:
      algorithm: best-of-n
      budget: 3
      judge_model: gpt-4o-mini
      judge_temperature: 0.3
      judge_fallback_score: 5.0
```

### Usage

```bash
# Use model with pre-configured ITS params
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o-mini-sc",
    "messages": [{"role": "user", "content": "What is 15 * 24?"}]
  }'
# Automatically uses self-consistency with budget=5
```

```python
# Python client
response = client.chat.completions.create(
    model="gpt-4o-bon",
    messages=[{"role": "user", "content": "Explain linked lists"}]
)
# Automatically uses best-of-n with gpt-4o-mini judge
```

## Priority Order

When ITS parameters come from multiple sources, priority is:

1. **Headers** (highest) - Always override everything
2. **Request body** - Standard parameters
3. **YAML `its_params`** (lowest) - Per-deployment defaults

### Example: Override YAML Config

```bash
# Model gpt-4o-mini-sc has budget=5 in YAML
# Override with header
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -H "X-ITS-Budget: 10" \
  -d '{
    "model": "gpt-4o-mini-sc",
    "messages": [{"role": "user", "content": "What is 10 * 10?"}]
  }'
# Uses budget=10 instead of 5
```

## Cost Optimization

### Separate Judge Model

Use a cheaper model for judging to reduce costs:

```yaml
- model_name: gpt-4o-cost-optimized
  litellm_params:
    model: openai/gpt-4o
  its_params:
    algorithm: best-of-n
    budget: 5
    judge_model: gpt-4o-mini  # Much cheaper!
```

**Cost Analysis:**
- Generate 5 with `gpt-4o`: 5 × $X
- Judge 5 with `gpt-4o-mini`: 5 × $Y (where Y ≪ X)
- **Total:** 5× gpt-4o + 5× gpt-4o-mini (vs 10× gpt-4o)

## Running the Tests

### Start Proxy
```bash
poetry run litellm --config test_its_proxy_config.yaml
```

### Run Tests
```bash
./test_its_proxy.sh
```

## Benefits for Agent Frameworks

Headers allow ITS configuration without modifying request body:
- ✅ No "unexpected argument" errors from agent frameworks
- ✅ Works with strict API schemas
- ✅ Transparent to downstream code
- ✅ Easy to toggle on/off per-request

## Response Metadata

ITS metadata is available as attributes on the response:
- `response.its_algorithm` - Algorithm used
- `response.its_budget` - Number of generations
- `response.its_vote_counts` - Vote counts (self-consistency)
- `response.its_scores` - Scores from judge (best-of-n)
- `response.its_selected_index` - Index of selected response
- `response.its_total_responses` - Total responses generated

## Troubleshooting

### Headers Not Working?
- Ensure header names are exact: `X-ITS-Algorithm` (case-sensitive)
- Check proxy logs for validation warnings
- Verify budget is an integer, temperatures are floats

### YAML Config Not Loading?
- Check YAML syntax with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
- Ensure `its_params` is at same level as `litellm_params`
- Restart proxy after config changes

### Algorithm Not Running?
- Check that `algorithm` header/param is set
- Verify `budget` is specified and >= 1
- For best-of-n, ensure judge_model is valid
