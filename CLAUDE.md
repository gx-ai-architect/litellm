# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**IMPORTANT**: Always activate the virtual environment before running commands:
```bash
source .venv/bin/activate
```

When running bash commands that use poetry or python, prefix with virtual environment activation:
```bash
source .venv/bin/activate && poetry install
source .venv/bin/activate && poetry run pytest tests/...
```

### Installation
- `make install-dev` - Install core development dependencies
- `make install-proxy-dev` - Install proxy development dependencies with full feature set
- `make install-test-deps` - Install all test dependencies
- `make install-dev-ci` - Install dev dependencies (CI-compatible, pins OpenAI version)

### Testing
- `make test` - Run all tests
- `make test-unit` - Run unit tests (tests/test_litellm) with 4 parallel workers via pytest-xdist
- `make test-integration` - Run integration tests (excludes unit tests)
- `poetry run pytest tests/path/to/test_file.py -v` - Run specific test file
- `poetry run pytest tests/path/to/test_file.py::test_function -v` - Run specific test

### Code Quality
- `make lint` - Run all linting (Ruff, MyPy, Black check, circular imports, import safety)
- `make format` - Apply Black code formatting (auto-fixes issues)
- `make format-check` - Check Black formatting without applying changes (matches CI)
- `make lint-ruff` - Run Ruff linting only
- `make lint-mypy` - Run MyPy type checking only
- `make check-circular-imports` - Check for circular import issues
- `make check-import-safety` - Verify safe imports

### Running Proxy Server Locally
- `poetry run litellm --config your_config.yaml` - Start proxy server with config file
- Config examples available in `litellm/proxy/example_config_yaml/`

## Architecture Overview

LiteLLM is a unified interface for 100+ LLM providers with two main components:

### Core Library (`litellm/`)
- **Main entry point**: `litellm/main.py` - Contains core completion() function
- **Provider implementations**: `litellm/llms/` - Each provider has its own subdirectory
- **Router system**: `litellm/router.py` + `litellm/router_utils/` - Load balancing and fallback logic
- **Type definitions**: `litellm/types/` - Pydantic models and type hints
- **Integrations**: `litellm/integrations/` - Third-party observability, caching, logging
- **Caching**: `litellm/caching/` - Multiple cache backends (Redis, in-memory, S3, etc.)

### Proxy Server (`litellm/proxy/`)
- **Main server**: `proxy_server.py` - FastAPI application
- **Authentication**: `auth/` - API key management, JWT, OAuth2
- **Database**: `db/` - Prisma ORM with PostgreSQL/SQLite support
- **Management endpoints**: `management_endpoints/` - Admin APIs for keys, teams, models
- **Pass-through endpoints**: `pass_through_endpoints/` - Provider-specific API forwarding
- **Guardrails**: `guardrails/` - Safety and content filtering hooks
- **UI Dashboard**: Served from `_experimental/out/` (Next.js build)

## Key Patterns

### Provider Implementation
- Providers inherit from base classes in `litellm/llms/base_llm/`
- Each provider has transformation functions for input/output formatting
- Support both sync and async operations
- Handle streaming responses and function calling
- All responses normalized to OpenAI-compatible format

### Error Handling
- Provider-specific exceptions mapped to OpenAI-compatible errors
- Fallback logic handled by Router system
- Comprehensive logging through `litellm/_logging.py`

### Configuration
- YAML config files for proxy server (see `litellm/proxy/example_config_yaml/`)
- Environment variables for API keys and settings
- Database schema managed via Prisma (`litellm/proxy/schema.prisma`)
- Model pricing and context windows defined in `model_prices_and_context_window.json`

## Development Notes

### Code Style
- Follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Uses Black formatter (line length: 120), Ruff linter, MyPy type checker
- Pydantic v2 for data validation (required v2.0.0+)
- Async/await patterns throughout
- Type hints required for all public APIs
- Package manager: Poetry

### Testing Strategy
- **Unit tests** in `tests/test_litellm/` - Mirrors `litellm/` structure, mocked only (no real API calls)
- **Integration tests** in `tests/llm_translation/` - Real provider API calls
- **Proxy tests** in `tests/proxy_unit_tests/` - FastAPI endpoint testing
- **Router tests** in `tests/router_unit_tests/` - Load balancing and fallback logic
- **Load tests** in `tests/load_tests/` - Performance benchmarking
- Parallel execution with pytest-xdist (4 workers for unit tests)
- **Hard requirement**: All PRs must include at least 1 test

### Contributing Requirements
- **Sign CLA**: All contributors must sign the [Contributor License Agreement](https://cla-assistant.io/BerriAI/litellm) before PRs can be merged
- **Add tests**: Minimum 1 test required per PR in `tests/test_litellm/`
- **Pass all checks**: `make lint` and `make test-unit` must pass
- **Keep scope isolated**: One feature/fix per PR
- Test file naming: `litellm/proxy/caching_routes.py` → `tests/test_litellm/proxy/test_caching_routes.py`

### Database Migrations
- Prisma handles schema migrations
- Migration files auto-generated with `prisma migrate dev`
- Always test migrations against both PostgreSQL and SQLite
- Schema files located at `litellm/proxy/schema.prisma`

### Enterprise Features
- Enterprise-specific code in `enterprise/` directory
- Optional features enabled via environment variables
- Separate licensing and authentication for enterprise features

### Docker Development
- Multiple Dockerfiles in `docker/` directory
- Use `-stable` tag for production (undergoes 12-hour load tests)
- Local development: `docker build -f docker/Dockerfile.non_root -t litellm_dev .`
- Proxy server runs on port 4000 by default

## ITS-Hub Integration Plan

### Overview
Integrate `its-hub` library (read: ITS_HUB.md; or /Users/gxxu/Desktop/its_integration/its_hub) to add inference-time scaling algorithms to LiteLLM without reimplementing them. This enables self-consistency and best-of-n sampling features by wrapping existing LiteLLM functionality.



### Existing Implementation (Context)

**What Was Already Tried** (commit `ec19dbab7`):
- **Location**: Router-level implementation in `litellm/router_utils/majority_voting.py`
- **Approach**: Custom majority voting algorithm integrated directly into Router's `_acompletion` method
- **Configuration**: YAML-based via `litellm_params`
- **Key Issue**: Calls `async_get_available_deployment()` N times → could get different deployments for each call!

**Learnings from Router-Level Implementation**:
1. ✅ **YAML config is intuitive** - `algorithm` + `budget` params are user-friendly
2. ✅ **Metadata is useful** - tracking vote counts, success/fail rates in response
3. ✅ **Error resilience needed** - must handle partial failures gracefully
4. ⚠️ **Router is wrong level** - router selects deployments; ITS needs same deployment N times
5. ⚠️ **Deployment inconsistency** - each call could hit different API keys, endpoints, regions
6. ⚠️ **Limited availability** - only router users benefit, not `litellm.acompletion()` direct users
7. ⚠️ **Semantic mismatch** - router = "which deployment?", ITS = "how to call?"

**Why Core Library Level is Better**:
- **Guaranteed consistency**: Decide params ONCE, call N times with same config
- **Universal availability**: Works for direct users, router users, and proxy users
- **Semantic fit**: ITS is about "how to call a model" not "which deployment to choose"
- **Simpler**: Just wrap `litellm.acompletion()` - one integration point
- **Future-proof**: All ITS algorithms work the same way

### Architecture Decision: Core Library Integration

**Integration Point**: `litellm/main.py` at the `acompletion()` level

**Key Principles**:
1. **Wrapper LM Class**: Create `AbstractLanguageModel` wrapper around `litellm.acompletion()`
2. **Early Dispatch**: If `algorithm` param present, route to ITS before normal completion flow
3. **Import, Don't Reimplement**: Use its-hub's battle-tested algorithms
4. **Minimal Dependencies**: Core its-hub only requires `numpy` and `typing-extensions`
5. **Backward Compatible**: Existing majority-voting in router can coexist or be deprecated

### Minimal Integration Approach

**Dependencies Added**:
```toml
# In pyproject.toml
its-hub = "^0.1.0"  # Core only: numpy + typing-extensions
```

#### 1. Wrapper LM Class (`litellm/its_hub_integration/adapter.py`)
**Purpose**: Wrap `litellm.acompletion()` to implement its-hub's `AbstractLanguageModel` interface

```python
from its_hub import AbstractLanguageModel
import litellm

class LiteLLMLanguageModel(AbstractLanguageModel):
    """Wrapper around litellm.acompletion for its-hub algorithms."""

    def __init__(self, model: str, **litellm_kwargs):
        """
        Args:
            model: LiteLLM model name (e.g., "gpt-4", "anthropic/claude-3")
            **litellm_kwargs: All other params for litellm.acompletion
                             (temperature, api_key, etc.)
        """
        self.model = model
        self.litellm_kwargs = litellm_kwargs

    async def agenerate(self, messages, stop=None, **kwargs):
        """
        Call litellm.acompletion with consistent params.

        Args:
            messages: list[dict] in OpenAI format OR list[list[dict]] for batch
            stop: Optional stop sequences
            **kwargs: Additional generation params (override litellm_kwargs)

        Returns:
            dict with {"role": "assistant", "content": "..."}
            OR list[dict] if messages was batched
        """
        # Merge kwargs
        call_kwargs = {**self.litellm_kwargs, **kwargs}
        if stop is not None:
            call_kwargs["stop"] = stop

        # Handle batch vs single
        is_batch = isinstance(messages[0], list)

        if is_batch:
            # Batch call - make N concurrent calls
            import asyncio
            tasks = [
                litellm.acompletion(model=self.model, messages=msg, **call_kwargs)
                for msg in messages
            ]
            responses = await asyncio.gather(*tasks)
            return [
                {"role": "assistant", "content": r.choices[0].message.content}
                for r in responses
            ]
        else:
            # Single call
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                **call_kwargs
            )
            return {"role": "assistant", "content": response.choices[0].message.content}
```

**Key Points**:
- ~40 lines total
- Wraps `litellm.acompletion()` - works with ALL providers
- Consistent params across N calls (no deployment re-selection)
- Handles both single and batch generation
- No dependencies on router internals

#### 2. Core Library Integration (`litellm/main.py`)
**Add early dispatch** in `acompletion()` function:

```python
# In litellm/main.py, at start of acompletion()
async def acompletion(
    model: str,
    messages: List = [],
    # ... existing params ...
    algorithm: Optional[str] = None,  # NEW: "self-consistency" | "best-of-n"
    budget: Optional[int] = None,     # NEW: number of generations
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Async completion with optional inference-time scaling algorithms.

    Args:
        algorithm: Optional ITS algorithm ("self-consistency", "best-of-n")
        budget: Number of generations for ITS algorithm
    """

    # Early dispatch to ITS if algorithm is specified
    if algorithm is not None:
        from litellm.its_hub_integration import apply_its_algorithm
        return await apply_its_algorithm(
            model=model,
            messages=messages,
            algorithm=algorithm,
            budget=budget,
            **kwargs
        )

    # Normal completion flow continues...
    # ... existing acompletion code ...
```

#### 3. ITS Algorithm Dispatcher (`litellm/its_hub_integration/__init__.py`)

```python
from its_hub import SelfConsistency, BestOfN
from its_hub.reward_models import LLMJudge
from .adapter import LiteLLMLanguageModel
from litellm import ModelResponse

async def apply_its_algorithm(
    model: str,
    messages: list,
    algorithm: str,
    budget: int = 5,
    **litellm_kwargs
) -> ModelResponse:
    """
    Apply its-hub algorithm using LiteLLM as the backend.

    Args:
        model: LiteLLM model name
        messages: Chat messages
        algorithm: "self-consistency" or "best-of-n"
        budget: Number of generations
        **litellm_kwargs: Passed to litellm.acompletion

    Returns:
        ModelResponse with best/majority result
    """
    # Validate
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if litellm_kwargs.get("stream"):
        raise ValueError(f"{algorithm} does not support streaming")

    # Create wrapper
    lm = LiteLLMLanguageModel(model=model, **litellm_kwargs)

    # Select algorithm
    if algorithm == "self-consistency":
        alg = SelfConsistency()
    elif algorithm == "best-of-n":
        judge = LLMJudge(lm=lm)  # Uses our wrapper for judging!
        alg = BestOfN(judge)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Run algorithm
    result = await alg.ainfer(lm, messages, budget)

    # Convert its-hub result to ModelResponse
    # (its-hub returns result.the_one which is a dict)
    # We need to wrap it back into ModelResponse format
    response = _convert_to_model_response(result, model, algorithm, budget)
    return response
```

**Key Points**:
- ~50 lines total
- Just imports from its-hub and dispatches
- Reward model is FREE (pass wrapper to `LLMJudge`)
- Converts its-hub result back to LiteLLM `ModelResponse`

#### 4. Implementation Steps (MVP)

**Phase 1: Wrapper LM Class** (~40 lines, 1 hour)
1. Create `litellm/its_hub_integration/adapter.py`
2. Implement `LiteLLMLanguageModel(AbstractLanguageModel)`
3. Test: Unit test that it wraps `litellm.acompletion()` correctly

**Phase 2: Algorithm Dispatcher** (~50 lines, 1 hour)
1. Create `litellm/its_hub_integration/__init__.py`
2. Implement `apply_its_algorithm()` function
3. Test: Unit test with mocked `litellm.acompletion`

**Phase 3: Core Library Integration** (~20 lines, 30 min)
1. Add `algorithm` and `budget` params to `acompletion()` signature
2. Add early dispatch at start of `acompletion()`
3. Test: Integration test with real `litellm.acompletion` calls

### Testing Requirements (MVP)

**Quick E2E Test** - Run after implementation to verify it works:
```bash
# Requires OpenAI API key in .env file at repository root
poetry run pytest tests/test_litellm/its_hub_integration/test_simple_integration.py -v -s
```

This runs 2 simple tests with real API calls:
- `test_self_consistency()` - Verifies self-consistency algorithm works
- `test_best_of_n()` - Verifies best-of-n algorithm works

### Usage Examples (MVP)

**Direct Usage** (new capability!):
```python
import litellm

# Self-consistency: Generate 5 times, return most common answer
response = await litellm.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 15 * 24?"}],
    algorithm="self-consistency",
    budget=5,
    temperature=0.7
)

print(response.choices[0].message.content)  # "360"
print(response._hidden_params["its_algorithm"])  # "self-consistency"
print(response._hidden_params["vote_counts"])    # {"360": 4, "350": 1}

# Best-of-N: Generate 5 times, use LLM judge to pick best
response = await litellm.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a sorting function"}],
    algorithm="best-of-n",
    budget=5
)
```

**Via Router** (works automatically):
```python
from litellm import Router

# Router just passes algorithm/budget to core litellm.acompletion
router = Router(
    model_list=[{
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"}
    }]
)

response = await router.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 15 * 24?"}],
    algorithm="self-consistency",
    budget=5
)
```

**Via YAML Config** (router extracts and passes to core):
```yaml
# Note: Router needs small update to extract algorithm/budget from litellm_params
# and pass to acompletion call
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
```

Then use:
```python
# Algorithm specified at call time (preferred)
response = await router.acompletion(
    model="gpt-4",
    messages=[...],
    algorithm="self-consistency",
    budget=5
)
```

Response: Standard OpenAI-compatible `ModelResponse` (single best answer in `choices[0]`)

### File Structure (MVP)
```
litellm/
├── main.py                     # MODIFIED: add algorithm/budget params, early dispatch
├── its_hub_integration/        # NEW
│   ├── __init__.py             # ~50 lines: apply_its_algorithm()
│   └── adapter.py              # ~40 lines: LiteLLMLanguageModel class
├── router.py                   # OPTIONAL: extract algorithm/budget from litellm_params
tests/
└── test_litellm/
    └── its_hub_integration/    # NEW
        ├── test_adapter.py
        └── test_dispatcher.py
```

**Total New Code**: ~110 lines (adapter 40 + dispatcher 50 + main.py integration 20)

### Router Integration (Optional Enhancement)

**If you want YAML config support**, add this to `router.py`:

```python
# In Router._acompletion, before calling litellm.acompletion
deployment_params = deployment["litellm_params"]

# Extract ITS params from deployment config
algorithm = deployment_params.get("algorithm")
budget = deployment_params.get("budget")

# Pass to core (which has early dispatch)
response = await litellm.acompletion(
    model=model_name,
    messages=messages,
    algorithm=algorithm,  # Will be None if not specified
    budget=budget,        # Will be None if not specified
    **kwargs
)
```

This way YAML config works, but the logic lives in core.

### Success Criteria (MVP)
1. ✅ Zero custom algorithm implementation (100% reuse from its-hub)
2. ✅ Minimal dependencies (numpy, typing-extensions only)
3. ✅ Minimal code (~110 lines total)
4. ✅ Universal availability (works for direct users, router auto-inherits)
5. ✅ Guaranteed consistency (same model/params for all N calls)
6. ✅ Clean separation (wrapper class, no tight coupling to router)