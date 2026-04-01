# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must be run inside the `dlrouter` conda environment:

```bash
conda run -n dlrouter <cmd>
```

Note: `conda run` does **not** support `--no-verify` or `--no-banner` flags in this setup.

## Common Commands

```bash
# Run all tests
conda run -n dlrouter pytest

# Run a single test file
conda run -n dlrouter pytest tests/backends/test_vllm_backend.py

# Run a single test
conda run -n dlrouter pytest tests/backends/test_vllm_backend.py::TestPDDisagg::test_prefill_request_uses_max_tokens_1

# Lint
conda run -n dlrouter ruff check .

# Format
conda run -n dlrouter ruff format .

# Lint + auto-fix
conda run -n dlrouter ruff check --fix .

# Start the router (development)
conda run -n dlrouter python -m dlrouter --backend vllm --serving_strategy distserve --log_level DEBUG
```

## Code Style

- Ruff for lint and format; single quotes; 120 char line length
- `print` statements are forbidden in library code (T20); use `get_logger()`
- `pytest-asyncio` with `asyncio_mode = "auto"` — async test methods work without decorators
- No need for `@pytest.mark.asyncio`

## Architecture

The router is a FastAPI proxy that sits in front of one or more LLM inference engine instances. Requests arrive at DLRouter, are routed to a backend node, and the response is streamed or returned.

### Request flow

```
HTTP request → FastAPI routes (api/routes/) → ProxyEngine.dispatch()
    → NodeManager.get_node_url()          # routing
    → backend.forward_request() / stream_forward() / prefill_request() + decode_request()
```

### Key components

**`dlrouter/api/app.py`** — application factory (`create_app`). Wires together backend, `NodeManager`, `ProxyEngine`, and `HealthChecker`. Route handlers receive injected `ProxyEngine` and `NodeManager` references via module-level setters (e.g. `chat.set_dependencies()`).

**`dlrouter/core/node_manager.py`** — central registry. Stores `{url: NodeStatus}` (thread-safe with `RLock`), handles `add/remove`, delegates routing to a `BaseRoutingStrategy`, and tracks in-flight request counts and latency via `pre_call`/`post_call`. Persists state to JSON (`router_config.json` by default).

**`dlrouter/core/proxy_engine.py`** — request orchestration. Two modes:
- `handle_hybrid`: single-node forward (streaming or non-streaming)
- `handle_distserve`: two-phase PD flow — phase 1 to Prefill node (`max_tokens=1`), phase 2 to Decode node with first token appended as assistant turn

**`dlrouter/backends/`** — backend adapters. `BaseBackend` (ABC) defines the interface. Concrete implementations: `VLLMBackend`, `LMDeployBackend`. `factory.py` instantiates the correct one from `BackendConfig`.

**`dlrouter/routing/`** — routing strategies. `BaseRoutingStrategy` ABC; five implementations selected by `RoutingStrategy` enum. All filter by model name first, then pick a node. `consistent_hash` and `load_aware` (`min_expected_latency`, `min_observed_latency`) are the most complex.

**`dlrouter/core/health_check.py`** — background daemon thread. Checks each node every `HEARTBEAT_EXPIRATION` seconds; removes a node only after `HEALTH_CHECK_MAX_FAILURES` consecutive failures.

### Serving strategies vs backend types

| | LMDeploy | vLLM |
|---|---|---|
| Hybrid | ✓ | ✓ |
| DistServe (PD disagg) | DLRouter manages RDMA pool (`PDConnectionPool`), `is_connected_pd` tracks connections | vLLM manages its own KV transfer via `--kv-transfer-config`; `is_connected_pd` always returns `True`, `connect_pd` is no-op |

### Node roles

Nodes are registered with a `role` field (`HYBRID`, `PREFILL`, or `DECODE`) stored in `NodeStatus`. `NodeManager.get_node_url()` filters candidates by role before routing.

### Environment variables for tuning

| Variable | Default | Purpose |
|---|---|---|
| `DLROUTER_HEARTBEAT_EXPIRATION` | 90s | Health check interval |
| `DLROUTER_HEALTH_CHECK_TIMEOUT` | 30s | Per-check HTTP timeout (increase for PD scenarios) |
| `DLROUTER_HEALTH_CHECK_MAX_FAILURES` | 3 | Consecutive failures before removal |
| `DLROUTER_AIOHTTP_TIMEOUT` | 1800s | Inference request timeout |
