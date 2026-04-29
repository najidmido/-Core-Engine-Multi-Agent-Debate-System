> **Module:** Core Engine &nbsp;|&nbsp; **Author:** Najid Mido &nbsp;|&nbsp; **Version:** 1.0.0  
> **Stack:** Python 3.10+ · LangGraph · HuggingFace Transformers · Qwen 2.5

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Structure](#module-structure)
4. [Getting Started](#getting-started)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Integration Guide](#integration-guide)
8. [Dependencies](#dependencies)
9. [Contact](#contact)

---

## Overview

The **Core Engine** is the foundational layer of the Multi-Agent Debate System. It is responsible for three critical concerns:

- **State management** — defines the shared data schema (`DebateState`) consumed by all agents across the system
- **LLM inference** — loads and serves a local Qwen 2.5 language model via HuggingFace Transformers, exposing a clean `generate_response()` interface
- **Workflow orchestration** — constructs and compiles the LangGraph `StateGraph` that governs the debate flow: Pro Agent → Con Agent → (loop or Judge)

All other modules in this project depend on the Core Engine. It must be initialised before any agent logic is executed.

---

## Architecture

The debate workflow follows a **Directed Acyclic Graph (DAG)** compiled by LangGraph:

```
┌───────────┐       ┌───────────┐
│  Pro Agent│──────▶│  Con Agent│
└───────────┘       └─────┬─────┘
       ▲                  │
       │    round_count   │   round_count
       │    < max_rounds  │   >= max_rounds
       └──────────────────┤
                          │
                    ┌─────▼──────┐
                    │ Judge Agent│
                    └─────┬──────┘
                          │
                        [END]
```

**Flow:**
1. The **Pro Agent** generates a supporting argument.
2. The **Con Agent** generates a counter-argument and increments the round counter.
3. A conditional check evaluates whether `round_count < max_rounds`:
   - If yes → loop back to the Pro Agent for the next round.
   - If no → hand off to the Judge Agent.
4. The **Judge Agent** reviews the full debate history, declares a winner, and terminates the graph.

---

## Module Structure

```
core_engine/
├── memory.py        # Shared state schema (DebateState)
├── qwen_utils.py    # LLM loader and inference interface
├── graph.py         # LangGraph workflow builder
└── README.md        # This document
```

### `memory.py` — Shared State Schema

Defines `DebateState`, a `TypedDict` that serves as the single source of truth passed through every node of the graph.

```python
class DebateState(TypedDict):
    topic: str                                    # The debate topic
    max_rounds: int                               # Total rounds to run
    round_count: int                              # Current round (incremented by Con Agent)
    history: Annotated[List[str], operator.add]  # Debate transcript (append-only)
    winner: str                                   # Set by Judge Agent: "Pro" | "Con"
    judge_reason: str                             # Judge Agent's reasoning
```

> ⚠️ **Important:** The `history` field uses `operator.add` as its reducer, meaning each node's returned list is **appended** to the existing history — not replaced.

---

### `qwen_utils.py` — LLM Inference Interface

Loads the `Qwen/Qwen2.5-0.5B-Instruct` model locally via HuggingFace and exposes a single function for text generation.

**Key behaviour:**
- Automatically uses `float16` precision if a CUDA GPU is detected; falls back to `float32` on CPU.
- Uses the Qwen chat template (`<|im_start|>` / `<|im_end|>`) for structured prompt formatting.
- Returns only the newly generated text (not the full prompt echo).

---

### `graph.py` — Workflow Builder

Assembles the LangGraph `StateGraph` and wires all agent nodes together via edges and conditional routing.

**Exports one function:**

```python
def build_debate_graph() -> CompiledGraph
```

The returned compiled graph accepts an `initial_state` dict and can be run via `.stream()` or `.invoke()`.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A virtual environment (strongly recommended)
- CUDA-compatible GPU (optional, but recommended for faster inference)

### Installation

```bash
# 1. Clone the repository or extract the provided archive
cd core_engine/

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> On first run, `qwen_utils.py` will automatically download the `Qwen/Qwen2.5-0.5B-Instruct` model weights (~1 GB) from HuggingFace Hub. Ensure you have an active internet connection.

### Smoke Test

To verify the Core Engine is functioning correctly in isolation:

```python
# test_core.py
from memory import DebateState
from qwen_utils import generate_response
from graph import build_debate_graph

# Test LLM
response = generate_response(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say hello in one sentence."
)
print("LLM response:", response)

# Test graph compilation
app = build_debate_graph()
print("Graph compiled successfully:", app)
```

---

## API Reference

### `generate_response(system_prompt, user_prompt) → str`

Generates a text response from the local Qwen LLM.

| Parameter | Type | Description |
|---|---|---|
| `system_prompt` | `str` | Instructions that define the model's role and behaviour |
| `user_prompt` | `str` | The actual input or question to respond to |
| **Returns** | `str` | The model's generated response, stripped of leading/trailing whitespace |

**Example:**

```python
from qwen_utils import generate_response

reply = generate_response(
    system_prompt="You are a professional debate coach.",
    user_prompt="Give me a strong opening argument for renewable energy."
)
print(reply)
```

---

### `build_debate_graph() → CompiledGraph`

Constructs and compiles the full LangGraph debate workflow.

| Parameter | — | No parameters |
|---|---|---|
| **Returns** | `CompiledGraph` | A compiled LangGraph application ready to run via `.stream()` or `.invoke()` |

**Example:**

```python
from graph import build_debate_graph

app = build_debate_graph()

initial_state = {
    "topic": "Remote work improves productivity",
    "max_rounds": 3,
    "round_count": 0,
    "history": [],
    "winner": "",
    "judge_reason": ""
}

for output in app.stream(initial_state):
    print(output)
```

---

## Configuration

The following parameters can be adjusted directly in `qwen_utils.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_ID` | `"Qwen/Qwen2.5-0.5B-Instruct"` | HuggingFace model identifier |
| `temperature` | `0.7` | Sampling temperature (higher = more creative) |
| `max_new_tokens` | `150` | Maximum tokens generated per response |
| `do_sample` | `True` | Enables sampling-based generation |

> To use a larger Qwen variant (e.g., `Qwen2.5-7B-Instruct`), update `MODEL_ID` and ensure sufficient VRAM is available.

---

## Integration Guide

The Core Engine is designed to be consumed by two other modules in this project:

| Module | What it uses from Core Engine |
|---|---|
| **Agents Module** (Person 2) | `DebateState` from `memory.py`; `generate_response()` from `qwen_utils.py` |
| **App & Logging Module** (Person 3) | `build_debate_graph()` from `graph.py` |

### Merging for Full-System Execution

When integrating all three modules, place all Python files into a **single flat directory**:

```
project/
├── memory.py         ← Core Engine (Person 1)
├── qwen_utils.py     ← Core Engine (Person 1)
├── graph.py          ← Core Engine (Person 1)
├── agents.py         ← Agents Module (Person 2)
├── judge.py          ← Agents Module (Person 2)
├── main.py           ← App & Logging Module (Person 3)
└── requirements.txt
```

Then run:
```bash
python main.py
```

### Contract: Do Not Break These Interfaces

The following are consumed by other modules and must remain stable:

- `DebateState` field names and types in `memory.py`
- `generate_response(system_prompt: str, user_prompt: str) -> str` signature in `qwen_utils.py`
- `build_debate_graph() -> CompiledGraph` signature in `graph.py`

> Any changes to the above must be communicated to all team members before merging.

---

## Dependencies

```
langgraph
langchain-core
transformers
torch
accelerate
```

Install all at once:

```bash
pip install langgraph langchain-core transformers torch accelerate
```

| Package | Purpose |
|---|---|
| `langgraph` | Stateful multi-agent graph orchestration |
| `langchain-core` | Base types and interfaces used by LangGraph |
| `transformers` | HuggingFace model loading and inference pipeline |
| `torch` | PyTorch backend for model execution |
| `accelerate` | Optimised model loading with automatic device mapping |

---

## Contact

**Module Owner:** Najid Mido 
**Role:** Core Engine — LLM Backbone & Graph Orchestration  

For questions about state schema changes, LLM configuration, or graph topology, contact Najid before making modifications that affect shared interfaces.
