# HiLM: Hierarchical Uncertainty-Aware Speculative Language Model Inference

A three-tier hierarchical language model system with SLM (edge device), ILM (edge server), and LLM (data center) services communicating via gRPC.

ALl of this is set up to run on a Single A100 (shown by the hardcoded IP address and port configurations)

## Quick Start

### 1. LLM Service
```
cd llm_service
python -m server.main
```

**Configuration:** Edit `llm_service/server/main.py`  to change model ID or port.

### 2. ILM Service
```
cd ilm_service
# without latency simulation
python -m server.main

# with latency simulation (10ms edge + 50ms datacenter)
python -m server.main --latency
```
**Features:**
- Intermediate model inference (Llama 3.2 3B)
- Uncertainty-based offloading to LLM
- Local verification for confident tokens
- Session mapping between SLM and LLM sessions

**Configuration:** Edit `ilm_service/server/main.py`:
- `MODEL_ID`: ILM model (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `LLM_HOST`: Upstream LLM host (default: `127.0.0.1`)
- `LLM_PORT`: Upstream LLM port (default: `8081`)
- `THRESHOLD`: Uncertainty threshold for offloading (default: `0.607`)
- `PORT`: Service port (default: `8082`)

**Flags:**
- `--latency` / `--simulate-latency`: Enable network latency simulation

### 3. SLM Service (Client)
```
# Basic usage
python -m slm_service.main

# With latency simulation
python -m slm_service.main --latency

# With chat template formatting
python -m slm_service.main --use-chat-template

# Combined
python -m slm_service.main --latency --use-chat-template
```

**Features:**
- Small language model inference (Llama 3.2 1B)
- Uncertainty-based routing (SKIP, VERIFY_ILM, VERIFY_LLM)
- Interactive prompt interface
- Automatic session management

**Configuration:** Edit `slm_service/main.py`:
- Model ID (default: `meta-llama/Llama-3.2-1B-Instruct`)
- Threshold values in `threshold_calc.py`
- Speculation parameters (K, theta_max)

**Flags:**
- `--latency` / `--simulate-latency`: Enable network latency simulation (10ms ILM, 60ms LLM)
- `--use-chat-template`: Use chat template formatting for prompts

## Installation

## Install dependencies
```
# optionally set up a virtual environment at root
pip install -r llm_service/requirements.txt
pip install -r ilm_service/requirements.txt
pip install -r slm_service/requirements.txt
```

## Service Order
Start services in this order:
1. LLM Service (port 8081)
2. ILM Service (port 8082) - connects to LLM
3. SLM Service (client) - connects to both ILM and LLM