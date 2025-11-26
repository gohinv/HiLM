# U-HLM Intermediate Layer Service (ILM)

This service implements the Intermediate Language Model (ILM) layer (Tier 2) of the 3-Tier Uncertainty-Aware Hybrid Language Model architecture.

## Overview
The ILM acts as a bridge between the lightweight SLM (Tier 1) and the powerful LLM (Tier 3). It uses a medium-sized model (e.g., Llama 3.2 3B) to verify tokens that the SLM is uncertain about. If the ILM itself is uncertain, it offloads the verification to the upstream LLM.

## Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or sufficient CPU/RAM
- Upstream LLM Service running (default: localhost:8081)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the `common` package is in your python path or installed (this is handled automatically by the script if run from repo root).

## Usage

Start the service:

```bash
python ilm_service/server/main.py
```

### Configuration
The service is configured via constants in `server/main.py` (or command line arguments if enabled):

- **Port**: 8082 (gRPC service port)
- **Model**: meta-llama/Llama3.2-3B-Instruct
- **Upstream LLM**: localhost:8081
- **Threshold (T_ILM)**: 0.4 (Uncertainty threshold for offloading to Tier 3)

## Architecture
- **Handler**: `ILMHandler` implements the `UHLMServicer` gRPC interface.
- **Session Management**: Uses `SessionManager` to track conversation state locally.
- **Inference**: Uses `vLLMClient` for efficient local inference.
- **Offloading**: Uses `UHLMRPCClient` to communicate with the upstream LLM.

