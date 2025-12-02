# HiLM Evaluation Guide

This directory contains scripts for evaluating HiLM performance against a baseline LLM.

## Setup

1. **Install additional dependencies:**
   ```bash
   pip install pandas bert-score
   ```

2. **Ensure services are configured:**
   - SLM: `meta-llama/Llama-3.2-1B-Instruct` (already configured)
   - LLM: `meta-llama/Llama-3.1-8B-Instruct` (already configured)
   - Threshold T2: `0.6316` (already updated)

## Running Evaluations

### Step 1: Start Services

You'll need to start the services in separate terminals:

**Terminal 1 - LLM Service:**
```bash
cd llm_service
python -m server.main
```

**Terminal 2 - ILM Service:**
```bash
cd ilm_service
python -m server.main
```

**Terminal 3 - Evaluation:**
```bash
cd eval
```

### Step 2: Run HiLM Evaluation

```bash
python run_hilm_eval.py --prompts test_prompts.txt --output results --runs 3 --max-tokens 50
```

This will:
- Run each prompt 3 times
- Collect metrics (TPS, TTNT, acceptance rate, RPC calls, etc.)
- Log GPU utilization automatically
- Save results to `results/` directory

### Step 3: Run Baseline Evaluation

For baseline comparison, you'll need to run the baseline script (note: this may need adjustment based on your baseline setup):

```bash
python run_baseline_eval.py --prompts test_prompts.txt --output results --runs 3 --max-tokens 50
```

### Step 4: Analyze Results

After both evaluations complete, compare the results:

```bash
python analyze_results.py \
  --hilm-results results/all_results_YYYYMMDD_HHMMSS.json \
  --hilm-gpu-log results/gpu_log_YYYYMMDD_HHMMSS.csv \
  --baseline-results results/baseline_all_results_YYYYMMDD_HHMMSS.json \
  --baseline-gpu-log results/gpu_log_baseline_YYYYMMDD_HHMMSS.csv \
  --output comparison.json
```

## Metrics Collected

### Performance Metrics
- **TPS (Tokens Per Second)**: Measured throughput
- **TPS_norm**: Normalized TPS (TPS / average GPU utilization)
- **TTNT (Time to Next Token)**: Latency between consecutive tokens
- **End-to-End Latency**: Total wall-clock time per query
- **GPU Utilization**: Average GPU compute utilization percentage

### Accuracy Metrics
- **Acceptance Rate**: Percentage of draft tokens accepted
- **BERTScore**: Semantic similarity (requires separate calculation)

### Overhead Metrics
- **RPC Call Counts**: Number of BeginSession, VerifyToken, Sync, EndSession calls
- **Decision Counts**: Breakdown of SKIP, VERIFY_ILM, VERIFY_LLM decisions

## Output Files

- `results/all_results_*.json`: Complete HiLM evaluation results
- `results/baseline_all_results_*.json`: Complete baseline evaluation results
- `results/gpu_log_*.csv`: GPU utilization logs
- `comparison.json`: Comparison report between HiLM and baseline

## Manual GPU Monitoring

If you want to monitor GPU manually (instead of automatic logging):

```bash
python gpu_monitor.py --output gpu_log.csv --rate 10.0
```

Press Ctrl+C to stop.

## Notes

- GPU monitoring runs automatically during evaluation
- Results are saved incrementally (per run) and aggregated at the end
- For BERTScore accuracy measurement, you'll need to run additional analysis on the output texts
- The baseline script may need adjustment depending on your baseline setup

