# Evaluation Setup Summary

## What's Been Configured

✅ **Threshold Updated**: T2 set to `0.6316` in `slm_service/threshold_calc.py`

✅ **Test Prompts**: 100 prompts saved to `eval/test_prompts.txt`

✅ **Evaluation Scripts Created**:
- `eval/run_hilm_eval.py` - HiLM evaluation with full metrics
- `eval/run_baseline_eval.py` - Baseline LLM evaluation  
- `eval/analyze_results.py` - Results analysis and comparison
- `eval/gpu_monitor.py` - GPU utilization logger

## Next Steps - Running the Evaluation

### 1. Install Dependencies
```bash
pip install pandas bert-score
```

### 2. Start Services (in separate terminals)

**Terminal 1 - LLM Service:**
```bash
cd llm_service
python -m server.main
```
Wait for: `gRPC server started on [::]:8081`

**Terminal 2 - ILM Service:**
```bash
cd ilm_service  
python -m server.main
```
Wait for: `✅ ILM Service started on [::]:8082`

**Terminal 3 - Run Evaluation:**
```bash
cd eval
```

### 3. Run HiLM Evaluation

```bash
python run_hilm_eval.py --prompts test_prompts.txt --output results --runs 3 --max-tokens 50
```

This will:
- Process all 100 prompts, 3 times each (300 total runs)
- Automatically start/stop GPU monitoring
- Save results to `results/` directory
- Take approximately 30-60 minutes depending on your hardware

**What gets logged:**
- Token timestamps (for TTNT calculation)
- Decision counts (SKIP, VERIFY_ILM, VERIFY_LLM)
- Acceptance/rejection stats
- RPC call counts
- GPU utilization (automatically)

### 4. Run Baseline Evaluation

In the same terminal (after HiLM completes), or in a new terminal:

```bash
python run_baseline_eval.py --prompts test_prompts.txt --output results --runs 3 --max-tokens 50
```

**Note**: The baseline script uses the LLM service's verify endpoint. For a more accurate baseline, you might want to modify it to use vLLM's generation directly, but this should work for comparison.

### 5. Analyze Results

After both evaluations complete, find the result files in `results/`:
- `all_results_YYYYMMDD_HHMMSS.json` (HiLM)
- `baseline_all_results_YYYYMMDD_HHMMSS.json` (Baseline)
- `gpu_log_YYYYMMDD_HHMMSS.csv` (HiLM GPU log)
- `gpu_log_baseline_YYYYMMDD_HHMMSS.csv` (Baseline GPU log)

Then run:
```bash
python analyze_results.py \
  --hilm-results results/all_results_YYYYMMDD_HHMMSS.json \
  --hilm-gpu-log results/gpu_log_YYYYMMDD_HHMMSS.csv \
  --baseline-results results/baseline_all_results_YYYYMMDD_HHMMSS.json \
  --baseline-gpu-log results/gpu_log_baseline_YYYYMMDD_HHMMSS.csv \
  --output comparison.json
```

## Metrics Collected

### Performance
- **TPS**: Tokens per second (measured)
- **TPS_norm**: Normalized TPS (TPS / GPU utilization)
- **TTNT**: Time to next token (mean, std, min, max)
- **E2E Latency**: End-to-end time per query

### Accuracy (requires additional step)
- **BERTScore**: Need to compute separately using output texts
- **Acceptance Rate**: Already collected

### Overhead
- **RPC Calls**: Counts by type
- **Decision Breakdown**: SKIP vs VERIFY_ILM vs VERIFY_LLM

## Notes

1. **GPU Monitoring**: Runs automatically at 10 Hz (every 100ms). You can also run it manually:
   ```bash
   python gpu_monitor.py --output gpu_log.csv --rate 10.0
   ```

2. **Baseline Script**: Currently uses verify endpoint with dummy drafts. For more accurate baseline, consider using vLLM generation directly.

3. **BERTScore**: Not automatically computed. You'll need to:
   - Extract response texts from results JSON
   - Run BERTScore comparison between HiLM and baseline outputs
   - Add to analysis script if needed

4. **Results Format**: All results saved as JSON for easy analysis. GPU logs are CSV.

5. **Multiple Runs**: The `--runs` parameter allows averaging across multiple runs to reduce variance (especially important for BERTScore).

## Troubleshooting

- **Services not starting**: Check ports 8081 and 8082 are available
- **GPU monitoring fails**: Ensure `nvidia-smi` is available and GPU is accessible
- **Import errors**: Make sure you're running from the repo root or have PYTHONPATH set correctly
- **Out of memory**: Reduce `--max-tokens` or run fewer prompts at a time

