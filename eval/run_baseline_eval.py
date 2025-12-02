"""
Baseline LLM Evaluation Script
Runs single LLM model inference for comparison
"""
import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import numpy as np

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.rpc_client import UHLMRPCClient
from common.uhlm import uhlm_pb2


def load_prompts(prompts_file):
    """Load prompts from file, skipping comments and empty lines"""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


async def generate_baseline_with_metrics(prompt, max_tokens=50):
    """
    Generate response using only LLM (baseline) with metrics collection
    Uses LLM's own distribution to generate tokens (always accepts)
    """
    metrics = {
        'prompt': prompt,
        'start_time': time.time(),
        'token_timestamps': [],
        'total_tokens': 0,
    }
    
    # We need to get logits from LLM service
    # For baseline, we'll use a direct approach: get probs from LLM and sample
    # But since we don't have direct logits access, we'll use verify with LLM's most likely token
    
    async with UHLMRPCClient(host="127.0.0.1", port=8081, simulate_latency=False, latency_ms=0) as llm:
        # Begin session
        session_id, eos_token_id = await llm.begin_session(prompt)
        
        response_token_ids = []
        current_text = prompt
        
        try:
            for step in range(max_tokens):
                token_start = time.time()
                
                # For baseline: get LLM's distribution by using verify with a dummy draft
                # The LLM will return its own token (since dummy draft will be rejected)
                # Use token 0 as dummy draft with uniform probs
                vocab_size = 32000  # Approximate, will be adjusted by actual response
                dummy_probs = np.ones(vocab_size, dtype=np.float32) / vocab_size
                
                accepted, token_id, _ = await llm.verify(
                    session_id, 0, dummy_probs
                )
                
                if int(token_id) == int(eos_token_id):
                    break
                
                response_token_ids.append(token_id)
                token_end = time.time()
                metrics['token_timestamps'].append(token_end)
                
                # Sync context
                await llm.sync(session_id, [token_id])
                
        finally:
            await llm.end_session(session_id)
    
    # Decode response
    # We need the tokenizer to decode, but we don't have it here
    # We'll store token IDs and decode later if needed
    metrics['end_time'] = time.time()
    metrics['token_ids'] = [int(t) for t in response_token_ids]
    metrics['total_tokens'] = len(response_token_ids)
    metrics['total_time'] = metrics['end_time'] - metrics['start_time']
    
    # Calculate TPS
    if metrics['total_tokens'] > 0:
        metrics['tps'] = metrics['total_tokens'] / metrics['total_time']
    else:
        metrics['tps'] = 0.0
    
    # Calculate TTNT
    if len(metrics['token_timestamps']) > 1:
        ttnts = []
        for i in range(1, len(metrics['token_timestamps'])):
            ttnt = metrics['token_timestamps'][i] - metrics['token_timestamps'][i-1]
            ttnts.append(ttnt)
        metrics['ttnt_mean'] = np.mean(ttnts) if ttnts else 0.0
        metrics['ttnt_std'] = np.std(ttnts) if ttnts else 0.0
        metrics['ttnt_min'] = np.min(ttnts) if ttnts else 0.0
        metrics['ttnt_max'] = np.max(ttnts) if ttnts else 0.0
    else:
        metrics['ttnt_mean'] = 0.0
        metrics['ttnt_std'] = 0.0
        metrics['ttnt_min'] = 0.0
        metrics['ttnt_max'] = 0.0
    
    return metrics


async def run_baseline_evaluation(prompts_file, output_dir, num_runs=1, max_tokens=50):
    """
    Run baseline evaluation on all prompts
    """
    prompts = load_prompts(prompts_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Start GPU monitoring
    gpu_log_file = output_path / f"gpu_log_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Starting GPU monitoring, logging to {gpu_log_file}")
    gpu_process = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "gpu_monitor.py"), 
         "--output", str(gpu_log_file), "--rate", "10.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    all_results = []
    
    try:
        for run_num in range(num_runs):
            print(f"\n{'='*60}")
            print(f"Baseline Run {run_num + 1}/{num_runs}")
            print(f"{'='*60}")
            
            run_results = []
            
            for i, prompt in enumerate(prompts):
                print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")
                
                try:
                    metrics = await generate_baseline_with_metrics(
                        prompt, max_tokens=max_tokens
                    )
                    
                    metrics['run_number'] = run_num + 1
                    metrics['prompt_index'] = i + 1
                    run_results.append(metrics)
                    all_results.append(metrics)
                    
                    print(f"  Tokens: {metrics['total_tokens']}, TPS: {metrics['tps']:.2f}")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save intermediate results
            results_file = output_path / f"baseline_results_run_{run_num + 1}.json"
            with open(results_file, 'w') as f:
                json.dump(run_results, f, indent=2)
            print(f"\nSaved results to {results_file}")
        
        # Save all results
        all_results_file = output_path / f"baseline_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved all results to {all_results_file}")
        
    finally:
        # Stop GPU monitoring
        print("\nStopping GPU monitoring...")
        gpu_process.terminate()
        try:
            gpu_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gpu_process.kill()
        
        print(f"GPU log saved to {gpu_log_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Baseline LLM Evaluation')
    parser.add_argument('--prompts', default='eval/test_prompts.txt', help='Path to prompts file')
    parser.add_argument('--output', default='eval/results', help='Output directory')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per prompt')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    asyncio.run(run_baseline_evaluation(args.prompts, args.output, args.runs, args.max_tokens))

