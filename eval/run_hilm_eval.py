"""
HiLM Evaluation Script
Runs inference on test prompts and collects comprehensive metrics
"""
import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import signal
import os

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from slm_service import threshold_calc
from slm_service import speculate
from common.rpc_client import UHLMRPCClient
from slm_service import utils
from transformers import AutoTokenizer

# Latency constants
EDGE_LATENCY_MS = 10
DATACENTER_LATENCY_MS = 50
TOTAL_LLM_LATENCY_MS = EDGE_LATENCY_MS + DATACENTER_LATENCY_MS

# Model setup
model, tokenizer = utils.setup("meta-llama/Llama-3.2-1B-Instruct")
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def load_prompts(prompts_file):
    """Load prompts from file, skipping comments and empty lines"""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


async def generate_with_metrics(prompt, max_tokens=50, K=20, theta_max=2.0, use_chat_template=False):
    """
    Generate response with comprehensive metrics collection
    Returns: (response_text, metrics_dict)
    """
    metrics = {
        'prompt': prompt,
        'start_time': time.time(),
        'token_timestamps': [],
        'token_decisions': [],
        'token_uncertainties': [],
        'rpc_calls': {
            'begin_session': 0,
            'verify_token': 0,
            'sync': 0,
            'end_session': 0,
        },
        'acceptance_stats': {
            'skipped': 0,
            'accepted': 0,
            'rejected': 0,
        },
        'decision_counts': {
            'SKIP': 0,
            'VERIFY_ILM': 0,
            'VERIFY_LLM': 0,
        }
    }
    
    # Format prompt
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if prompt.strip().endswith('?'):
            formatted_prompt = f"Q: {prompt}\nA:"
        else:
            formatted_prompt = prompt
    
    # Tokenize prompt
    current_token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    slm_eos_id = tokenizer.eos_token_id
    current_token_ids = [t for t in current_token_ids if t != slm_eos_id]
    
    if not current_token_ids:
        return "", metrics
    
    response_token_ids = []
    
    async with UHLMRPCClient(host="127.0.0.1", port=8081, simulate_latency=False, latency_ms=TOTAL_LLM_LATENCY_MS) as llm, \
                UHLMRPCClient(host="127.0.0.1", port=8082, simulate_latency=False, latency_ms=EDGE_LATENCY_MS) as ilm:
        
        # Begin sessions
        session_start = time.time()
        session_id, llm_eos_token_id = await llm.begin_session(formatted_prompt)
        metrics['rpc_calls']['begin_session'] += 1
        
        ilm_session_id, ilm_eos_token_id = await ilm.begin_session(formatted_prompt)
        metrics['rpc_calls']['begin_session'] += 1
        
        try:
            for step in range(max_tokens):
                token_start = time.time()
                
                # Run SLM to propose draft token
                context_tensor = torch.tensor([current_token_ids]).to(model.device)
                draft = speculate.sample_draft_tokens(
                    model, context_tensor, K=K, theta_max=theta_max, device=model.device
                )
                
                base_draft_id = draft["base_draft_id"]
                base_probs = draft["base_probs"].detach().cpu().numpy()
                
                # Measure uncertainty
                sampled_ids = draft["sampled_ids"]
                u_t = sum(d_k != base_draft_id for d_k in sampled_ids) / len(sampled_ids)
                t1, t2 = threshold_calc.get_threshold()
                
                final_token_id = None
                decision = ""
                accepted = None
                
                # Decision logic
                if u_t < t1:
                    decision = "SKIP"
                    final_token_id = base_draft_id
                    metrics['acceptance_stats']['skipped'] += 1
                    metrics['decision_counts']['SKIP'] += 1
                    
                    await asyncio.gather(
                        ilm.sync(ilm_session_id, [final_token_id]),
                        llm.sync(session_id, [final_token_id])
                    )
                    metrics['rpc_calls']['sync'] += 2
                    
                elif u_t < t2:
                    decision = "VERIFY_ILM"
                    metrics['decision_counts']['VERIFY_ILM'] += 1
                    metrics['rpc_calls']['verify_token'] += 1
                    
                    accepted, final_token_id, _ = await ilm.verify(
                        ilm_session_id, base_draft_id, base_probs
                    )
                    
                    if accepted:
                        metrics['acceptance_stats']['accepted'] += 1
                    else:
                        metrics['acceptance_stats']['rejected'] += 1
                    
                    await llm.sync(session_id, [final_token_id])
                    metrics['rpc_calls']['sync'] += 1
                    
                else:
                    decision = "VERIFY_LLM"
                    metrics['decision_counts']['VERIFY_LLM'] += 1
                    metrics['rpc_calls']['verify_token'] += 1
                    
                    accepted, final_token_id, _ = await llm.verify(
                        session_id, base_draft_id, base_probs
                    )
                    
                    if accepted:
                        metrics['acceptance_stats']['accepted'] += 1
                    else:
                        metrics['acceptance_stats']['rejected'] += 1
                    
                    await ilm.sync(ilm_session_id, [final_token_id])
                    metrics['rpc_calls']['sync'] += 1
                
                # Check EOS
                if decision == "VERIFY_ILM" and int(final_token_id) == int(ilm_eos_token_id):
                    break
                elif decision == "VERIFY_LLM" and int(final_token_id) == int(llm_eos_token_id):
                    break
                elif decision == "SKIP" and int(final_token_id) == int(slm_eos_id):
                    break
                
                # Append token
                current_token_ids.append(final_token_id)
                response_token_ids.append(final_token_id)
                
                # Record metrics
                token_end = time.time()
                metrics['token_timestamps'].append(token_end)
                metrics['token_decisions'].append({
                    'step': step + 1,
                    'decision': decision,
                    'accepted': accepted if accepted is not None else True,  # SKIP is always "accepted"
                    'token_id': int(final_token_id),
                })
                metrics['token_uncertainties'].append(float(u_t))
                
        finally:
            await llm.end_session(session_id)
            await ilm.end_session(ilm_session_id)
            metrics['rpc_calls']['end_session'] += 2
    
    # Decode response
    decoded = tokenizer.decode(response_token_ids, skip_special_tokens=True)
    metrics['end_time'] = time.time()
    metrics['response_text'] = decoded
    metrics['total_tokens'] = len(response_token_ids)
    metrics['total_time'] = metrics['end_time'] - metrics['start_time']
    
    # Calculate derived metrics
    if metrics['total_tokens'] > 0:
        metrics['tps'] = metrics['total_tokens'] / metrics['total_time']
    else:
        metrics['tps'] = 0.0
    
    # Calculate TTNT (Time to Next Token)
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
    
    # Acceptance rate
    total_decisions = metrics['acceptance_stats']['skipped'] + metrics['acceptance_stats']['accepted'] + metrics['acceptance_stats']['rejected']
    if total_decisions > 0:
        metrics['acceptance_rate'] = (metrics['acceptance_stats']['skipped'] + metrics['acceptance_stats']['accepted']) / total_decisions
    else:
        metrics['acceptance_rate'] = 0.0
    
    return decoded, metrics


async def run_evaluation(prompts_file, output_dir, num_runs=1, max_tokens=50):
    """
    Run evaluation on all prompts
    """
    prompts = load_prompts(prompts_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Start GPU monitoring
    gpu_log_file = output_path / f"gpu_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
            print(f"Run {run_num + 1}/{num_runs}")
            print(f"{'='*60}")
            
            run_results = []
            
            for i, prompt in enumerate(prompts):
                print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")
                
                try:
                    response, metrics = await generate_with_metrics(
                        prompt, max_tokens=max_tokens, use_chat_template=False
                    )
                    
                    metrics['run_number'] = run_num + 1
                    metrics['prompt_index'] = i + 1
                    run_results.append(metrics)
                    all_results.append(metrics)
                    
                    print(f"  Tokens: {metrics['total_tokens']}, TPS: {metrics['tps']:.2f}, "
                          f"Acceptance: {metrics['acceptance_rate']:.2%}")
                    print(f"  Response: {response}")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
            
            # Save intermediate results
            results_file = output_path / f"results_run_{run_num + 1}.json"
            with open(results_file, 'w') as f:
                json.dump(run_results, f, indent=2)
            print(f"\nSaved results to {results_file}")
        
        # Save all results
        all_results_file = output_path / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    parser = argparse.ArgumentParser(description='HiLM Evaluation')
    parser.add_argument('--prompts', default='eval/test_prompts.txt', help='Path to prompts file')
    parser.add_argument('--output', default='eval/results', help='Output directory')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per prompt')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args.prompts, args.output, args.runs, args.max_tokens))

