"""
BERTScore Computation Script
Computes BERTScore between HiLM and baseline outputs from saved results
Handles multiple runs per prompt and averages across runs as specified in requirements
"""
import json
import sys
from pathlib import Path
import numpy as np
from bert_score import score
from transformers import AutoTokenizer
from collections import defaultdict

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def load_results(results_file):
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def decode_baseline_tokens(token_ids, tokenizer):
    """Decode token IDs to text using tokenizer"""
    if not token_ids:
        return ""
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded
    except Exception as e:
        print(f"Warning: Could not decode tokens: {e}")
        return ""


def extract_responses_by_prompt_and_run(results, is_baseline=False, tokenizer=None):
    """
    Extract response texts grouped by prompt and run_number
    
    Returns: dict of {(prompt, run_number): response_text}
    """
    responses_dict = {}
    
    for result in results:
        prompt = result.get('prompt', '')
        run_number = result.get('run_number', 1)  # Default to 1 if not present
        key = (prompt, run_number)
        
        if is_baseline:
            # Baseline stores token_ids, need to decode
            token_ids = result.get('token_ids', [])
            if tokenizer:
                response_text = decode_baseline_tokens(token_ids, tokenizer)
            else:
                # If no tokenizer, try to use response_text if it exists
                response_text = result.get('response_text', '')
                if not response_text:
                    print(f"Warning: No tokenizer provided and no response_text for baseline. Skipping {key}.")
                    response_text = ""
            responses_dict[key] = response_text
        else:
            # HiLM stores response_text directly
            response_text = result.get('response_text', '')
            responses_dict[key] = response_text
    
    return responses_dict


def compute_bertscore(hilm_results_file, baseline_results_file, output_file=None, model_type='microsoft/deberta-xlarge-mnli'):
    """
    Compute BERTScore between HiLM and baseline outputs
    Handles multiple runs per prompt and averages across runs
    
    Args:
        hilm_results_file: Path to HiLM results JSON
        baseline_results_file: Path to baseline results JSON
        output_file: Optional path to save BERTScore results
        model_type: BERTScore model to use (default: deberta-xlarge-mnli)
    """
    print("Loading results...")
    hilm_results = load_results(hilm_results_file)
    baseline_results = load_results(baseline_results_file)
    
    print(f"HiLM results: {len(hilm_results)} queries")
    print(f"Baseline results: {len(baseline_results)} queries")
    
    # Check if we need to decode baseline tokens
    needs_decoding = False
    if baseline_results and 'token_ids' in baseline_results[0]:
        if 'response_text' not in baseline_results[0] or not baseline_results[0].get('response_text'):
            needs_decoding = True
    
    tokenizer = None
    if needs_decoding:
        print("\nBaseline results contain token_ids. Loading tokenizer to decode...")
        # Use the LLM tokenizer (Llama-3.1-8B-Instruct)
        try:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Trying alternative: meta-llama/Llama-3.2-1B-Instruct")
            try:
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
                print("Tokenizer loaded successfully.")
            except Exception as e2:
                print(f"Error loading alternative tokenizer: {e2}")
                print("Cannot decode baseline tokens. Please ensure tokenizer is available.")
                return None
    
    print("\nExtracting responses grouped by prompt and run...")
    hilm_responses = extract_responses_by_prompt_and_run(hilm_results, is_baseline=False)
    baseline_responses = extract_responses_by_prompt_and_run(baseline_results, is_baseline=True, tokenizer=tokenizer)
    
    print(f"HiLM: {len(hilm_responses)} (prompt, run) pairs")
    print(f"Baseline: {len(baseline_responses)} (prompt, run) pairs")
    
    # Find common (prompt, run) pairs
    common_keys = set(hilm_responses.keys()) & set(baseline_responses.keys())
    print(f"Found {len(common_keys)} matching (prompt, run) pairs")
    
    if len(common_keys) == 0:
        print("ERROR: No matching (prompt, run) pairs found!")
        print("This might indicate different prompt sets, run numbers, or formatting issues.")
        return None
    
    # Filter out empty responses
    valid_keys = [k for k in common_keys if hilm_responses[k] and baseline_responses[k]]
    if len(valid_keys) < len(common_keys):
        print(f"Warning: {len(common_keys) - len(valid_keys)} empty responses filtered out")
    
    if len(valid_keys) == 0:
        print("ERROR: No valid response pairs found!")
        return None
    
    # Group by prompt to compute per-prompt averages
    prompt_groups = defaultdict(list)
    for key in valid_keys:
        prompt, run_num = key
        prompt_groups[prompt].append((run_num, key))
    
    print(f"\nFound {len(prompt_groups)} unique prompts")
    print(f"Average {len(valid_keys) / len(prompt_groups):.1f} runs per prompt")
    
    # Compute BERTScore for all pairs first
    print(f"\nComputing BERTScore for {len(valid_keys)} response pairs...")
    print(f"Using model: {model_type}")
    print("This may take a few minutes...")
    
    hilm_texts = [hilm_responses[k] for k in valid_keys]
    baseline_texts = [baseline_responses[k] for k in valid_keys]
    
    # Compute BERTScore
    # Reference = baseline (ground truth), Candidate = HiLM (what we're evaluating)
    P, R, F1 = score(
        hilm_texts,           # candidates (HiLM outputs)
        baseline_texts,       # references (baseline outputs)
        lang='en',
        model_type=model_type,
        verbose=True
    )
    
    # Convert to numpy arrays
    P = P.cpu().numpy()
    R = R.cpu().numpy()
    F1 = F1.cpu().numpy()
    
    # Create mapping from key to scores
    scores_dict = {key: (p, r, f1) for key, p, r, f1 in zip(valid_keys, P, R, F1)}
    
    # Compute per-prompt averages (across runs)
    per_prompt_results = []
    for prompt in sorted(prompt_groups.keys()):
        runs = prompt_groups[prompt]
        run_scores = [scores_dict[key] for _, key in runs]
        
        # Average across runs for this prompt
        avg_p = np.mean([p for p, _, _ in run_scores])
        avg_r = np.mean([r for _, r, _ in run_scores])
        avg_f1 = np.mean([f1 for _, _, f1 in run_scores])
        
        # Get sample responses (from first run)
        first_key = runs[0][1]
        hilm_sample = hilm_responses[first_key]
        baseline_sample = baseline_responses[first_key]
        
        per_prompt_results.append({
            'prompt': prompt,
            'num_runs': len(runs),
            'hilm_response_sample': hilm_sample[:200] + "..." if len(hilm_sample) > 200 else hilm_sample,
            'baseline_response_sample': baseline_sample[:200] + "..." if len(baseline_sample) > 200 else baseline_sample,
            'precision_mean': float(avg_p),
            'recall_mean': float(avg_r),
            'f1_mean': float(avg_f1),
            'run_scores': [
                {
                    'run_number': run_num,
                    'precision': float(p),
                    'recall': float(r),
                    'f1': float(f1)
                }
                for run_num, key in runs
                for p, r, f1 in [scores_dict[key]]
            ]
        })
    
    # Compute overall statistics
    all_p = [p for p, _, _ in scores_dict.values()]
    all_r = [r for _, r, _ in scores_dict.values()]
    all_f1 = [f1 for _, _, f1 in scores_dict.values()]
    
    results = {
        'model_type': model_type,
        'total_pairs': len(valid_keys),
        'num_prompts': len(prompt_groups),
        'avg_runs_per_prompt': len(valid_keys) / len(prompt_groups),
        'precision': {
            'mean': float(np.mean(all_p)),
            'std': float(np.std(all_p)),
            'min': float(np.min(all_p)),
            'max': float(np.max(all_p)),
        },
        'recall': {
            'mean': float(np.mean(all_r)),
            'std': float(np.std(all_r)),
            'min': float(np.min(all_r)),
            'max': float(np.max(all_r)),
        },
        'f1': {
            'mean': float(np.mean(all_f1)),
            'std': float(np.std(all_f1)),
            'min': float(np.min(all_f1)),
            'max': float(np.max(all_f1)),
        },
        'per_prompt_averages': per_prompt_results,
        # Also include per-prompt mean F1 for easy analysis
        'per_prompt_f1_means': [r['f1_mean'] for r in per_prompt_results]
    }
    
    # Print summary
    print("\n" + "="*60)
    print("BERTScore Results")
    print("="*60)
    print(f"\nModel: {model_type}")
    print(f"Total response pairs: {len(valid_keys)}")
    print(f"Number of unique prompts: {len(prompt_groups)}")
    print(f"Average runs per prompt: {results['avg_runs_per_prompt']:.2f}")
    print(f"\nOverall Statistics (across all runs):")
    print(f"\nPrecision (P):")
    print(f"  Mean: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
    print(f"  Range: [{results['precision']['min']:.4f}, {results['precision']['max']:.4f}]")
    print(f"\nRecall (R):")
    print(f"  Mean: {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")
    print(f"  Range: [{results['recall']['min']:.4f}, {results['recall']['max']:.4f}]")
    print(f"\nF1 Score:")
    print(f"  Mean: {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
    print(f"  Range: [{results['f1']['min']:.4f}, {results['f1']['max']:.4f}]")
    
    # Print per-prompt average F1 statistics
    per_prompt_f1s = results['per_prompt_f1_means']
    print(f"\nPer-Prompt Average F1 (averaged across runs):")
    print(f"  Mean: {np.mean(per_prompt_f1s):.4f} ± {np.std(per_prompt_f1s):.4f}")
    print(f"  Range: [{np.min(per_prompt_f1s):.4f}, {np.max(per_prompt_f1s):.4f}]")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute BERTScore between HiLM and baseline outputs')
    parser.add_argument('--hilm-results', required=True, help='Path to HiLM results JSON file')
    parser.add_argument('--baseline-results', required=True, help='Path to baseline results JSON file')
    parser.add_argument('--output', default='bertscore_results.json', help='Output JSON file for BERTScore results')
    parser.add_argument('--model', default='microsoft/deberta-xlarge-mnli', 
                       help='BERTScore model to use (default: microsoft/deberta-xlarge-mnli)')
    
    args = parser.parse_args()
    
    results = compute_bertscore(
        args.hilm_results,
        args.baseline_results,
        args.output,
        args.model
    )
    
    if results is None:
        sys.exit(1)

