"""
Results Analysis Script
Computes normalized metrics, averages, and generates comparison report
"""
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_gpu_log(gpu_log_file):
    """Load GPU utilization log and compute average utilization"""
    try:
        df = pd.read_csv(gpu_log_file)
        if 'gpu_util_pct' in df.columns:
            # Convert to float, handle any non-numeric values
            df['gpu_util_pct'] = pd.to_numeric(df['gpu_util_pct'], errors='coerce')
            avg_util = df['gpu_util_pct'].mean() / 100.0  # Convert to 0-1 range
            return avg_util
        return None
    except Exception as e:
        print(f"Warning: Could not load GPU log {gpu_log_file}: {e}")
        return None


def compute_normalized_tps(tps, avg_gpu_util):
    """Compute normalized TPS"""
    if avg_gpu_util and avg_gpu_util > 0:
        return tps / avg_gpu_util
    return None


def analyze_results(results_file, gpu_log_file=None, system_name="HiLM"):
    """Analyze results from a single results file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load GPU utilization if available
    avg_gpu_util = None
    if gpu_log_file and Path(gpu_log_file).exists():
        avg_gpu_util = load_gpu_log(gpu_log_file)
        print(f"Average GPU Utilization: {avg_gpu_util*100:.2f}%")
    
    # Aggregate metrics
    metrics = {
        'total_queries': len(results),
        'tps_values': [],
        'tps_norm_values': [],
        'ttnt_mean_values': [],
        'ttnt_std_values': [],
        'total_time_values': [],
        'total_tokens_values': [],
        'acceptance_rates': [],
        'rpc_calls_total': defaultdict(int),
        'decision_counts': defaultdict(int),
    }
    
    for result in results:
        # TPS
        if 'tps' in result:
            metrics['tps_values'].append(result['tps'])
            if avg_gpu_util:
                tps_norm = compute_normalized_tps(result['tps'], avg_gpu_util)
                if tps_norm:
                    metrics['tps_norm_values'].append(tps_norm)
        
        # TTNT
        if 'ttnt_mean' in result:
            metrics['ttnt_mean_values'].append(result['ttnt_mean'])
        if 'ttnt_std' in result:
            metrics['ttnt_std_values'].append(result['ttnt_std'])
        
        # Total time
        if 'total_time' in result:
            metrics['total_time_values'].append(result['total_time'])
        
        # Total tokens
        if 'total_tokens' in result:
            metrics['total_tokens_values'].append(result['total_tokens'])
        
        # Acceptance rate
        if 'acceptance_rate' in result:
            metrics['acceptance_rates'].append(result['acceptance_rate'])
        
        # RPC calls
        if 'rpc_calls' in result:
            for call_type, count in result['rpc_calls'].items():
                metrics['rpc_calls_total'][call_type] += count
        
        # Decision counts
        if 'decision_counts' in result:
            for decision, count in result['decision_counts'].items():
                metrics['decision_counts'][decision] += count
    
    # Compute statistics
    stats = {
        'system': system_name,
        'avg_gpu_util': avg_gpu_util,
    }
    
    if metrics['tps_values']:
        stats['tps_mean'] = np.mean(metrics['tps_values'])
        stats['tps_std'] = np.std(metrics['tps_values'])
        stats['tps_min'] = np.min(metrics['tps_values'])
        stats['tps_max'] = np.max(metrics['tps_values'])
    
    if metrics['tps_norm_values']:
        stats['tps_norm_mean'] = np.mean(metrics['tps_norm_values'])
        stats['tps_norm_std'] = np.std(metrics['tps_norm_values'])
    
    if metrics['ttnt_mean_values']:
        stats['ttnt_mean'] = np.mean(metrics['ttnt_mean_values'])
        stats['ttnt_std'] = np.std(metrics['ttnt_mean_values'])
        stats['ttnt_min'] = np.min(metrics['ttnt_mean_values'])
        stats['ttnt_max'] = np.max(metrics['ttnt_mean_values'])
    
    if metrics['total_time_values']:
        stats['e2e_latency_mean'] = np.mean(metrics['total_time_values'])
        stats['e2e_latency_std'] = np.std(metrics['total_time_values'])
    
    if metrics['acceptance_rates']:
        stats['acceptance_rate_mean'] = np.mean(metrics['acceptance_rates'])
        stats['acceptance_rate_std'] = np.std(metrics['acceptance_rates'])
    
    stats['rpc_calls'] = dict(metrics['rpc_calls_total'])
    stats['decision_counts'] = dict(metrics['decision_counts'])
    
    return stats


def compare_results(hilm_results_file, hilm_gpu_log, baseline_results_file, baseline_gpu_log, output_file):
    """Compare HiLM and baseline results"""
    print("Analyzing HiLM results...")
    hilm_stats = analyze_results(hilm_results_file, hilm_gpu_log, "HiLM")
    
    print("\nAnalyzing Baseline results...")
    baseline_stats = analyze_results(baseline_results_file, baseline_gpu_log, "Baseline")
    
    # Compute improvements
    comparison = {
        'hilm': hilm_stats,
        'baseline': baseline_stats,
        'improvements': {}
    }
    
    if 'tps_mean' in hilm_stats and 'tps_mean' in baseline_stats:
        tps_improvement = ((hilm_stats['tps_mean'] - baseline_stats['tps_mean']) / baseline_stats['tps_mean']) * 100
        comparison['improvements']['tps_pct'] = tps_improvement
    
    if 'tps_norm_mean' in hilm_stats and 'tps_norm_mean' in baseline_stats:
        tps_norm_improvement = ((hilm_stats['tps_norm_mean'] - baseline_stats['tps_norm_mean']) / baseline_stats['tps_norm_mean']) * 100
        comparison['improvements']['tps_norm_pct'] = tps_norm_improvement
    
    if 'ttnt_mean' in hilm_stats and 'ttnt_mean' in baseline_stats:
        ttnt_improvement = ((baseline_stats['ttnt_mean'] - hilm_stats['ttnt_mean']) / baseline_stats['ttnt_mean']) * 100
        comparison['improvements']['ttnt_pct'] = ttnt_improvement
    
    if 'e2e_latency_mean' in hilm_stats and 'e2e_latency_mean' in baseline_stats:
        latency_improvement = ((baseline_stats['e2e_latency_mean'] - hilm_stats['e2e_latency_mean']) / baseline_stats['e2e_latency_mean']) * 100
        comparison['improvements']['e2e_latency_pct'] = latency_improvement
    
    # Save comparison
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nTPS (Tokens Per Second):")
    print(f"  HiLM:     {hilm_stats.get('tps_mean', 'N/A'):.2f} ± {hilm_stats.get('tps_std', 0):.2f}")
    print(f"  Baseline: {baseline_stats.get('tps_mean', 'N/A'):.2f} ± {baseline_stats.get('tps_std', 0):.2f}")
    if 'tps_pct' in comparison['improvements']:
        print(f"  Improvement: {comparison['improvements']['tps_pct']:.2f}%")
    
    if 'tps_norm_mean' in hilm_stats:
        print(f"\nNormalized TPS:")
        print(f"  HiLM:     {hilm_stats['tps_norm_mean']:.2f} ± {hilm_stats.get('tps_norm_std', 0):.2f}")
        if 'tps_norm_mean' in baseline_stats:
            print(f"  Baseline: {baseline_stats['tps_norm_mean']:.2f} ± {baseline_stats.get('tps_norm_std', 0):.2f}")
            if 'tps_norm_pct' in comparison['improvements']:
                print(f"  Improvement: {comparison['improvements']['tps_norm_pct']:.2f}%")
    
    print(f"\nTTNT (Time to Next Token):")
    print(f"  HiLM:     {hilm_stats.get('ttnt_mean', 'N/A'):.4f}s ± {hilm_stats.get('ttnt_std', 0):.4f}s")
    print(f"  Baseline: {baseline_stats.get('ttnt_mean', 'N/A'):.4f}s ± {baseline_stats.get('ttnt_std', 0):.4f}s")
    if 'ttnt_pct' in comparison['improvements']:
        print(f"  Improvement: {comparison['improvements']['ttnt_pct']:.2f}%")
    
    print(f"\nEnd-to-End Latency:")
    print(f"  HiLM:     {hilm_stats.get('e2e_latency_mean', 'N/A'):.4f}s ± {hilm_stats.get('e2e_latency_std', 0):.4f}s")
    print(f"  Baseline: {baseline_stats.get('e2e_latency_mean', 'N/A'):.4f}s ± {baseline_stats.get('e2e_latency_std', 0):.4f}s")
    if 'e2e_latency_pct' in comparison['improvements']:
        print(f"  Improvement: {comparison['improvements']['e2e_latency_pct']:.2f}%")
    
    print(f"\nAcceptance Rate:")
    print(f"  HiLM:     {hilm_stats.get('acceptance_rate_mean', 'N/A'):.2%} ± {hilm_stats.get('acceptance_rate_std', 0):.2%}")
    
    print(f"\nRPC Calls (HiLM):")
    for call_type, count in hilm_stats.get('rpc_calls', {}).items():
        print(f"  {call_type}: {count}")
    
    print(f"\nDecision Counts (HiLM):")
    for decision, count in hilm_stats.get('decision_counts', {}).items():
        print(f"  {decision}: {count}")
    
    print(f"\nFull comparison saved to {output_file}")
    
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--hilm-results', help='HiLM results JSON file')
    parser.add_argument('--hilm-gpu-log', help='HiLM GPU log CSV file')
    parser.add_argument('--baseline-results', help='Baseline results JSON file')
    parser.add_argument('--baseline-gpu-log', help='Baseline GPU log CSV file')
    parser.add_argument('--output', default='eval/comparison.json', help='Output comparison file')
    
    args = parser.parse_args()
    
    if args.hilm_results and args.baseline_results:
        compare_results(
            args.hilm_results, args.hilm_gpu_log,
            args.baseline_results, args.baseline_gpu_log,
            args.output
        )
    elif args.hilm_results:
        stats = analyze_results(args.hilm_results, args.hilm_gpu_log, "HiLM")
        print("\nHiLM Statistics:")
        print(json.dumps(stats, indent=2))
    elif args.baseline_results:
        stats = analyze_results(args.baseline_results, args.baseline_gpu_log, "Baseline")
        print("\nBaseline Statistics:")
        print(json.dumps(stats, indent=2))
    else:
        print("Please provide at least one results file")

