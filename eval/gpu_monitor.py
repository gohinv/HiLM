"""
GPU Utilization Monitor using nvidia-smi
Samples GPU metrics at specified frequency and logs to CSV
"""
import subprocess
import time
import csv
import sys
from pathlib import Path
from datetime import datetime

def monitor_gpu(output_file, sample_rate_hz=10, duration=None):
    """
    Monitor GPU utilization using nvidia-smi
    
    Args:
        output_file: Path to CSV file for logging
        sample_rate_hz: Sampling frequency (default 10 Hz = every 100ms)
        duration: Maximum duration in seconds (None = until interrupted)
    """
    interval = 1.0 / sample_rate_hz
    
    # nvidia-smi query format
    # Query: timestamp, utilization.gpu, utilization.memory, memory.used, memory.total
    query = "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gpu_util_pct', 'memory_util_pct', 'memory_used_mb', 'memory_total_mb'])
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Run nvidia-smi query
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 5:
                                timestamp = datetime.now().isoformat()
                                gpu_util = parts[1].strip()
                                mem_util = parts[2].strip()
                                mem_used = parts[3].strip()
                                mem_total = parts[4].strip()
                                writer.writerow([timestamp, gpu_util, mem_util, mem_used, mem_total])
                                f.flush()  # Ensure data is written immediately
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\nGPU monitoring stopped. Data saved to {output_path}")
        except Exception as e:
            print(f"Error in GPU monitoring: {e}", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor GPU utilization')
    parser.add_argument('--output', '-o', default='gpu_log.csv', help='Output CSV file')
    parser.add_argument('--rate', '-r', type=float, default=10.0, help='Sampling rate in Hz (default: 10)')
    parser.add_argument('--duration', '-d', type=float, help='Duration in seconds (default: until interrupted)')
    
    args = parser.parse_args()
    monitor_gpu(args.output, args.rate, args.duration)

