import pandas as pd
import subprocess
import os
import time
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
PYTHON_PATH = "/train-data-5-hdd/charles/project-oma/.venv/bin/python"  # Or your .venv/bin/python path
SCRIPT_PATH = "src/train.py"
CSV_PATH = "configs/nameofHPOconfigs.csv"
PROJECT_NAME = "OMA-nameofHPOmethod"

# Specify exact GPU IDs available for use (e.g., [0, 2] if only 0 and 2 are free)
AVAILABLE_GPUS = [0, 1, 2, 3]  
# Number of concurrent experiments to run on EACH GPU to maximize memory
RUNS_PER_GPU = 2  
CONCURRENT_RUNS = len(AVAILABLE_GPUS) * RUNS_PER_GPU

# Skip everything except 11 and 12
# SKIP_CONFIG_IDS = {
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
# } # Set of config_ids to skip (already run)
SKIP_CONFIG_IDS = None

def run_experiment(row_data, gpu_id):
    """Function to run a single CleanRL command on a specific GPU."""
    config_id = int(row_data['config_id'])
    seed = int(row_data['seed'])
    
    # Construct the command using the column names from your CSV
    cmd = [
        PYTHON_PATH, SCRIPT_PATH,
        "--run_name", f"cfg{config_id}_s{seed}",
        "--seed", str(seed),
        "--lr", str(row_data['lr']),
        "--momentum", str(row_data['momentum']),
        "--weight_decay", str(row_data['weight_decay']),
        "--epochs", str(int(row_data['epochs'])),
        "--batch_size", str(int(row_data['batch_size'])),
        "--dropout_rate", str(row_data['dropout_rate']),
        "--conv_ch1", str(int(row_data['conv_ch1'])),
        "--conv_ch2", str(int(row_data['conv_ch2'])),
        "--fc1_dim", str(int(row_data['fc1_dim'])),
        "--fc2_dim", str(int(row_data['fc2_dim'])),
        "--wandb_project_name", PROJECT_NAME,
    ]

    # Assign the specific GPU to this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Starting Config {config_id} (Seed {seed}) on GPU {gpu_id}...")
    
    # Start the process and wait for it to finish
    with open(f"logs/config_{config_id}_s{seed}.log", "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=log_file)
    
    print(f"Finished Config {config_id}. Status: {result.returncode}")
    return config_id

def main():
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    os.makedirs("logs", exist_ok=True)

    # 2. Manage the Queue
    # Create the pool of available GPUs, factoring in runs per GPU
    available_gpus = AVAILABLE_GPUS * RUNS_PER_GPU
    
    # Track active processes: {future: gpu_id}
    active_runs = {}
    
    with ProcessPoolExecutor(max_workers=CONCURRENT_RUNS) as executor:
        row_idx = 0
        
        while row_idx < len(df) or active_runs:
            # If we have free GPUs and more rows to process, launch them
            while len(active_runs) < CONCURRENT_RUNS and row_idx < len(df):
                gpu_id = available_gpus.pop(0)
                row = df.iloc[row_idx].to_dict()
                
                if SKIP_CONFIG_IDS and int(row['config_id']) in SKIP_CONFIG_IDS:
                    print(f"Skipping Config {int(row['config_id'])} (Seed {int(row['seed'])}) as it's already run.")
                    available_gpus.append(gpu_id) # Return the GPU to the pool
                    row_idx += 1
                    continue
                
                future = executor.submit(run_experiment, row, gpu_id)
                active_runs[future] = gpu_id
                row_idx += 1

            # Check for finished processes
            done_futures = [f for f in active_runs if f.done()]
            for f in done_futures:
                gpu_id = active_runs.pop(f)
                available_gpus.append(gpu_id) # Return the GPU to the pool
            
            time.sleep(30) # Wait 30 seconds before checking again

if __name__ == "__main__":
    main()
