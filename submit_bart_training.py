import os
import subprocess

# Directory where the SLURM job scripts are located
current_dir = os.getcwd()

# Dataset names and sizes
datasets = ["gold", "silver"]
sizes = [100, 1000, 10000]

# Loop through each dataset and size to submit the scripts
for dataset in datasets:
    for size in sizes:
        script_name = f"{current_dir}/train_bart_{dataset}_{size}.sh"
        if os.path.isfile(script_name):
            print(f"Submitting job script {script_name}")
            subprocess.run(["sbatch", script_name])
        else:
            print(f"Job script {script_name} does not exist")
