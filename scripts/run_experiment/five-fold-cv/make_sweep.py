import yaml
import json
from astra.constants import PROJECT_ROOT

# 1. Read the master list to find out how many experiments there are.
with open("master_experiment_list.json", 'r') as f:
    num_experiments = len(json.load(f))

print(f"Found {num_experiments} experiments in master_experiment_list.json")

# 2. Define the sweep configuration as a Python dictionary.
sweep_config = {
    'method': 'grid',
    'project': 'astra-cv-controller',
    'name': '5-Fold-CV-Execution-Queue',
    'parameters': {
        'experiment_index': {
            # 3. Programmatically generate the list of indices.
            'values': list(range(num_experiments))
        }
    },
    'command': [
        '${env}',
        'uv',
        'run',
        '../../../scripts/run_experiment/five-fold-cv/run_cv_job.py',
        '${args}'
    ]
}

# 4. Create directory.
output_dir = PROJECT_ROOT / "configs/experiments/five-fold-cv/"
output_dir.mkdir(parents=True, exist_ok=True)

# 5. Write the dictionary to a YAML file.
output_file = output_dir / "sweep_cv.yaml"

with open(output_file, 'w') as f:
    yaml.dump(sweep_config, f, sort_keys=False, default_flow_style=False)

print(f"Successfully created '{output_file}' with {num_experiments} jobs.")