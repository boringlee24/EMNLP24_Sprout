import subprocess
import itertools
import os
from joblib import Parallel, delayed

# Define the options for each argument
regions = ['NL', 'AU-SA', 'GB', "US-TEX", "US-CAL"]
optimizers = ["static"]#['baseline', 'clover', 'co2opt', "lp", "oracle", "static"]
start_hours = ['800']#, '3900', '7000']

# pref_torelance from 0.02 to 0.3
pref_torelances = ["0.1"]#[]
# for i in range(2, 31, 1):
#     pref_torelances.append(str(i/100))

logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

def run_command(region, optimizer, start_hr, pref_torelance):
    # Construct the command
    command = [
        'python', 'main.py',
        '--region', region,
        '--optimizer_name', optimizer,
        '--pref_torelance', pref_torelance,
        '--start_hour', start_hr
    ]
    log_filename = f"{region}_{optimizer}_{start_hr}_{pref_torelance}"
    stdout_path = os.path.join(logs_dir, f"{log_filename}_stdout.log")
    stderr_path = os.path.join(logs_dir, f"{log_filename}_stderr.log")

    with open(stdout_path, 'w') as stdout_file, open(stderr_path, 'w') as stderr_file:
        subprocess.run(command, stdout=stdout_file, stderr=stderr_file, text=True)

    print(f'Completed: {region}, {optimizer}, {start_hr}, {pref_torelance}')


# Generate all combinations of options
combinations = list(itertools.product(regions, optimizers, start_hours, pref_torelances))

# Use joblib to parallelize the subprocess execution
Parallel(n_jobs=-1)(delayed(run_command)(region, optimizer, start_hr, pref_torelance) for region, optimizer, start_hr, pref_torelance in combinations)
