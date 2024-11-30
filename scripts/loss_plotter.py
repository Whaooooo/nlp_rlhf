import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process_log_files(input_dir, model_keys_dict):
    """
    Processes the log files in the input directory and generates plots for specified models and keys.

    Parameters:
    - input_dir: Path to the directory containing the log files.
    - model_keys_dict: Dictionary where keys are model names and values are lists of keys to plot.
    """

    # Initialize a dictionary to store data for each model
    data_dict = {}
    processed_steps = {}

    # Define the paths to the specific log files
    master_worker_path = os.path.join(input_dir, "master_worker-0")
    model_worker_path = os.path.join(input_dir, "model_worker-0")

    # Regular expression patterns to extract information
    pattern_old = r"RPC name\s*(.*?)\s*returns\s*(\{.*\})"
    pattern_new = r"CGPO Actor\s*(.*?)\s*step\s*(\d+)\s*returns\s*(\{.*\})"

    # Function to process a single file with a given pattern
    def process_file(file_path, pattern, is_new_pattern=False):
        if not os.path.isfile(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            return

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ('RPC name' in line and 'returns' in line) or \
                   (is_new_pattern and 'CGPO Actor' in line and 'step' in line and 'returns' in line):
                    match = re.search(pattern, line)
                    if match:
                        if is_new_pattern:
                            model_name_str = match.group(1)
                            global_step_str = match.group(2)
                            dict_str = match.group(3)
                            # Clean up model_name and parse global_step
                            model_name = model_name_str.strip('{}').strip()
                            try:
                                global_step = int(global_step_str)
                            except ValueError:
                                print(f"Invalid step value: {global_step_str} in line: {line}")
                                continue
                            if model_name in model_keys_dict:
                                # Check for duplicate steps
                                if model_name not in processed_steps:
                                    processed_steps[model_name] = set()
                                if global_step in processed_steps[model_name]:
                                    continue  # Skip duplicate step
                                else:
                                    processed_steps[model_name].add(global_step)
                                step = global_step
                        else:
                            model_name_str = match.group(1)
                            dict_str = match.group(2)
                            # Clean up model_name
                            model_name = model_name_str.strip('{}').strip()
                            if model_name in model_keys_dict:
                                step = len(data_dict[model_name]['step']) if model_name in data_dict else 0

                        # Parse the dict_str
                        dict_str_clean = dict_str.replace("'", '"')
                        try:
                            data_dict_line = json.loads(dict_str_clean)
                        except json.JSONDecodeError:
                            print(f"JSON decode error for line: {line}")
                            continue

                        # Get the keys we are interested in
                        keys = model_keys_dict.get(model_name, [])
                        if not keys:
                            continue

                        # Initialize data_dict for this model_name if not exist
                        if model_name not in data_dict:
                            data_dict[model_name] = {key: [] for key in keys}
                            data_dict[model_name]['step'] = []
                        # Append values for each key
                        for key in keys:
                            value = data_dict_line.get(key, np.nan)
                            if isinstance(value, (int, float)):
                                data_dict[model_name][key].append(value)
                            else:
                                try:
                                    value_float = float(value)
                                    data_dict[model_name][key].append(value_float)
                                except:
                                    data_dict[model_name][key].append(np.nan)
                        data_dict[model_name]['step'].append(step)

    # Process the master_worker-0 file with the old pattern
    process_file(master_worker_path, pattern_old, is_new_pattern=False)

    # Process the model_worker-0 file with the new pattern
    process_file(model_worker_path, pattern_new, is_new_pattern=True)

    # Prepare to plot the data for each model
    for model_name in data_dict:
        data = data_dict[model_name]
        df = pd.DataFrame(data)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.sort_values('step')  # Ensure data is sorted by step

        if df.empty:
            print(f"No valid data for model {model_name}. Skipping plot.")
            continue

        # Apply exponentially weighted moving average to smooth the curve
        window_size = 50  # Adjust the window size as needed
        df_smooth = df.ewm(span=window_size, min_periods=1, adjust=True).mean()

        # Plotting
        keys = model_keys_dict[model_name]
        num_keys = len(keys)
        fig, axes = plt.subplots(nrows=num_keys, ncols=1, sharex=True, figsize=(12, 4 * num_keys))
        if num_keys == 1:
            axes = [axes]
        colors = plt.cm.get_cmap('tab10', num_keys)
        for i, key in enumerate(keys):
            ax = axes[i]
            key_data = df_smooth[key].dropna()
            if not key_data.empty:
                p90 = np.percentile(key_data, 95)
                p10 = np.percentile(key_data, 5)
                y_max = 3 * p90 - 2 * p10
                y_min = 3 * p10 - 2 * p90
                ax.plot(df_smooth['step'], df_smooth[key], color=colors(i))
                ax.set_ylabel(key, fontsize=12)
                ax.set_ylim(y_min, y_max)
                ax.set_title(f'{model_name} - {key}', fontsize=14)
                ax.grid(True)
        axes[-1].set_xlabel('Step', fontsize=12)
        plt.tight_layout()
        fig_path = os.path.join(input_dir, f'{model_name}.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Plot saved for model {model_name} at {fig_path}")

# Example usage
if __name__ == "__main__":
    input_dir = "/mnt/zzo/.cache/realhf/logs/root/mistral-mix-2task-cgpo/debug1/"
    model_keys_dict = {
        "actor0": ["actor_loss", "importance_weight", "task_reward", "n_tokens"],
        "actor1": ["actor_loss", "importance_weight", "task_reward", "n_tokens"],
        "critic_train_0": ["value_loss", "returns", "n_tokens"],
        "critic_train_1": ["value_loss", "returns", "n_tokens"],
    }
    process_log_files(input_dir, model_keys_dict)
