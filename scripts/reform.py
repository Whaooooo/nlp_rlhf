import os 
import json
import re
import numpy as np


def process_file(file_path):
    # Load the original data from the provided file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Load example data from the airoboros-33b model outputs
    with open('/home/zzo/Quickstart/alpaca_eval/my_results/mistral-7b-sft-alpha_greedy/mistral-7b-sft-alpha_greedy.json', 'r') as example_file:
        example_data = json.load(example_file)
    
    # Create the instruction order list from example data
    instruction_order = [item['instruction'] for item in example_data]
    new_data = []
    for item in data:
        item['dataset'] = 'helpful_base'
        new_data.append(item)

    # Sort new_data according to instruction_order
    def get_instruction_index(new_item):
        try:
            return instruction_order.index(new_item['instruction'])
        except ValueError:
            return len(instruction_order)  # If instruction is not found, place it at the end

    new_data.sort(key=get_instruction_index)

    # Create the 'sorted/filename' folder structure
    base_file_name = os.path.basename(file_path).replace(".json", "")  # Get the filename without extension
    dir_name = os.path.dirname(file_path)  # Get the directory of the file_path
    sorted_dir = os.path.join(dir_name, 'sorted', base_file_name)  # Define the new 'sorted/filename' folder path

    # Create the sorted directory if it doesn't exist
    os.makedirs(sorted_dir, exist_ok=True)

    # Save the sorted data in a file named 'filename.json' inside the 'sorted/filename' folder
    sorted_file_path = os.path.join(sorted_dir, f'{base_file_name}.json')  # Path for the sorted file

    # Write the sorted data to the new file
    with open(sorted_file_path, 'w') as sorted_file:
        json.dump(new_data, sorted_file, indent=4)

    print(f"Sorted data saved to {sorted_file_path}")
    
def main(target_dir):
    # Define the regex pattern to match the required file name
    pattern = re.compile(r'(.*)\.json')

    # Iterate over the files in the target directory
    for file_name in os.listdir(target_dir):
        # Check if the file name matches the required pattern
        match = pattern.match(file_name)
        if match:
            # Construct the full path to the JSON file
            file_path = os.path.join(target_dir, file_name)

            # Process the file
            print(f"Processing file: {file_path}")
            try:
                process_file(file_path)
            except:
                print(f"Error processing file: {file_path}")


if __name__ == "__main__":
    # Example usage: replace 'your_target_directory' with the actual target directory
    target_dir = '/home/zzo/Quickstart/alpaca_eval/my_results/mistral-7b-sft-alpha_greedy'
    main(target_dir)
