import os
import sys

# Define the directory path where you want to perform the renaming
root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)
relative_path = root_path + "/GNN_partitioning_single/GNN_Data"

# Recursively iterate over all files and subdirectories
for root, dirs, files in os.walk(relative_path):
    for filename in files:
        # Check if the file doesn't end with '.txt'
        if not filename.endswith('.txt') and not filename.endswith('.json'):
            # Generate the new filename by replacing the extension with '.json'
            new_filename = filename + '.json'
            
            # Construct the full paths for the old and new filenames
            old_filepath = os.path.join(root, filename)
            new_filepath = os.path.join(root, new_filename)
            
            try:
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f'Renamed: {old_filepath} -> {new_filepath}')
            except Exception as e:
                print(f'Error renaming {old_filepath}: {str(e)}')