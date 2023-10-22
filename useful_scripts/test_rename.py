
import os
import sys

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

root_path = os.path.join(root_path, "GNN_partitioning", "GNN_Model")
# List of file names
file_names = ["gnn_model_exc_data.txt", "gnn_model_seq_data.txt", "gnn_model_loop_data.txt", "gnn_model_par_data.txt"]

# Iterate through each file
for file_name in file_names:
    try:
        # Open the file for reading and create a temporary list to store modified lines
        with open(os.path.join(root_path,file_name), 'r') as file:
            lines = file.readlines()
            modified_lines = []

            # Replace occurrences of 'GNN_partitioning_single' with 'GNN_partitioning' in each line
            for line in lines:
                modified_line = line.replace('GNN_partitioning_single', 'GNN_partitioning')
                modified_lines.append(modified_line)

        # Open the same file for writing and overwrite it with the modified lines
        with open(file_name, 'w') as file:
            file.writelines(modified_lines)

        print(f"Modified {file_name} successfully.")
    except Exception as e:
        print(f"Error modifying {file_name}: {str(e)}")