import os
from tqdm import tqdm
import concurrent.futures

import os
import sys

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

def delete_file(file_path):
    """
    Deletes a file.

    :param file_path: The path to the file to delete.
    """
    try:
        os.remove(file_path)
        return file_path
    except Exception as e:
        return None

def delete_files_in_directory(directory_path, file_extensions=None, max_workers=4):
    """
    Recursively deletes files in a directory and its subdirectories using multi-threading.

    :param directory_path: The path to the directory to start deleting files from.
    :param file_extensions: A list of file extensions to restrict the deletion to.
                            If None, all files will be deleted.
    :param max_workers: The maximum number of worker threads to use.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Get a list of all files in the directory and its subdirectories
    files_to_delete = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_extensions is None or file_path.endswith(tuple(file_extensions)):
                files_to_delete.append(file_path)

    # Delete files using multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(delete_file, files_to_delete), total=len(files_to_delete), desc="Deleting files"))


if __name__ == "__main__":
    
    directory_to_clean = os.path.join(root_path,os.path.join("GNN_partitioning","GNN_Data"))  # Replace with the directory you want to clean
    file_extensions_to_delete = None  # Replace with the file extensions you want to delete (or None for all files)
    print("Deleting: " + directory_to_clean)
    delete_files_in_directory(directory_to_clean, file_extensions_to_delete)
