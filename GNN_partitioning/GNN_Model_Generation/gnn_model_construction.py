import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import time
import multiprocessing
from sklearn.model_selection import train_test_split
from pm4py import play_out
import pandas as pd
# from pm4py.objects.log.importer.xes.variants.iterparse import Parameters as Import_Parameter
from pm4py.objects.log.importer.xes.variants.iterparse import Parameters as Export_Parameter
from pm4py.objects.log.importer.xes.importer import apply
import sys
from pm4py.objects.process_tree.importer import importer as tree_importer

desired_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(desired_path)

from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_score_for_cut_type
import random
# Import libraries
from matplotlib import pyplot as plt
import warnings
from tqdm import tqdm

import gnn_models

relative_path = "GNN_partitioning/GNN_Data"
random_seed = 1996
show_gradient = False
use_symmetric = True

def analyse_dataframe_result(df, data_settings = None, detailed = False, file_path = ""):
    accuracy = df["Accuracy"].mean()
    full_accuracy_number = 0
    for index, row in df.iterrows():
        if row['Right_Prediction'] == row['Nodes']:
            full_accuracy_number += 1
    full_accuracy = full_accuracy_number / len(df)
    print("Accuracy: " + str(accuracy))
    print("Full Accuracy: " + str(full_accuracy))
    
    if detailed:
        df['distance_ratio'] = df.apply(lambda row: (abs(row['Actual_Score'] - row['Predicted_Score'])) / max(abs(row['Actual_Score']),(abs(row['Actual_Score'] - row['Predicted_Score']))) if row['Actual_Score'] != 0 else 0, axis=1)

        df_grouped = df.groupby('Nodes')
        
        data_accuracy_list = []
        data_missClass_list = []
        data_distance_list = []
        data_Label_list = []
        # Iterate over the groups and store each group as a separate DataFrame
        for group_name, group_df in df_grouped:
            data_accuracy_list.append(group_df["Accuracy"])
            data_missClass_list.append(group_df["Nodes"] - group_df["Right_Prediction"])
            data_distance_list.append(group_df['distance_ratio'])
            data_Label_list.append(group_name)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
        
        # Add a global title
        fig.suptitle('Cut type: ' + data_settings["Cut_type"] + " | Dataset size: " + str(len(df)) + " | Model: " + str(data_settings["model_number"]) + " | Epochs: " + str(data_settings["num_epochs"]) + " | Batch size: " + str(data_settings["batch_size"]))
        
        axes[0].boxplot(data_accuracy_list)
        axes[0].set_title('Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xticklabels(data_Label_list)
        
        axes[1].boxplot(data_missClass_list)
        axes[1].set_title('Number of misclassified nodes')
        axes[1].set_ylabel('# of misclassified nodes')
        axes[1].set_xticklabels(data_Label_list)
        
        axes[2].boxplot(data_distance_list)
        axes[2].set_title('Difference actual to predicted score')
        axes[2].set_ylabel('Percentage')
        axes[2].set_xticklabels(data_Label_list)

        fig.text(0.5, 0.005, 'Total accuracy: ' + str(round(accuracy,2)) + " | " + "Full accuracy: " + str(round(full_accuracy,2)), ha='center')
        
        # axes[1].plot(x2, y2)
        plt.tight_layout()
        # show plot
        # plt.show()
        
        fig_name = "/GNN_results_" + data_settings["Cut_type"] + "_dataset_" + str(len(df)) + "_model_" + str(data_settings["model_number"]) + "_epochs_" + str(data_settings["num_epochs"]) + "_batch_" + str(data_settings["batch_size"])
        fig.savefig(file_path + fig_name + ".pdf")
    

def get_ignore_mask(data):
    # 0
    ignoreValue = 0
    
    maskList = []
    for label in data["Labels"]:
        if label in data["PartitionA"] or label in data["PartitionB"]:
            maskList.append(1)
        else:
            maskList.append(ignoreValue)
            
    # ignore padded nodes
    if len(maskList) < data["Number_nodes"]:
        maskList = maskList + [ignoreValue for _ in range(data["Number_nodes"] - len(maskList))]
        
    # converting the mask to tensor
    mask = torch.tensor(maskList, dtype=torch.float32)
    return mask

def get_log(file_name):
  warnings.filterwarnings("ignore")
  # Export the event log to a XES file
  parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  log = apply(file_name + ".xes", parameters=parameter)
  return log

def get_score_value_from_partition(A, B, cut_type, dataSet_numbers, sup, ratio, datapiece, random_seed_P, random_seed_M):
    typeName = "Sup_"  + str(sup) + "_Ratio_" + str(ratio)
    filePath = relative_path + "/" + cut_type + "/Data_" + str(dataSet_numbers) + "/" + typeName
    path_tree_P = filePath + "/treeP_" + str(dataSet_numbers) + "_" + typeName + "_Data_" + str(datapiece)
    path_tree_M = filePath + "/treeM_" + str(dataSet_numbers) + "_" + typeName + "_Data_" + str(datapiece)
    
    treeP = tree_importer.apply(path_tree_P)
    treeM = tree_importer.apply(path_tree_M)
    
    
    random.seed(random_seed_P)
    logP = play_out(treeP)
    random.seed(random_seed_M)
    logM = play_out(treeM)

    score = get_score_for_cut_type(logP, logM, A, B, cut_type, sup, ratio)
    return score
    

def partition_names(predictions, names, mask):
    # Convert the tensors to numpy arrays for easier manipulation
    predictions = np.array(predictions)
    names = np.array(names)
    mask = np.array(mask)
    
    names_partitionA = []
    names_partitionB = []
    
    for i in range(len(mask)):
        if mask[i] == 1:
            if predictions[i] == 0:
                names_partitionA.append(names[i])
            else:
                names_partitionB.append(names[i])

    return names_partitionA, names_partitionB
       
def combine_dataframes(current_df, results):
    # Combine the list of DataFrames into a single DataFrame
    combined_df = pd.concat([current_df, results])
    return combined_df
    
def evaluate_model_helper(model_number, model_params, data, model_args, detailed = False):
    model = gnn_models.generate_model_args(model_args)
    model.load_state_dict(model_params)
    
    # 2. Set the Model to Evaluation Mode
    model.eval()
    
    df_res = pd.DataFrame(columns=["Nodes", "Right_Prediction", "Wrong_Prediction", "Accuracy"])
    
    # 3. Forward Pass
    with torch.no_grad():
        predictions = gnn_models.get_model_outcome(model_number, model, data)
        
        # Apply softmax to obtain probability distributions over classes
        probs = F.sigmoid(predictions)
        
        binary_predictions = torch.round(probs)
        
        # Assuming you have a tensor `mask` that indicates which nodes to ignore
        mask = get_ignore_mask(data)
        mask = torch.unsqueeze(mask, dim=1)
        
        # Convert the mask to a boolean mask
        boolean_mask = (mask == 1)
        
        target_tensor = torch.tensor(data["Truth"])
        
        target_tensor_transposed = torch.unsqueeze(target_tensor, dim=1)
        
        # Apply the mask to the ground truth labels and logits tensors 
        masked_ground_truth_labels = target_tensor_transposed[boolean_mask]
        masked_logits = binary_predictions[boolean_mask]
        
        # Apply the mask to the ground truth labels and logits tensors
        masked_ground_truth_labels = masked_ground_truth_labels.bool()
        masked_predicted_labels = masked_logits.bool()
        
        right_prediction = torch.sum(masked_ground_truth_labels == masked_predicted_labels).item()
        wrong_prediction = torch.sum(masked_ground_truth_labels != masked_predicted_labels).item()
        accuracy = (masked_ground_truth_labels == masked_predicted_labels).float().mean().item()
        if use_symmetric:
            right_prediction_inverse = torch.sum(~masked_ground_truth_labels == masked_predicted_labels).item()
            right_prediction = max(right_prediction, right_prediction_inverse)
            wrong_prediction_inverse = torch.sum(~masked_ground_truth_labels != masked_predicted_labels).item()
            wrong_prediction = min(wrong_prediction, wrong_prediction_inverse)
            accuracy_inverse = (~masked_ground_truth_labels == masked_predicted_labels).float().mean().item()
            accuracy = max(accuracy, accuracy_inverse)
        
        partitionA, partitionB = partition_names(binary_predictions, data["Labels"], mask)

        if detailed:
            
            actual_score = get_score_value_from_partition(set(data["PartitionA"]), set(data["PartitionB"]), data["Cut_type"], len(data["PartitionA"]) + len(data["PartitionB"]), data["Support"], data["Ratio"], data["Dataitem"], data["random_seed_P"], data["random_seed_M"])
            predicted_score = get_score_value_from_partition(set(partitionA), set(partitionB), data["Cut_type"], len(data["PartitionA"]) + len(data["PartitionB"]), data["Support"], data["Ratio"], data["Dataitem"], data["random_seed_P"], data["random_seed_M"])
            
            
        else:
            actual_score = None
            predicted_score = None

        df_res = pd.concat([df_res, pd.DataFrame.from_records([{
            "Nodes" : len(masked_predicted_labels),
            "Right_Prediction": right_prediction,
            "Wrong_Prediction": wrong_prediction,
            "Accuracy" : accuracy,
            "Predicted_Score" : predicted_score,
            "Actual_Score" : actual_score
        }])])
    return df_res
    
def evaluate_model_helper_star(args):
    return evaluate_model_helper(*args)    
   
def evaluate_model(model_number, model_params, test_dict, model_args, detailed = False):

    combined_df = pd.DataFrame(columns=["Nodes", "Right_Prediction", "Wrong_Prediction", "Accuracy"])

    all_train_items = [value for values in test_dict.values() for value in values]
    
    list_data_pool = [(model_number, model_params, data, model_args, detailed) for data in all_train_items]
    
    if detailed:
        num_processors_available = multiprocessing.cpu_count()
        print("Number of available processors:", num_processors_available)
        if num_processors_available > 20:
            num_processors = max(1,round(num_processors_available))
        else:
            num_processors = max(1,round(num_processors_available/2))
        print("Number of used processors:", num_processors)
        
        pool_res = None
        with multiprocessing.Pool(num_processors) as pool:
            pool_res = tqdm(pool.imap(evaluate_model_helper_star, list_data_pool),total=len(list_data_pool))
            
            for result in pool_res:
                # Process individual evaluation result
                combined_df = pd.concat([combined_df, result])
    else:
        for data in all_train_items:
            df_res = evaluate_model_helper(model_number, model_params, data, model_args, detailed)
            combined_df = combine_dataframes(combined_df, df_res)
    
            

    return combined_df

def train_model(model_number, model, num_epochs, batch_size, training_dic, model_args, data_settings):
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Create a data loader with the specified batch size
    # data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # for batch_data in data_loader:

    all_train_items = [value for values in training_dic.values() for value in values]
    
    # Compute the total number of batches
    num_batches = len(all_train_items) // batch_size
    
    # Set model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        # Shuffle the training data for each epoch
        random.shuffle(all_train_items)
        
        for batch in range(num_batches):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Extract the current batch
            batch_data = all_train_items[batch * batch_size : (batch + 1) * batch_size]
            
            logits_list = []
            mask_list = []
            batch_labels = []
            for data in batch_data:
                logits = gnn_models.get_model_outcome(model_number, model, data)
                logits_list.append(logits)
            
            

            
            for data in batch_data:
                # Assuming you have a tensor `mask` that indicates which nodes to ignore
                mask = get_ignore_mask(data)
                mask = torch.unsqueeze(mask, dim=1)
                mask_list.append(mask)
                
                target_tensor = torch.tensor(data["Truth"],dtype=torch.float32)
                target_tensor_transposed = torch.unsqueeze(target_tensor, dim=1)
                batch_labels.append(target_tensor_transposed)
                
            
            
            batch_labels = torch.cat(batch_labels, dim=0)
            mask_list = torch.cat(mask_list, dim=0)
            logits_list = torch.cat(logits_list, dim=0)
            
            # Apply the mask to the ground truth labels and logits tensors
            masked_ground_truth_labels = batch_labels * mask_list
            masked_logits = logits_list * mask_list

            # Compute the loss
            loss = criterion(masked_logits, masked_ground_truth_labels)
            
            # for par and exc, solutions are symmetric, so we adjust the loss
            if use_symmetric:
                if data_settings["Cut_type"] == "par" or data_settings["Cut_type"] == "exc":
                    inverse_loss = criterion(masked_logits, 1 - masked_ground_truth_labels)
                    # Take the minimum loss between both tasks
                    loss = torch.min(loss, inverse_loss)

            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
        # Adjust the learning rate
        # scheduler.step()

        # Print the loss for monitoring
        print()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")
        df_res = evaluate_model(model_number, model.state_dict(), training_dic, model_args)
        analyse_dataframe_result(df_res)
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None and show_gradient:
                print(f"Gradients for {name}:")
                print(param.grad)



    print("Training finished.")
    
def read_data_from_path(file_path):
    data = {}
    if os.path.exists(file_path):
        # Open the text file in read mode
        with open(file_path, 'r') as file:
            # Iterate over each line in the file
            matrix_P_arrayList = []
            matrix_M_arrayList = []
            state = -1
            for line in file:
                if state == -1:
                    state += 1
                    continue
                elif state == 0 :
                    data["Labels"] = line.split(" ")[:-1]
                    state += 1
                    continue
                elif state == 1:
                    data["Activity_count_P"] = np.array(line.split(" ")[:-1]).astype(int)
                    state += 1
                    continue
                elif state == 2:
                    data["Activity_count_M"] = np.array(line.split(" ")[:-1]).astype(int)
                    state += 1
                    continue
                elif state == 3 and line == "\n":
                    data["Adjacency_matrix_P"] = np.vstack(matrix_P_arrayList)
                    state += 1
                    continue
                elif state == 4 and line == "\n":
                    data["Adjacency_matrix_M"] = np.vstack(matrix_M_arrayList)
                    state += 1
                    continue   
                elif state == 5:
                    data["Cut_type"] = line[:-1]
                    state += 1
                    continue 
                elif state == 6:
                    data["Support"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 7:
                    data["Ratio"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 8:
                    data["Size_par"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 9:
                    data["Dataitem"] = line[:-1]
                    state += 1
                    continue 
                elif state == 10:
                    data["PartitionA"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 11:
                    data["PartitionB"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 12:
                    data["Score"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 13:
                    data["random_seed_P"] = int(line[:-1])
                    state += 1
                    continue 
                elif state == 14:
                    data["random_seed_M"] = int(line[:-1])
                    state += 1
                    continue 
                
                if state == 3:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_P_arrayList.append(np_array)
                if state == 4:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_M_arrayList.append(np_array)
    return data

def generate_ground_truth(data):
    partitionA = data["PartitionA"]
    partitionB = data["PartitionB"]
    labels = data["Labels"]
    truthList = []
    for label in labels:
        if label in partitionA:
            truthList.append(0)
        elif label in partitionB:
            truthList.append(1)
        else:
            truthList.append(-1)
            
    return truthList

def setup_dataSet(data_dic, max_node_size_in_dataset):
    for dataList in data_dic.values():
        for data in dataList:
            data["Truth"] = generate_ground_truth(data)
            # Calculate the difference in size
            diff_size = max_node_size_in_dataset - data["Adjacency_matrix_P"].shape[0]
            # Pad the adjacency matrix with zeros
            data["Adjacency_matrix_P"] = np.pad(data["Adjacency_matrix_P"], ((0, diff_size), (0, diff_size)), mode='constant')
            data["Adjacency_matrix_M"] = np.pad(data["Adjacency_matrix_M"], ((0, diff_size), (0, diff_size)), mode='constant')
            
            data["Activity_count_P"] = np.pad(data["Activity_count_P"], (0, diff_size), mode='constant')
            data["Activity_count_M"] = np.pad(data["Activity_count_M"], (0, diff_size), mode='constant')
            
            data["Truth"] = data["Truth"] + [-1 for _ in range(max_node_size_in_dataset - len(data["Truth"]))]
            data["Number_nodes"] = max_node_size_in_dataset
        
            
    return data_dic

def get_data_length_from_dic(data_dic):
    length = 0
    for key in data_dic.keys():
        length += len(data_dic[key])
    return length

def read_all_data_for_cut_Type(file_path, cut_type, max_dataSet_numbers):
    data_dic = dict()
    max_node_size_in_dataset = 0
    currentPath = file_path
    pathFiles = []
    
    if os.path.exists(currentPath):
        currentPath += "/" + cut_type
        if os.path.exists(currentPath):
            for root, _ , files in os.walk(currentPath):
                for file in files:
                    if file.endswith(".txt"):  # Filter for text files
                        pathFiles.append(os.path.join(root, file))


    # we sort the files in reverse, so we start with high node graphs
    pathFiles = sorted(pathFiles, reverse=True)
    # random.shuffle(pathFiles)
    for pathFile in pathFiles:
        if get_data_length_from_dic(data_dic) > max_dataSet_numbers:
            break
        data = read_data_from_path(pathFile)
        max_node_size_in_dataset = max(max_node_size_in_dataset, len(data["Labels"]))
        nodeSize = len(data["PartitionA"]) + len(data["PartitionB"])
        if nodeSize in data_dic:
            data_dic[nodeSize].append(data)
        else:
            data_dic[nodeSize] = [data]

                        
    data_dic = setup_dataSet(data_dic,max_node_size_in_dataset)
    return data_dic, max_node_size_in_dataset
               
def save_model_parameter(file_name, data_settings, model_args):
    with open(file_name, 'w') as file:
        file.write('# Data Settings\n')
        for key, value in data_settings.items():
            file.write(key + ': ' + str(value) + '\n')
        file.write('\n')
        file.write('# Model Settings\n')
        for key, value in model_args.items():
            file.write(key + ': ' + str(value) + '\n')
            
                    
def generate_Models(file_path_models, save_results = False, file_path_results = ""):
    time_start = time.time()

    hidden_dim = 32
    # best models:
    # 8 - conv
    # 9 - 3 dense with adj and weight
    # 10 - k dense with adj and weight
    # 11 - 3 dense with adj and weight and node frequency
    # 12 - conv with node degree as node feature
    model_number = 13
    # cut_types = ["par", "exc","loop", "seq"]
    cut_types = ["seq","loop"]
    num_epochs = 30
    batch_size = 10
    max_dataSet_numbers = 100000
    
    for cut_type in cut_types:
        print("Cut type: " + cut_type)
        print("Reading Data")
        data_dic, max_node_size_in_dataset = read_all_data_for_cut_Type(relative_path, cut_type, max_dataSet_numbers)

        data_settings = {"Cut_type" : cut_type,
                        "model_number" : model_number,
                        "num_epochs" : num_epochs,
                        "batch_size" : batch_size}


        # Example usage
        output_dim = 1  # Number of output classes
        global_features = 3
        node_features = 2
        print("Generating Model: " + str(model_number))
        
        # model_number, input_dim, hidden_dim, output_dim, node_features, global_features
        model_args = {"model_number" : model_number,
                        "input_dim" : max_node_size_in_dataset,
                        "hidden_dim" : hidden_dim,
                        "output_dim" : output_dim,
                        "node_features" : node_features,
                        "global_features" : global_features}
        
        model = gnn_models.generate_model_args(model_args)
        
        train_dict = {}
        test_dict = {}
        for key, value in data_dic.items():
            # Split the data and labels into training and test sets
            train_data, test_data = train_test_split(value, test_size=0.2, random_state=1996)
            train_dict[key] = train_data
            test_dict[key] = test_data

        print("Train data size: " + str(get_data_length_from_dic(train_dict)))
        print("Test data size: " + str(get_data_length_from_dic(test_dict)))

        print()
        print("INITIAL STATISTIC")
        df_res = evaluate_model(model_number,model.state_dict(), test_dict, model_args)
        analyse_dataframe_result(df_res)

        print()
        print("Training Model")
        train_model(model_number,model, num_epochs, batch_size, train_dict ,model_args, data_settings)

        print()
        print("FINAL STATISTIC")
        print("Evaluating Model")
        df_res = evaluate_model(model_number,model.state_dict(), test_dict, model_args, detailed=save_results)
        analyse_dataframe_result(df_res, data_settings, detailed=save_results, file_path=file_path_results)
        torch.save(model.state_dict(), file_path_models + '/gnn_model_' + cut_type + ".pt")
        save_model_parameter(file_path_models + '/gnn_model_' + cut_type + ".txt", data_settings, model_args)
    
    
    time_end = time.time()
    print("Runtime of the program is " + str(round(time_end - time_start,2)) + " seconds")





if __name__ == '__main__':
    random.seed(random_seed)
    generate_Models("GNN_partitioning/GNN_Model", True, "GNN_partitioning/GNN_Accuracy_Results")





