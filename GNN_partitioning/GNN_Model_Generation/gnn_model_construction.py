import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import math
import time
import multiprocessing
from sklearn.model_selection import train_test_split
from pm4py import play_out
import pandas as pd
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.log.importer.xes.variants.iterparse import Parameters as Export_Parameter
from pm4py.objects.log.importer.xes.importer import apply
import sys
from pm4py.objects.process_tree.importer import importer as tree_importer
import datetime

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_score_for_cut_type
import random
# Import libraries
from matplotlib import pyplot as plt
import warnings
from tqdm import tqdm
from GNN_partitioning.GNN_Model_Generation.uni import gnn_models as uni_gnn_models
from GNN_partitioning.GNN_Model_Generation.bi import gnn_models as bi_gnn_models
import json

random_seed = 1996
show_gradient = False
use_symmetric = True

def analyse_dataframe_result(df, data_settings = None, detailed = False, file_path = ""):
    sum_Right_prediction = df['Right_Prediction'].sum()
    sum_wrong_prediction = df['Wrong_Prediction'].sum()
    sum_amount_full_accuracy = df['Amount_full_accuracy'].sum()
    sum_number_data = df['Number_Data_instances'].sum()
    
    # Calculate accuracy
    accuracy = sum_Right_prediction / (sum_Right_prediction + sum_wrong_prediction)
    
    # Calculate full accuracy
    full_accuracy = sum_amount_full_accuracy / sum_number_data
    
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

def show_dfg(log):
    from pm4py.statistics.end_activities.log import get as end_activities_get
    from pm4py.statistics.start_activities.log import get as start_activities_get
    from pm4py import view_dfg
    from pm4py.algo.discovery.dfg.variants import native as dfg_inst
    
    parameters = {}
    start_act_cur_dfg = start_activities_get.get_start_activities(log, parameters=parameters)
    end_act_cur_dfg = end_activities_get.get_end_activities(log, parameters=parameters)
    cur_dfg = dfg_inst.apply(log, parameters=parameters)
    view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)

def uni_get_score_value_from_partition(A, B, cut_type, dataSet_numbers, sup, ratio, datapiece, random_seed_P, data_path):
    typeName = "Sup_"  + str(sup)
    filePath = data_path + "/" + cut_type + "/Data_" + str(dataSet_numbers) + "/" + typeName
    path_tree_P = filePath + "/treeP_" + str(dataSet_numbers) + "_" + typeName + "_" + str(datapiece)
    
    treeP = load_tree(path_tree_P + ".json")
    
    random.seed(random_seed_P)
    logP = play_out(treeP)

    logM = logP.__deepcopy__()

    score = get_score_for_cut_type(logP, logM, A, B, cut_type, sup, ratio)
    return score
    
def bi_get_score_value_from_partition(A, B, cut_type, dataSet_numbers, sup, ratio, datapiece, random_seed_P, random_seed_M, data_path):
    typeName = "Sup_"  + str(sup) + "_Ratio_" + str(ratio)
    filePath = data_path + "/" + cut_type + "/Data_" + str(dataSet_numbers) + "/" + typeName
    path_tree_P = filePath + "/treeP_" + str(dataSet_numbers) + "_" + typeName + "_" + str(datapiece)
    path_tree_M = filePath + "/treeM_" + str(dataSet_numbers) + "_" + typeName + "_" + str(datapiece)
    
    treeP = load_tree(path_tree_P + ".json")
    treeM = load_tree(path_tree_M + ".json")
    
    
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
       
def get_Operator_from_string(operator_string):
    if operator_string == "->":
        return Operator.SEQUENCE
    elif operator_string == "X":
        return Operator.XOR
    elif operator_string == "+":
        return Operator.PARALLEL
    elif operator_string == "*":
        return Operator.LOOP
    return None
       
def load_tree(file_name):
    def deserialize_tree(serialized_node):
        if serialized_node is None:
            return None
        
        node = ProcessTree()
        node.label = serialized_node["label"]
        node.operator = get_Operator_from_string(serialized_node["operand"])
        node.children = [deserialize_tree(child) for child in serialized_node["children"]]
        return node

    with open(file_name, "r") as file:
        serialized_tree = json.load(file)

    return deserialize_tree(serialized_tree)
       
def combine_dataframes(current_df, results):
    # Combine the list of DataFrames into a single DataFrame
    combined_df = pd.concat([current_df, results])
    return combined_df  
    
def evaluate_model_helper(model_number, model_params, data_list, model_args, data_path, consider_ratio, detailed = False):
    df_res = pd.DataFrame(columns=["Nodes", "Right_Prediction", "Wrong_Prediction", "Accuracy"])
    if len(data_list) == 0:
        return df_res
    
    if consider_ratio:
        model = bi_gnn_models.generate_model_from_args(model_args)
    else:
        model = uni_gnn_models.generate_model_from_args(model_args)    
    
    model.load_state_dict(model_params)
    
    # 2. Set the Model to Evaluation Mode
    model.eval()
    
    # 3. Forward Pass
    with torch.no_grad():
        
        predictions_list = []
        mask_list = []
        batch_labels = []
        for data in data_list:
            if consider_ratio:
                predictions = bi_gnn_models.get_model_outcome(model_number, model, data)
            else:
                predictions = uni_gnn_models.get_model_outcome(model_number, model, data)  

            # Apply softmax to obtain probability distributions over classes
            probs = F.sigmoid(predictions)
            binary_predictions = torch.round(probs)
            predictions_list.append(binary_predictions)
            
        for data in data_list:
            # Assuming you have a tensor `mask` that indicates which nodes to ignore
            mask = get_ignore_mask(data)
            mask = torch.unsqueeze(mask, dim=1)
            boolean_mask = (mask == 1)
            mask_list.append(boolean_mask)
            
            target_tensor = torch.tensor(data["Truth"],dtype=torch.float32)
            target_tensor_transposed = torch.unsqueeze(target_tensor, dim=1)
            batch_labels.append(target_tensor_transposed)
        
        # print(batch_labels)
        batch_labels = torch.cat(batch_labels, dim=0)
        mask_list = torch.cat(mask_list, dim=0)
        predictions_list = torch.cat(predictions_list, dim=0)

        # Apply the mask to the ground truth labels and logits tensors 
        masked_ground_truth_labels = batch_labels * mask_list
        masked_logits = predictions_list * mask_list
        
        # Apply the mask to the ground truth labels and logits tensors
        masked_ground_truth_labels = masked_ground_truth_labels.bool()
        masked_predicted_labels = masked_logits.bool()
        
        res_data_dic = dict()
        # Iterate over data_list
        offset = 0
        for data in data_list:
            length = len(data["PartitionA"]) + len(data["PartitionB"])
            ground_truth_nodes = masked_ground_truth_labels[offset:offset + length]
            predicted_nodes = masked_predicted_labels[offset:offset + length]
            amount_full_accuracy = 0
            right_predictions = 0
            wrong_predictions = 0
            
            right_predictions = torch.sum(ground_truth_nodes == predicted_nodes).item()
            wrong_predictions = torch.sum(ground_truth_nodes != predicted_nodes).item()
            if use_symmetric:
                cut_type = data["Cut_type"]
                if cut_type == "par" or cut_type == "exc":
                    right_prediction_inverse = torch.sum(~ground_truth_nodes == predicted_nodes).item()
                    right_predictions = max(right_predictions, right_prediction_inverse)
                    wrong_prediction_inverse = torch.sum(~ground_truth_nodes != predicted_nodes).item()
                    wrong_predictions = min(wrong_predictions, wrong_prediction_inverse)
                    
                    if torch.all(ground_truth_nodes == predicted_nodes) or torch.all(~ground_truth_nodes == predicted_nodes):
                        amount_full_accuracy = 1
                else:
                    # Check if all nodes in the current segment are equal
                    if torch.all(ground_truth_nodes == predicted_nodes):
                        amount_full_accuracy = 1
            else:
                # Check if all nodes in the current segment are equal
                if torch.all(ground_truth_nodes == predicted_nodes):
                    amount_full_accuracy = 1
                    
            if detailed == False:
                if length not in res_data_dic:
                    res_data_dic[length] = {
                        "right_prediction" : right_predictions,
                        "wrong_prediction" : wrong_predictions,
                        "amount_full_accuracy" : amount_full_accuracy,
                        "number_instances" : 1,
                        "predicted_score" : 0,
                        "actual_score" : 0
                    }
                else:
                    res_data_dic[length]["right_prediction"] += right_predictions
                    res_data_dic[length]["wrong_prediction"] += wrong_predictions
                    res_data_dic[length]["amount_full_accuracy"] += amount_full_accuracy
                    res_data_dic[length]["number_instances"] += 1
            else:
                add_data = True
                if consider_ratio:
                    actual_score = bi_get_score_value_from_partition(set(data["PartitionA"]), set(data["PartitionB"]), data["Cut_type"], len(data["PartitionA"]) + len(data["PartitionB"]), data["Support"], data["Ratio"], data["Dataitem"], data["random_seed_P"], data["random_seed_M"], data_path)
                else:
                    actual_score = uni_get_score_value_from_partition(set(data["PartitionA"]), set(data["PartitionB"]), data["Cut_type"], len(data["PartitionA"]) + len(data["PartitionB"]), data["Support"], data["Ratio"], data["Dataitem"], data["random_seed_P"], data_path)
            
                mask = get_ignore_mask(data)
                if consider_ratio:
                    predictions = bi_gnn_models.get_model_outcome(model_number, model, data)
                else:
                    predictions = uni_gnn_models.get_model_outcome(model_number, model, data)  

                # Apply softmax to obtain probability distributions over classes
                probs = F.sigmoid(predictions)
                binary_predictions = torch.round(probs)
                

                partitionA, partitionB = partition_names(binary_predictions, data["Labels"], mask)
                
                if consider_ratio:
                    predicted_score = bi_get_score_value_from_partition(set(partitionA), set(partitionB), data["Cut_type"], len(data["PartitionA"]) + len(data["PartitionB"]), data["Support"], data["Ratio"], data["Dataitem"], data["random_seed_P"], data["random_seed_M"], data_path)
                else:
                    predicted_score = uni_get_score_value_from_partition(set(partitionA), set(partitionB), data["Cut_type"], len(data["PartitionA"]) + len(data["PartitionB"]), data["Support"], data["Ratio"], data["Dataitem"], data["random_seed_P"], data_path)

                if length not in res_data_dic:
                    res_data_dic[length] = [{
                        "right_prediction" : right_predictions,
                        "wrong_prediction" : wrong_predictions,
                        "amount_full_accuracy" : amount_full_accuracy,
                        "number_instances" : 1,
                        "predicted_score" : predicted_score,
                        "actual_score" : actual_score
                    }]
                else:
                    res_data_dic[length].append({
                        "right_prediction" : right_predictions,
                        "wrong_prediction" : wrong_predictions,
                        "amount_full_accuracy" : amount_full_accuracy,
                        "number_instances" : 1,
                        "predicted_score" : predicted_score,
                        "actual_score" : actual_score
                    })
                
            # Update the offset for the next iteration
            offset += length

        df_res = None
        for key, value in res_data_dic.items():
            if detailed == False:
                df_res = pd.concat([df_res, pd.DataFrame.from_records([{
                    "Nodes" : key,
                    "Number_Data_instances" : value["number_instances"],
                    "Right_Prediction": value["right_prediction"],
                    "Wrong_Prediction": value["wrong_prediction"],
                    "Amount_full_accuracy" : value["amount_full_accuracy"],
                    "Predicted_Score" : value["predicted_score"],
                    "Actual_Score" : value["actual_score"]
                }])])
            else:
                for i, data in enumerate(value):
                    df_res = pd.concat([df_res, pd.DataFrame.from_records([{
                        "Nodes" : key,
                        "Number_Data_instances" : data["number_instances"],
                        "Right_Prediction": data["right_prediction"],
                        "Wrong_Prediction": data["wrong_prediction"],
                        "Amount_full_accuracy" : data["amount_full_accuracy"],
                        "Predicted_Score" : data["predicted_score"],
                        "Actual_Score" : data["actual_score"],
                        "Accuracy" : data["right_prediction"] / (data["right_prediction"] + data["wrong_prediction"])
                    }])])
    return df_res
    
def evaluate_model_helper_star(args):
    return evaluate_model_helper(*args)    
   
def evaluate_model(model_number, model_params, test_dict, model_args, data_path, consider_ratio, parallel = True, detailed = False):
    time_cur = time.time()
    combined_df = pd.DataFrame(columns=["Nodes", "Right_Prediction", "Wrong_Prediction"])

    all_train_items = [value for values in test_dict.values() for value in values]
    
    # print("Model data setup time: " + str(time.time() - time_cur))
    
    if parallel:
        num_processors_available = multiprocessing.cpu_count()
        if num_processors_available > 20:
            num_processors = max(1,round(num_processors_available))
        else:
            num_processors = max(1,round(num_processors_available/2))
            
        print("Running parallel evaluation with " + str(num_processors) + "/" + str(num_processors_available) + " processors")
        
        batch_size = math.ceil(len(all_train_items) / num_processors)
        list_data_pool = []
        offset = 0
        for i in range(num_processors):
            batch_data = all_train_items[offset:offset + batch_size]
            batch = (model_number, model_params, batch_data, model_args, data_path, consider_ratio, detailed)
            list_data_pool.append(batch)
            offset += batch_size

        pool_res = None
        with multiprocessing.Pool(num_processors) as pool:
            # pool_res = tqdm(pool.imap(evaluate_model_helper_star, list_data_pool),total=len(list_data_pool))
            pool_res = pool.imap(evaluate_model_helper_star, list_data_pool)
            
            for result in pool_res:
                # Process individual evaluation result
                combined_df = pd.concat([combined_df, result])
    else:
        for data in all_train_items:
            df_res = evaluate_model_helper(model_number, model_params, [data], model_args, data_path, consider_ratio, detailed)
            combined_df = combine_dataframes(combined_df, df_res)
    
            

    return combined_df

def generate_criterion(use_symmetric, cut_type):
    if use_symmetric == True:
        if cut_type == "par" or cut_type == "exc":
            print("Using SymmetricContrastiveLoss")
            class SymmetricContrastiveLoss(nn.Module):
                def __init__(self, margin=1.0):
                    super(SymmetricContrastiveLoss, self).__init__()
                    self.margin = margin

                def forward(self, embeddings, labels):
                    # Compute pairwise cosine similarity
                    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

                    labels = labels.squeeze()  # Remove the extra dimension

                    # Create a mask to identify positive and negative pairs
                    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
                    neg_mask = ~pos_mask

                    # Extract the similarity scores for positive and negative pairs
                    pos_scores = similarity_matrix[pos_mask]
                    neg_scores = similarity_matrix[neg_mask]

                    # Compute the contrastive loss
                    loss = 0.5 * (F.relu(self.margin - pos_scores).mean() + F.relu(neg_scores - self.margin).mean())

                    return loss

            return SymmetricContrastiveLoss()
    
    print("Using BCEWithLogitsLoss")
    return nn.BCEWithLogitsLoss()

def train_model(model_number, model, num_epochs, batch_size, training_dic, model_args, data_settings, data_path, consider_ratio):
    # Define the loss function
    criterion = generate_criterion(use_symmetric, data_settings["Cut_type"])

    # Define the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    all_train_items = [value for values in training_dic.values() for value in values]
    
    # Compute the total number of batches
    num_batches = len(all_train_items) // batch_size

    # Set model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        time_start = time.time()
        # Shuffle the training data for each epoch
        random.shuffle(all_train_items)
        loss = None
        for batch in range(num_batches):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Extract the current batch
            batch_data = all_train_items[batch * batch_size : (batch + 1) * batch_size]
            
            logits_list = []
            mask_list = []
            batch_labels = []
            for data in batch_data:
                if consider_ratio:
                    logits = bi_gnn_models.get_model_outcome(model_number, model, data)
                else:
                    logits = uni_gnn_models.get_model_outcome(model_number, model, data) 
                logits_list.append(logits)
            
            for data in batch_data:
                # Assuming you have a tensor `mask` that indicates which nodes to ignore
                mask = get_ignore_mask(data)
                mask = torch.unsqueeze(mask, dim=1)
                mask_list.append(mask)
                
                target_tensor = torch.tensor(data["Truth"],dtype=torch.float32)
                target_tensor_transposed = torch.unsqueeze(target_tensor, dim=1)
                batch_labels.append(target_tensor_transposed)
                
            batch_losses = []
            for logits, labels, mask in zip(logits_list, batch_labels, mask_list):
                logits = logits * mask
                labels = labels * mask
                # Compute the loss for each data point in the batch
                loss_data = criterion(logits, labels)

                # Accumulate the losses
                batch_losses.append(loss_data)  # Assuming you want to store the total loss as a scalar

            # Backward pass
            # Create a scalar tensor with gradients enabled
            loss = torch.mean(torch.stack(batch_losses))

            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
        # Adjust the learning rate
        # scheduler.step()
        time_diff = time.time() - time_start
        time_start = time.time()
        # Print the loss for monitoring
        print()
        if loss is not None:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()} | Time: {time_diff}")
        df_res = evaluate_model(model_number, model.state_dict(), training_dic, model_args, data_path, consider_ratio=consider_ratio)
        analyse_dataframe_result(df_res)
        time_diff = time.time() - time_start
        print("Time for evaluation: " + str(time_diff))
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None and show_gradient:
                print(f"Gradients for {name}:")
                print(param.grad)



    print("Training finished.")

def bi_read_data_from_path(file_path):
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
    data["Path"] = file_path
    return data

def uni_read_data_from_path(file_path):
    data = {}
    if os.path.exists(file_path):
        # Open the text file in read mode
        with open(file_path, 'r') as file:
            # Iterate over each line in the file
            matrix_P_arrayList = []
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
                elif state == 2 and line == "\n":
                    data["Adjacency_matrix_P"] = np.vstack(matrix_P_arrayList)
                    state += 1
                    continue
                elif state == 3:
                    data["Cut_type"] = line[:-1]
                    state += 1
                    continue 
                elif state == 4:
                    data["Support"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 5:
                    data["Ratio"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 6:
                    data["Size_par"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 7:
                    data["Dataitem"] = line[:-1]
                    state += 1
                    continue 
                elif state == 8:
                    data["PartitionA"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 9:
                    data["PartitionB"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 10:
                    data["Score"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 11:
                    data["random_seed_P"] = int(line[:-1])
                    state += 1
                    continue 
                elif state == 12:
                    data["tree"] = str(line[:-1])
                    state += 1
                    continue 
                
                if state == 2:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_P_arrayList.append(np_array)
                    
    data["Path"] = file_path
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


def setup_data(data, max_node_size_in_dataset, consider_ratio):
    data["Truth"] = generate_ground_truth(data)
    # Calculate the difference in size
    diff_size = max_node_size_in_dataset - data["Adjacency_matrix_P"].shape[0]
    # Pad the adjacency matrix with zeros
    data["Adjacency_matrix_P"] = np.pad(data["Adjacency_matrix_P"], ((0, diff_size), (0, diff_size)), mode='constant')
    data["Activity_count_P"] = np.pad(data["Activity_count_P"], (0, diff_size), mode='constant')
    
    if consider_ratio:
        data["Adjacency_matrix_M"] = np.pad(data["Adjacency_matrix_M"], ((0, diff_size), (0, diff_size)), mode='constant')
        data["Activity_count_M"] = np.pad(data["Activity_count_M"], (0, diff_size), mode='constant')
    
    data["Truth"] = data["Truth"] + [-1 for _ in range(max_node_size_in_dataset - len(data["Truth"]))]
    data["Number_nodes"] = max_node_size_in_dataset
    
    return data
    
def setup_dataSet(data_dic, max_node_size_in_dataset, consider_ratio):
    for dataList in data_dic.values():
        for data in dataList:
            data = setup_data(data, max_node_size_in_dataset, consider_ratio)
        
    return data_dic

def get_data_length_from_dic(data_dic):
    length = 0
    for key in data_dic.keys():
        length += len(data_dic[key])
    return length

def does_tree_exist(data, data_path, consider_ratio):
    support = data["Support"]
    dataSet_numbers = len(data["PartitionA"]) + len(data["PartitionB"])
    datapiece = data["Dataitem"]
    cut_type = data["Cut_type"]
    ratio = data["Ratio"]
    
    if consider_ratio:
        typeName = "Sup_"  + str(support) + "_Ratio_" + str(ratio)
    else:
        typeName = "Sup_"  + str(support)
    filePath = data_path + "/" + cut_type + "/Data_" + str(dataSet_numbers) + "/" + typeName
    path_tree_P = filePath + "/treeP_" + str(dataSet_numbers) + "_" + typeName + "_" + str(datapiece) + ".json"
    
    if os.path.exists(path_tree_P):
        return True
    else:
        return False
    
def does_path_files_has_ratio(pathFiles):
    for pathFile in pathFiles:
        if "Ratio_" in pathFile:
            separator = os.path.sep
            splitted = pathFile.split("Ratio_")
            number = splitted[1].split(separator)[0]
            if float(number) != 0:
                return True
        else:
            return False
    return False

def read_all_data_for_cut_Type(file_path, cut_type, max_dataSet_numbers):
    data_dic = dict()
    max_node_size_in_dataset = 0
    currentPath = file_path
    pathFiles = []
    consider_ratio = False
    
    if os.path.exists(currentPath):
        currentPath += "/" + cut_type
        if os.path.exists(currentPath):
            for root, _ , files in os.walk(currentPath):
                for file in files:
                    if file.endswith(".txt"):  # Filter for text files
                        pathFiles.append(os.path.join(root, file))

    consider_ratio = does_path_files_has_ratio(pathFiles)

    # we sort the files in reverse, so we start with high node graphs
    pathFiles = sorted(pathFiles, reverse=True)
    # random.shuffle(pathFiles)
    for pathFile in pathFiles:
        if get_data_length_from_dic(data_dic) > max_dataSet_numbers:
            break
        if consider_ratio:
            data = bi_read_data_from_path(pathFile)
        else:
            data = uni_read_data_from_path(pathFile)
        if does_tree_exist(data, file_path, consider_ratio):
            max_node_size_in_dataset = max(max_node_size_in_dataset, len(data["Labels"]))
            nodeSize = len(data["PartitionA"]) + len(data["PartitionB"])
            if nodeSize in data_dic:
                data_dic[nodeSize].append(data)
            else:
                data_dic[nodeSize] = [data]
                                  
    data_dic = setup_dataSet(data_dic,max_node_size_in_dataset, consider_ratio)
    return data_dic, max_node_size_in_dataset, consider_ratio
               
def save_model_parameter(file_name, data_settings, model_args):
    with open(file_name, 'w') as file:
        file.write('# Data Settings\n')
        for key, value in data_settings.items():
            file.write(key + ': ' + str(value) + '\n')
        file.write('\n')
        file.write('# Model Settings\n')
        for key, value in model_args.items():
            file.write(key + ': ' + str(value) + '\n')
            
def save_used_data(file_name, train_data, test_data):
    with open(file_name, 'w') as file:
        file.write('# Test Data\n')
        for key, value in test_data.items():
            for item in value:
                file.write(str(key) + ': ' + str(item["Path"]) + '\n')
        file.write('\n')
        file.write('# Train Data\n')
        for key, value in train_data.items():
            for item in value:
                file.write(str(key) + ': ' + str(item["Path"]) + '\n')
            
                    
def generate_Models(file_path_models, save_results = False, file_path_results = "", relative_path_data = ""):
    
    print("Running model generation")
    print("Relative path for models: " + file_path_models)
    print("Relative path for results: " + file_path_results)
    print("Relative path for data: " + relative_path_data)
    print("Current time: " + str(datetime.datetime.now()))
    
    time_start = time.time()
    hidden_dim = 32
    # best models:
    # 8 - conv
    # 9 - 3 dense with adj and weight
    # 10 - k dense with adj and weight
    # 11 - 3 dense with adj and weight and node frequency
    # 12 - conv with node degree as node feature
    model_number = 13
    cut_types = ["loop", "seq", "par", "exc"]
    # cut_types = ["seq", "loop"]
    num_epochs = 3
    batch_size = 10
    max_dataSet_numbers = 100000
    
    for cut_type in cut_types:
        print("Cut type: " + cut_type)
        print("Reading Data")
        data_dic, max_node_size_in_dataset, consider_ratio = read_all_data_for_cut_Type(relative_path_data, cut_type, max_dataSet_numbers)
        print("Mode: consider_ratio: " + str(consider_ratio))
        
        if len(data_dic) == 0:
            print("No data found for cut type: " + cut_type)
            continue

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
        
        
        if consider_ratio:
            model = bi_gnn_models.generate_model_from_args(model_args)
        else:
            model = uni_gnn_models.generate_model_from_args(model_args)
        
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
        df_res = evaluate_model(model_number,model.state_dict(), test_dict, model_args, relative_path_data, consider_ratio=consider_ratio)
        analyse_dataframe_result(df_res)

        print()
        print("Training Model")
        train_model(model_number,model, num_epochs, batch_size, train_dict ,model_args, data_settings, relative_path_data, consider_ratio=consider_ratio)

        print()
        print("FINAL STATISTIC")
        print("Evaluating Model")
        df_res = evaluate_model(model_number,model.state_dict(), test_dict, model_args,relative_path_data, consider_ratio=consider_ratio, detailed=save_results)
        analyse_dataframe_result(df_res, data_settings, detailed=save_results, file_path=file_path_results)
        if not os.path.exists(file_path_models):
            os.makedirs(file_path_models)
            
        torch.save(model.state_dict(), file_path_models + '/gnn_model_' + cut_type + ".pt")
        save_model_parameter(file_path_models + '/gnn_model_' + cut_type + "_setting.txt", data_settings, model_args)
        save_used_data(file_path_models + '/gnn_model_' + cut_type + "_data.txt", train_dict, test_dict)
    
    
    time_end = time.time()
    print("Runtime of the program is " + str(round(time_end - time_start,2)) + " seconds")

if __name__ == '__main__':
    random.seed(random_seed)

    relative_path_model = root_path + "/GNN_partitioning/GNN_Model"
    relative_path_results = root_path + "/GNN_partitioning/GNN_Accuracy_results"
    relative_path_data = root_path + "/GNN_partitioning/GNN_Data"
    
    generate_Models(relative_path_model, True, relative_path_results, relative_path_data)





