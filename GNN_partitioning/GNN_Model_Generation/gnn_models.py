import torch
import torch.nn as nn
import numpy as np
from local_pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import torch.nn.functional as F
import os
import math
from pm4py.util import xes_constants
from itertools import combinations
from pm4py.objects.log import obj as log_instance


def artificial_start_end(log):
    st = 'start'
    en = 'end'
    activity_key = xes_constants.DEFAULT_NAME_KEY
    start_event = log_instance.Event()
    start_event[activity_key] = st
    
    end_event = log_instance.Event()
    end_event[activity_key] = en

    for trace in log:
        trace.insert(0, start_event)
        trace.append(end_event)
    return log

def generate_adjacency_matrix_from_log(log):
  log_art = artificial_start_end(log.__deepcopy__())
  dfg = dfg_discovery.apply(log_art, variant=dfg_discovery.Variants.FREQUENCY)
  
  
  unique_nodes = ["start", "end"]
  unique_nodes = unique_nodes + list(set([node for edge in dfg.keys() for node in edge]) - {"start", "end"})
  num_nodes = len(unique_nodes)
  
  # Initialize an empty adjacency matrix
  adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

  # Populate the adjacency matrix
  for edge, count in dfg.items():
      source_node, target_node = edge
      source_index = unique_nodes.index(source_node)
      target_index = unique_nodes.index(target_node)
      adj_matrix[source_index, target_index] = count
  
  # print(unique_nodes)
  # print(adj_matrix)
  return unique_nodes, adj_matrix

def add_epsilon_to_matrix(adjacency_matrix, number_relevant_nodes):
    def add_epsilon(adjacency_matrix, epsilon, k):
        adjacency_matrix[:k, :k] += epsilon
        return adjacency_matrix
    
    epsilon = 1e-8
    adjacency_matrix = add_epsilon(adjacency_matrix, epsilon, number_relevant_nodes)
    # def transform_matrix(matrix):
    #     return torch.where(matrix > 0, torch.ones_like(matrix), matrix)
    # transformed_matrix = transform_matrix(adjacency_matrix)
    # print(transformed_matrix)
    
    return adjacency_matrix

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

def get_model_outcome(model_number, model, data):
    torch_matrix_P, global_feature, feature_node_frequencies_P = transform_data_to_model(data)
    number_relevant_nodes = sum(1 for element in data["Activity_count_P"] if element != 0)
    
    if model_number < 13:
        print("Unstable models are not supported anymore")
        return None
    
    elif model_number == 13:
        node_degree_P = get_node_degree(torch_matrix_P)
        return model(torch_matrix_P, node_degree_P, global_feature)
        
def get_activity_count(log):
  # Count the occurrences of each activity
  activity_count = {}
  for trace in log:
      for event in trace:
          activity = event["concept:name"]
          if activity in activity_count:
              activity_count[activity] += 1
          else:
              activity_count[activity] = 1
  return activity_count

def get_activity_count_list_from_unique_list(activity_count, unique_node_list):
  res = []
  for node in unique_node_list:
    if node in activity_count:
      res.append(activity_count[node])
    else:
      res.append(0)
  return res


def read_model_parameter(file_name):
    data_settings = {}
    model_args = {}

    with open(file_name, 'r') as file:
        section = None
        for line in file:
            # Remove leading/trailing whitespace and newline characters
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Detect section changes based on '#' symbol
            if line.startswith('#'):
                if line == '# Data Settings':
                    section = 'data_settings'
                elif line == '# Model Settings':
                    section = 'model_args'
                else:
                    section = None
            elif section == 'data_settings':
                # Split the line into key and value
                key, value = line.split(': ')
                data_settings[key] = value
            elif section == 'model_args':
                # Split the line into key and value
                key, value = line.split(': ')
                model_args[key] = value

    return data_settings, model_args

def check_substring_after_last_slash(input_string, substring):
    # Find the last occurrence of '/'
    last_slash_index = input_string.rfind('\\')
    if last_slash_index == -1:
        last_slash_index = input_string.rfind('/')
    
    # Check if the substring is present after the last slash
    if last_slash_index != -1 and substring in input_string[last_slash_index + 1:]:
        return True
    else:
        return False

def get_local_partitions_from_partition(partition, percentage_of_nodes):
    number_nodes = len(partition[0]) + len(partition[1])
    number_exchanging_nodes = math.ceil(number_nodes * percentage_of_nodes)
    
    res_partitions = []
    list_A = list(partition[0])
    list_B = list(partition[1])
    
    for n in range(number_exchanging_nodes + 1):
        for switched_indices in combinations(range(len(list_A) + len(list_B)), n):
            new_list_A = list(list_A)
            new_list_B = list(list_B)
            
            for i in switched_indices:
                if i < len(list_A):
                    element = list_A[i]
                    new_list_A.remove(element)
                    new_list_B.append(element)
                else:
                    element = list_B[i - len(list_A)]
                    new_list_B.remove(element)
                    new_list_A.append(element)
            
            res_partitions.append((set(new_list_A), set(new_list_B), partition[2]))
    
    return res_partitions

def get_local_partitions(partitions, percentage_of_nodes):
    res_partitions = []
    for partition in partitions:
        local_partitions = get_local_partitions_from_partition(partition, percentage_of_nodes)
        for local_partition in local_partitions:
            res_partitions.append(local_partition)
        
    return res_partitions
    
def clean_error_partitions(partitions):
    res_partitions = []
    for partition in partitions:
        if len(partition[0]) != 0 and len(partition[1]) != 0:
            res_partitions.append(partition)
    return res_partitions


def is_possible_partition(partition, activities, weighted_adjacency_matrix):
    def dfs(node, visited, partition_index_A_set):
        visited[node] = True
        for neighbor in range(len(weighted_adjacency_matrix)):
            if weighted_adjacency_matrix[node][neighbor] != 0 or weighted_adjacency_matrix[neighbor][node] != 0:
                if neighbor in partition_index_A_set:
                    if not visited[neighbor]:
                        dfs(neighbor, visited, partition_index_A_set)

    def find_indices(names_list, target_names):
        indices = []
        for name in target_names:
            if name in names_list:
                indices.append(names_list.index(name))
            else:
                indices.append(None)  # Name not found in the list
        return indices
    

    # Create a dictionary to convert node labels to indices
    partition_A = partition[0].copy()
    partition_A.add("start")
    partition_A_index = find_indices(activities, partition_A)

    # Convert nodes to indices and initialize visited array
    visited = [False] * len(weighted_adjacency_matrix)

    # Start DFS from the first node in the set
    dfs(partition_A_index[0], visited, partition_A_index)

    # Check if all nodes in the set were visited
    return all(visited[index] for index in partition_A_index)

def is_possible_partition_accurate(partition, activities, weighted_adjacency_matrix):
    
    cut_type = partition[2]
    start_act_set = set()
    end_act_set = set()
    for index, value in enumerate(weighted_adjacency_matrix[activities.index("start")]):
        if value != 0:
            start_act_set.add(activities[index])
    for index, value in enumerate(weighted_adjacency_matrix[:,activities.index("end")]):
        if value != 0:
            end_act_set.add(activities[index])
            
    
    
    if cut_type == "seq":
        if len(set(partition[0]).intersection(start_act_set)) == 0:
            return False
        if len(set(partition[1]).intersection(end_act_set)) == 0:
            return False
        return True
    elif cut_type == "par" or cut_type == "exc":
        if len(set(partition[0]).intersection(start_act_set)) == 0:
            return False
        if len(set(partition[1]).intersection(start_act_set)) == 0:
            return False
        if len(set(partition[0]).intersection(end_act_set)) == 0:
            return False
        if len(set(partition[1]).intersection(end_act_set)) == 0:
            return False
        return True
    elif cut_type == "loop":
        if len(set(partition[0]).intersection(start_act_set)) == 0:
            return False
        if len(set(partition[0]).intersection(end_act_set)) == 0:
            return False
        return True
    return True
    

def filter_impossible_partitions(partitions, activities, adjacency_matrix):
    res_partitions = []
    for partition in partitions:
        if is_possible_partition(partition,activities, adjacency_matrix):
            res_partitions.append(partition)
            
    res_partitions_accurate = []
    for partition in res_partitions:
        if is_possible_partition_accurate(partition,activities, adjacency_matrix):
            res_partitions_accurate.append(partition)
        
    return res_partitions_accurate

def get_start_end_activites(activities, adjacency_matrix):
    start_activities = []
    end_activities = []
    
    num_columns = len(adjacency_matrix[0])
    start_activities_index = []

    for col in range(num_columns):
        if adjacency_matrix[0][col] != 0:
            start_activities_index.append(col)
            
    end_activities_index = []
    for row_idx, row in enumerate(adjacency_matrix):
        if row[1] != 0:
            end_activities_index.append(row_idx)
    
    for index in start_activities_index:
        start_activities.append(activities[index])
    for index in end_activities_index:
        end_activities.append(activities[index])
    return start_activities, end_activities
    
    
def generate_custom_cut_type_partitions(activities, adjacency_matrix_P):
    list_activities = set(activities) - {"start", "end"}
    cut_types = ["exc_tau"]
    res_partitions = []
    for cut_type in cut_types:
        res_partitions.append((list_activities, set(), {cut_type}))
        
    cut_types = ["loop_tau"]
    for cut_type in cut_types:
        start_activities, end_activities = get_start_end_activites(activities, adjacency_matrix_P)
        res_partitions.append((set(start_activities), set(end_activities), {cut_type}))
    
    return res_partitions


def get_partitions_from_gnn(root_file_path, gnn_file_path, logP, logM, sup, ratio, size_par, percentage_of_nodes = 0):
    model_setting_paths = []
    model_paths = []
    
    gnn_file_path = os.path.join(root_file_path, gnn_file_path)
    
    
    if os.path.exists(gnn_file_path):
        for root, _ , files in os.walk(gnn_file_path):
            for file in files:
                if file.endswith("_setting.txt"):  # Filter for text files
                    model_setting_paths.append(os.path.join(root, file))
                if file.endswith(".pt"):  # Filter for pt files
                    model_paths.append(os.path.join(root, file))
    
    if len(model_setting_paths) != 4 or len(model_paths) != 4:
        print("Error: not all models or text files are present. Found " + str(len(model_setting_paths)) + " text files and " + str(len(model_paths)) + " models, but expected 4 of each.")
        return None
    model_parameters = []
    for model_setting_path in model_setting_paths:
        data_settings, model_args = read_model_parameter(model_setting_path)
        model_parameters.append((data_settings, model_args))

    data = generate_data_from_log(logP, sup, ratio, size_par)
    

    possible_partitions = []
    for data_setting, model_args in model_parameters:
        cut_type = data_setting["Cut_type"]
        model_number = int(data_setting["model_number"])
        
        if data["Number_nodes"] > int(model_args["input_dim"]):
            print("Error: the number of nodes in the generated dataset is bigger then the number of nodes in the model. Dataset: " + str(data["Number_nodes"]) + ", model: " + str(model_args["input_dim"]))
            return None
        
        data = pad_data(data, int(model_args["input_dim"]))
        if data["Number_nodes"] != int(model_args["input_dim"]):
            print("Error: the number of nodes in the generated dataset is not equal to the number of nodes in the model. Dataset: " + str(data["Number_nodes"]) + ", model: " + str(model_args["input_dim"]))
            return None
        
        cur_model_path = ""
        for model_path in model_paths:
            if check_substring_after_last_slash(model_path, cut_type):
                cur_model_path = model_path
                break
            
        if cur_model_path == "":
            print("Error: no model found for cut type " + cut_type)
            return None
        
        cur_model = generate_model(model_args)
        cur_model.load_state_dict(torch.load(cur_model_path))
        
        # Freeze the model's parameters
        for param in cur_model.parameters():
            param.requires_grad = False
        
        binary_prediction = get_prediction_from_model(model_number, cur_model, data)
        
        if torch.any(torch.isnan(binary_prediction)):
            print("Error: the model returned a nan value.")
            return None
        
        binary_prediction_list = binary_prediction.view(-1).tolist()
        binary_prediction_list = [int(element) for element in binary_prediction_list]
        binary_mask = [1 if element != 0 else 0 for element in data["Activity_count_P"] ]
        binary_mask[0] = 0 # start
        binary_mask[1] = 0 # end
        
        for i, mask in enumerate(binary_mask):
            if mask == 0:
                binary_prediction_list[i] = -1
        
        partitionA = set()
        partitionB = set()
        
        for i, pred in enumerate(binary_prediction_list):
            if pred == 0:
                partitionA.add(data["Labels"][i])
            if pred == 1:
                partitionB.add(data["Labels"][i])
        possible_partitions.append((partitionA, partitionB, {cut_type}))

    possible_local_partitions = get_local_partitions(possible_partitions, percentage_of_nodes)
    possible_local_partitions_cleaned = clean_error_partitions(possible_local_partitions)
    possible_local_partitions_filtered = filter_impossible_partitions(possible_local_partitions_cleaned, data["Labels"], data["Adjacency_matrix_P"])
    
    if len(possible_local_partitions_filtered) == 0:
        print("Error: no possible partitions found. Filtered out all partitions.")
        return None
    
    # custom_cut_types = generate_custom_cut_type_partitions(data["Labels"],data["Adjacency_matrix_P"])
    custom_cut_types = []
    
    possible_local_partitions = possible_local_partitions_cleaned + custom_cut_types

    return possible_local_partitions    

def get_prediction_from_model(model_number, model, data):
    model_outcome = get_model_outcome(model_number, model, data)
    probs = F.sigmoid(model_outcome)
        
    binary_predictions = torch.round(probs)
    
    return binary_predictions




