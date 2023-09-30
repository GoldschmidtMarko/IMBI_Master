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

import sys
root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)
from GNN_partitioning.GNN_Model_Generation import gnn_models

# GCN with global variables weight matrix, and 2 graph convolution layers
def generate_Model_V13(input_dim, node_features, global_features, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, node_features, global_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear = nn.Linear(node_features + global_features, out_features)

        def forward(self, weight_matrix_1, features_P, global_variables):
            # First graph convolution
            output_conv_P = torch.matmul(weight_matrix_1, features_P)
            
            # output = output_conv_P + output_conv_M
            output = torch.cat((output_conv_P, global_variables), dim=1)
            
            # n x (2*node_features + global variables)
            output = self.linear(output)
            
            return output
    
    class SelfAttentionGCNLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(SelfAttentionGCNLayer, self).__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.W = nn.Parameter(torch.Tensor(in_features, out_features))
            self.b = nn.Parameter(torch.Tensor(out_features))

            nn.init.xavier_uniform_(self.W)
            nn.init.zeros_(self.b)

        def forward(self, node_features, weighted_adjacency_matrix):
            """
            node_features: Node feature matrix of shape (num_nodes, in_features)
            weighted_adjacency_matrix: Weighted adjacency matrix of shape (num_nodes, num_nodes)
            """
            # Compute self-attention scores
            attention_scores = torch.matmul(node_features, self.W) + self.b

            # Compute attention weights using softmax
            attention_weights = nn.functional.softmax(attention_scores, dim=1)

            # Apply attention weights to input node features
            updated_features = torch.matmul(weighted_adjacency_matrix, attention_weights)

            return updated_features
    
    class GCNClassifier(nn.Module):
        def __init__(self, input_dim,  node_Features, global_features, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(node_Features, global_features, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim, global_features, hidden_dim)
            self.attention = SelfAttentionGCNLayer(hidden_dim, hidden_dim)
            self.gcEnd = GraphConvolutionLayer(hidden_dim, global_features, output_dim)
            self.global_attention = nn.Linear(hidden_dim, 1)
            self.final_linear = nn.Linear(hidden_dim + input_dim, output_dim)


        def forward(self, weight_matrix_1, features_P, global_variables):
            hidden = F.relu(self.gc1(weight_matrix_1, features_P, global_variables))
            # splitIndex = int(hidden.shape[1]/2)
            # hidden_left = hidden[:, :splitIndex]
            # hidden_right = hidden[:, splitIndex:]
            hidden = F.relu(self.gc2(weight_matrix_1, hidden, global_variables))
            
            # Global attention mechanism
            # hidden = (N x hidden_dim)
            # global_attention_weights = (N x 1)
            
            global_attention_weights = torch.softmax(self.global_attention(hidden), dim=0)
            global_embedding = torch.sum(global_attention_weights * hidden, dim=1)
            # (N x 1)

            global_embedding = global_embedding.unsqueeze(0).repeat(hidden.shape[0], 1)  # Repeat for all nodes

            combined = torch.cat([hidden, global_embedding], dim=1)
            
            output = self.final_linear(combined)
            return output
    
    # Create the GNN classifier model  
    model = GCNClassifier(input_dim, node_features, global_features, hidden_dim, output_dim)
    
    return model


def generate_model(model_args):
    model_number = int(model_args["model_number"])
    input_dim = int(model_args["input_dim"])
    hidden_dim = int(model_args["hidden_dim"])
    output_dim = int(model_args["output_dim"])
    node_features = int(model_args["node_features"])
    global_features = int(model_args["global_features"])
    
    if model_number < 13:
        print("Unstable models are not supported anymore")
        return None
    if model_number == 13:
        return generate_Model_V13(input_dim, node_features, global_features, hidden_dim, output_dim)
    
def generate_model_from_args(model_args):
    return generate_model(model_args)

def transform_data_to_model(data):
    # converting the data to torch
    torch_matrix_P = torch.from_numpy(data["Adjacency_matrix_P"]).to(torch.float32)
    
    feature_node_frequencies_P = torch.from_numpy(data["Activity_count_P"]).unsqueeze(dim=1).to(torch.float32)
      
    # normalization, removed since it could have negative effects
    if torch_matrix_P.max() != 0:
        torch_matrix_P = torch_matrix_P / torch_matrix_P.max()

    if feature_node_frequencies_P.max() != 0: 
        feature_node_frequencies_P = feature_node_frequencies_P / feature_node_frequencies_P.max()
        
    # Create the feature matrix
    global_feature = torch.tensor([[data["Support"], data["Ratio"], data["Size_par"]] for _ in range(data["Adjacency_matrix_P"].shape[0])])
    
    return torch_matrix_P, global_feature, feature_node_frequencies_P

def get_node_degree(torch_matrix_P):
    # Calculate the degree for each node
    node_out_degree_P = torch.sum(torch_matrix_P > 0, dim=1, dtype=torch.float32, keepdim=True)
    node_in_degree_P = torch.sum(torch_matrix_P > 0, dim=0, dtype=torch.float32, keepdim=True)
    node_in_degree_P_T = torch.transpose(node_in_degree_P, 0, 1)
    node_degree_P = torch.cat((node_out_degree_P, node_in_degree_P_T), dim=1)
    
    return node_degree_P

# diff
def get_model_outcome(model_number, model, data):
    torch_matrix_P, global_feature, feature_node_frequencies_P = transform_data_to_model(data)
    number_relevant_nodes = sum(1 for element in data["Activity_count_P"] if element != 0)
    
    if model_number < 13:
        print("Unstable models are not supported anymore")
        return None
    
    elif model_number == 13:
        node_degree_P = get_node_degree(torch_matrix_P)
        return model(torch_matrix_P, node_degree_P, global_feature)
        

def generate_data_from_log(logP, sup, ratio, size_par):
    unique_node_P, adj_matrix_P = gnn_models.generate_adjacency_matrix_from_log(logP)

    logP_art = gnn_models.artificial_start_end(logP.__deepcopy__())
    
    activity_count_P = gnn_models.get_activity_count(logP_art)
    unique_activity_count_P = gnn_models.get_activity_count_list_from_unique_list(activity_count_P, unique_node_P)
    
    size_par = 1
    
    data = {"Adjacency_matrix_P": adj_matrix_P,
            "Support": sup,
            "Ratio": ratio,
            "Size_par": size_par,
            "Labels": unique_node_P,
            "Activity_count_P": np.array(unique_activity_count_P).astype(int),
            "Number_nodes": adj_matrix_P.shape[0],
            "PartitionA": [],
            "PartitionB": []
            }
    
    return data

def pad_data(data, max_node_size_in_dataset):
    diff_size = max_node_size_in_dataset - data["Adjacency_matrix_P"].shape[0]
    # Pad the adjacency matrix with zeros
    data["Adjacency_matrix_P"] = np.pad(data["Adjacency_matrix_P"], ((0, diff_size), (0, diff_size)), mode='constant')

    data["Activity_count_P"] = np.pad(data["Activity_count_P"], (0, diff_size), mode='constant')

    data["Number_nodes"] = max_node_size_in_dataset
    return data

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
        data_settings, model_args = gnn_models.read_model_parameter(model_setting_path)
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
            if gnn_models.check_substring_after_last_slash(model_path, cut_type):
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

    possible_local_partitions = gnn_models.get_local_partitions(possible_partitions, percentage_of_nodes)
    possible_local_partitions_cleaned = gnn_models.clean_error_partitions(possible_local_partitions)
    possible_local_partitions_filtered = gnn_models.filter_impossible_partitions(possible_local_partitions_cleaned, data["Labels"], data["Adjacency_matrix_P"])
    
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




