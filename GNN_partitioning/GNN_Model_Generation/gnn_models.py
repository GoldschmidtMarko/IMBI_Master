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


# expands the matrix matrix_P and matrix_M so that the columns/ rows of both matrix have the the same activity
def generate_union_adjacency_matrices(matrix_P, nodeListP, matrix_M, nodeListM):
  # Find the common set of labels
  common_labels = ["start", "end"]
  common_labels = common_labels + list(set(nodeListP).union(set(nodeListM)) - {"start", "end"})
  
  # Create new expanded matrices
  expanded_matrix1 = np.zeros((len(common_labels), len(common_labels)), dtype=int)
  expanded_matrix2 = np.zeros((len(common_labels), len(common_labels)), dtype=int)

  # Copy values from original matrices to expanded matrices
  for i, label in enumerate(common_labels):
    if label in nodeListP:
        row_idx = nodeListP.index(label)
        expanded_matrix1[i, i] = matrix_P[row_idx, row_idx]  # Diagonal element
        for j, other_label in enumerate(common_labels[i+1:], start=i+1):
            if other_label in nodeListP:
                other_row_idx = nodeListP.index(other_label)
                expanded_matrix1[i, j] = matrix_P[row_idx, other_row_idx]
                expanded_matrix1[j, i] = matrix_P[other_row_idx, row_idx]
    if label in nodeListM:
        row_idx = nodeListM.index(label)
        expanded_matrix2[i, i] = matrix_M[row_idx, row_idx]  # Diagonal element
        for j, other_label in enumerate(common_labels[i+1:], start=i+1):
            if other_label in nodeListM:
                other_row_idx = nodeListM.index(other_label)
                expanded_matrix2[i, j] = matrix_M[row_idx, other_row_idx]
                expanded_matrix2[j, i] = matrix_M[other_row_idx, row_idx]

  return common_labels, expanded_matrix1, expanded_matrix2

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



def generate_Model_V1(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, adjacency_matrix_P, adjacency_matrix_M, features):
            
            # Multiplay adjacency_matrix_P with features (sup, ratio)
            output1  = torch.matmul(adjacency_matrix_P, features)
            
            # Multiplay adjacency_matrix_M with features (sup, ratio)
            output2  = torch.matmul(adjacency_matrix_M, features)
            
            # Aggregate information
            output = output1 + output2
            
            # Apply linear transformation
            output = self.linear(output)
            
            # Apply activation function
            # output = F.relu(output)
            
            return output
    
    class GNNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(2, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

        def forward(self, adjacency_matrix_P, adjacency_matrix_M, features):
            # Perform first graph convolution
            hidden1 = self.gc1(adjacency_matrix_P, adjacency_matrix_M, features)
            hidden1 = F.relu(hidden1)            
            
            # Perform second graph convolution
            hidden2 = self.gc2(adjacency_matrix_P, adjacency_matrix_M, hidden1)

            return hidden2

    # Create the GNN classifier model
    model = GNNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

def generate_Model_V2(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear_P = nn.Linear(in_features, out_features)
            self.linear_M = nn.Linear(in_features, out_features)

        # global_features dim 6x2
        def forward(self, adjacency_matrix_P, adjacency_matrix_M, node_feature, global_features):
            
            # Normalize adjacency matrix
            degree_matrix_P = torch.diag(torch.sum(adjacency_matrix_P, dim=1))
            normalized_adjacency_matrix_P = torch.inverse(degree_matrix_P) @ adjacency_matrix_P
            degree_matrix_M = torch.diag(torch.sum(adjacency_matrix_M, dim=1))
            normalized_adjacency_matrix_M = torch.inverse(degree_matrix_M) @ adjacency_matrix_M

            # Conv1:
            # 6x6 * 6x1 = 6x1
            # Conv2:
            # 6x6 * 6x32 = 6x32

            # Multiplay with node features
            normalized_adjacency_matrix_P = normalized_adjacency_matrix_P @ node_feature
            normalized_adjacency_matrix_M = normalized_adjacency_matrix_M @ node_feature

            # Conv1:
            # 6x1 + 6x2 = 6x3
            # Conv2:
            # 6x32 + 6x2 = 6x34

            # Concatenate global features with node features
            output1 = torch.cat((normalized_adjacency_matrix_P, global_features), dim=1)
            output2 = torch.cat((normalized_adjacency_matrix_M, global_features), dim=1)
            
            # Conv1:
            # 6x3 -> 6x32
            # Conv2:
            # 6x34 -> 6x1
            
            # Apply linear transformationgc1.linear.in_features
            output_1_linear = self.linear_P(output1)
            output_2_linear = self.linear_M(output2)
             
            output = output_1_linear + output_2_linear   
            
            return output
    
    class GNNClassifier(nn.Module):
        def __init__(self, input_features, hidden_dim, output_dim, number_global_features = 3):
            super(GNNClassifier, self).__init__()
            # + 2 for the global features sup and ratio
            self.gc1 = GraphConvolutionLayer(input_features + number_global_features, hidden_dim)
            # self.dense = nn.Linear(hidden_dim, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + number_global_features, output_dim)

        def forward(self, adjacency_matrix_P, adjacency_matrix_M, global_features):
            
            # Placeholder tensor when there are no node features
            features = torch.ones(adjacency_matrix_P.size(0), self.gc1.linear_P.in_features - 3)

            # Perform first graph convolution
            hidden1 = self.gc1(adjacency_matrix_P, adjacency_matrix_M, features, global_features)
            # hidden1 = self.dense(hidden1)
            hidden1 = F.relu(hidden1)  
            
            # Perform second graph convolution
            hidden2 = self.gc2(adjacency_matrix_P, adjacency_matrix_M, hidden1, global_features)

            return hidden2

    # Create the GNN classifier model
    model = GNNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

def generate_Model_V3(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear1 = nn.Linear(in_features, out_features)
            self.linear2 = nn.Linear(in_features, out_features)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features, global_variables):
            # First graph convolution
            output_1 = torch.matmul(adjacency_matrix_1, features)
            output_1 = torch.cat((output_1, global_variables), dim=1)
            output_1 = self.linear1(output_1)
            # output_1 = F.relu(output_1)
            
            # Second graph convolution
            output_2 = torch.matmul(adjacency_matrix_2, features)
            output_2 = torch.cat((output_2, global_variables), dim=1)
            output_2 = self.linear2(output_2)
            # output_2 = F.relu(output_2)

            return output_1 + output_2

    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(64 + 3, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 3, output_dim)
            

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, global_variables):
            features = torch.zeros(adjacency_matrix_1.shape[1], 64)
            hidden1 = self.gc1(adjacency_matrix_1, adjacency_matrix_2, features, global_variables)
            hidden1 = F.relu(hidden1)            
            hidden2 = self.gc2(adjacency_matrix_1, adjacency_matrix_2, hidden1, global_variables)

            return hidden2

    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

def generate_Model_V4(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear1 = nn.Linear(in_features, out_features)
            self.linear2 = nn.Linear(in_features, out_features)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features, global_variables):
            # First graph convolution
            output_1 = torch.matmul(adjacency_matrix_1, features)
            output_1 = torch.cat((output_1, global_variables), dim=1)
            output_1 = self.linear1(output_1)
            # output_1 = F.relu(output_1)
            
            # Second graph convolution
            output_2 = torch.matmul(adjacency_matrix_2, features)
            output_2 = torch.cat((output_2, global_variables), dim=1)
            output_2 = self.linear2(output_2)
            # output_2 = F.relu(output_2)

            return output_1 + output_2

    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(input_dim + 3, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 3, output_dim)
            

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, global_variables):
            features = torch.eye(adjacency_matrix_1.shape[1])
            hidden1 = self.gc1(adjacency_matrix_1, adjacency_matrix_2, features, global_variables)
            hidden1 = F.relu(hidden1)            
            hidden2 = self.gc2(adjacency_matrix_1, adjacency_matrix_2, hidden1, global_variables)

            return hidden2

    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# unstable
def generate_Model_V5(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, adjacency_matrix, edge_weights, features):
            
            # Normalize edge weights
            normalized_weights = self.normalize_weights(adjacency_matrix, edge_weights)
            
            # Perform graph convolution
            aggregated_features = torch.sparse.mm(adjacency_matrix, features.mul(normalized_weights))
            output = self.linear(aggregated_features)

            return output
        
        def normalize_weights(self, adjacency_matrix, edge_weights):
            adjacency_matrix_dense = adjacency_matrix.to_dense()
            row_sum = torch.sum(adjacency_matrix_dense, dim=1)
            normalized_weights = torch.zeros_like(edge_weights)
            mask = row_sum != 0
            normalized_weights[mask] = edge_weights[mask] / row_sum[mask].unsqueeze(1)
            return normalized_weights

    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)
            

        def forward(self, adjacency_matrix_1, edge_weights):
            features = torch.eye(adjacency_matrix_1.shape[1])
            hidden1 = self.gc1(adjacency_matrix_1, edge_weights, features)
            hidden1 = F.relu(hidden1)            
            hidden2 = self.gc2(adjacency_matrix_1, edge_weights, hidden1)

            return hidden2

    # Create the GNN classifier model
    model = GCNClassifier(input_dim, input_dim, output_dim)
    
    return model

def generate_Model_V6(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear1 = nn.Linear(in_features, out_features)
            self.linear2 = nn.Linear(in_features, out_features)
            self.linear = nn.Linear(2 * in_features, out_features)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            # First graph convolution
            output_1 = torch.matmul(adjacency_matrix_1, features_P)
            output_1 = torch.cat((output_1, global_variables), dim=1)
            
            # Second graph convolution
            output_2 = torch.matmul(adjacency_matrix_2, features_M)
            output_2 = torch.cat((output_2, global_variables), dim=1)
            
            output = torch.cat((output_1, output_2), dim=1)
            output = self.linear(output)
            
            return output

    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(input_dim + 3, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 3, output_dim)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            hidden1 = self.gc1(adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables)
            hidden1 = F.relu(hidden1)         
            output = self.gc2(adjacency_matrix_1, adjacency_matrix_2, hidden1, hidden1, global_variables)
            return output

    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

def generate_Model_V7(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear = nn.Linear(2 * in_features, out_features)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            # First graph convolution
            output_1 = torch.matmul(adjacency_matrix_1, features_P)
            output_1 = torch.cat((output_1, global_variables), dim=1)
            
            # Second graph convolution
            output_2 = torch.matmul(adjacency_matrix_2, features_M)
            output_2 = torch.cat((output_2, global_variables), dim=1)
            
            output = torch.cat((output_1, output_2), dim=1)
            output = self.linear(output)
            
            
            return output
    
    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(input_dim + 3, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 3, hidden_dim)
            self.gc3 = GraphConvolutionLayer(hidden_dim + 3, hidden_dim)
            self.gc4 = GraphConvolutionLayer(hidden_dim + 3, hidden_dim)
            self.gc5 = GraphConvolutionLayer(hidden_dim + 3, hidden_dim)
            self.gc6 = GraphConvolutionLayer(hidden_dim + 3, output_dim)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            hidden1 = self.gc1(adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables)
            hidden1 = F.relu(hidden1)    
            
            list_gc = [self.gc2, self.gc3, self.gc4, self.gc5] 
            for gc in list_gc:
                hidden1 = gc(adjacency_matrix_1, adjacency_matrix_2, hidden1, hidden1, global_variables)
                hidden1 = F.relu(hidden1)      
                
            output = self.gc6(adjacency_matrix_1, adjacency_matrix_2, hidden1, hidden1, global_variables)
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# GCN with global variables weight matrix, and 2 graph convolution layers
def generate_Model_V8(input_dim, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear_Node_Convolution_P = nn.Linear(in_features, out_features)
            self.linear_Node_Convolution_M = nn.Linear(in_features, out_features)
            self.linear = nn.Linear(2 * (out_features + 3), out_features)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            # First graph convolution
            node_conv_P = self.linear_Node_Convolution_P(features_P)  # (N x out_features)
            node_conv_M = self.linear_Node_Convolution_M(features_M)  # (N x out_features)
            
            output_1 = torch.matmul(adjacency_matrix_1, node_conv_P)
            output_1 = torch.cat((output_1, global_variables), dim=1)
            
            # Second graph convolution
            output_2 = torch.matmul(adjacency_matrix_2, node_conv_M)
            output_2 = torch.cat((output_2, global_variables), dim=1)
            
            output = torch.cat((output_1, output_2), dim=1)
            output = self.linear(output)
            
            return output
    
    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            hidden1 = self.gc1(adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables)
            hidden1 = F.relu(hidden1)  

            output = self.gc2(adjacency_matrix_1, adjacency_matrix_2, hidden1, hidden1, global_variables)
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# full connected 3 dense layer
def generate_Model_V9(input_dim, hidden_dim, output_dim):
    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.dense1 = nn.Linear(input_dim*4 + 3, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, hidden_dim)
            self.dense3 = nn.Linear(hidden_dim, hidden_dim)
            self.output_layer  = nn.Linear(hidden_dim, output_dim)

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            output = torch.cat((adjacency_matrix_1, adjacency_matrix_2), dim=1)
            output = torch.cat((output, features_P), dim=1)
            output = torch.cat((output, features_M), dim=1)
            output = torch.cat((output, global_variables), dim=1)
            
            hidden1 = F.relu(self.dense1(output))
            hidden2 = F.relu(self.dense2(hidden1))
            hidden3 = F.relu(self.dense3(hidden2))
            # hidden4 = F.relu(self.dense3(hidden3))
            output = self.output_layer(hidden3)
            
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# full connected 7 dense layer
def generate_Model_V10(input_dim, hidden_dim, output_dim):
    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim*4 + 3, hidden_dim))
            for i in range(1, 8):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        def forward(self, adjacency_matrix_1, adjacency_matrix_2, features_P, features_M, global_variables):
            output = torch.cat((adjacency_matrix_1, adjacency_matrix_2), dim=1)
            output = torch.cat((output, features_P), dim=1)
            output = torch.cat((output, features_M), dim=1)
            output = torch.cat((output, global_variables), dim=1)
            
            for i, layer in enumerate(self.layers):
                if i < len(self.layers) - 1:
                    output = layer(output)
                    output = F.relu(output)
                else:
                    output = layer(output)  
            
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# full connected 3 dense layer with global feature size_par and graph node frequencies
def generate_Model_V11(input_dim, hidden_dim, output_dim):
    class GCNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.dense1 = nn.Linear(input_dim*4 + 2 + 3, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, hidden_dim)
            self.dense3 = nn.Linear(hidden_dim, hidden_dim)
            self.output_layer  = nn.Linear(hidden_dim, output_dim)

        def forward(self,adj_mat_P, adj_mat_M, weighted_adjacency_matrix_1, weighted_adjacency_matrix_2, features_P, features_M, global_variables):
            output = torch.cat((adj_mat_P, adj_mat_M), dim=1)
            output = torch.cat((output, weighted_adjacency_matrix_1), dim=1)
            output = torch.cat((output, weighted_adjacency_matrix_2), dim=1)
            output = torch.cat((output, features_P), dim=1)
            output = torch.cat((output, features_M), dim=1)
            output = torch.cat((output, global_variables), dim=1)
            
            hidden1 = F.relu(self.dense1(output))
            hidden2 = F.relu(self.dense2(hidden1))
            hidden3 = F.relu(self.dense3(hidden2))
            # hidden4 = F.relu(self.dense3(hidden3))
            output = self.output_layer(hidden3)
            
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# GCN with global variables weight matrix, and 2 graph convolution layers
def generate_Model_V12(node_features, global_features, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, node_features, global_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear = nn.Linear(2 * node_features + global_features, out_features)

        def forward(self, weight_matrix_1, weight_matrix_2, features_P, features_M, global_variables):
            # First graph convolution
            output_conv_P = torch.matmul(weight_matrix_1, features_P)
            
            # Second graph convolution
            output_conv_M = torch.matmul(weight_matrix_2, features_M)

            output = torch.cat((output_conv_P, output_conv_M), dim=1)
            output = torch.cat((output, global_variables), dim=1)
            
            # n x (2*node_features + global variables)
            output = self.linear(output)
            
            return output
    
    class GCNClassifier(nn.Module):
        def __init__(self, node_features, global_features, hidden_dim, output_dim):
            super(GCNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(node_features, global_features, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim, global_features, output_dim)

        def forward(self, weight_matrix_1, weight_matrix_2, features_P, features_M, global_variables):
            hidden1 = self.gc1(weight_matrix_1, weight_matrix_2, features_P, features_M, global_variables)
            hidden1 = F.relu(hidden1)  

            output = self.gc2(weight_matrix_1, weight_matrix_2, hidden1, hidden1, global_variables)
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(node_features, global_features, hidden_dim, output_dim)
    
    return model

# GCN with global variables weight matrix, and 2 graph convolution layers
def generate_Model_V13(input_dim, node_features, global_features, hidden_dim, output_dim):
    class GraphConvolutionLayer(nn.Module):
        def __init__(self, node_features, global_features, out_features):
            super(GraphConvolutionLayer, self).__init__()
            self.linear = nn.Linear(2 * node_features + global_features, out_features)

        def forward(self, weight_matrix_1, weight_matrix_2, features_P, features_M, global_variables):
            # First graph convolution
            output_conv_P = torch.matmul(weight_matrix_1, features_P)
            
            # Second graph convolution
            output_conv_M = torch.matmul(weight_matrix_2, features_M)

            # output = output_conv_P + output_conv_M
            output = torch.cat((output_conv_P, output_conv_M), dim=1)
            output = torch.cat((output, global_variables), dim=1)
            
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


        def forward(self, weight_matrix_1, weight_matrix_2, features_P, features_M, global_variables):
            hidden = F.relu(self.gc1(weight_matrix_1, weight_matrix_2, features_P, features_M, global_variables))
            # splitIndex = int(hidden.shape[1]/2)
            # hidden_left = hidden[:, :splitIndex]
            # hidden_right = hidden[:, splitIndex:]
            hidden = F.relu(self.gc2(weight_matrix_1, weight_matrix_2, hidden, hidden, global_variables))
            
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
    
    if model_number == 1:
        return generate_Model_V1(input_dim, hidden_dim, output_dim)
    elif model_number == 2:
        return generate_Model_V2(input_dim, hidden_dim, output_dim)
    elif model_number == 3:
        return generate_Model_V3(input_dim, hidden_dim, output_dim)
    elif model_number == 4:
        return generate_Model_V4(input_dim, hidden_dim, output_dim)
    elif model_number == 5:
        return generate_Model_V5(input_dim, hidden_dim, output_dim)
    elif model_number == 6:
        return generate_Model_V6(input_dim, hidden_dim, output_dim)
    elif model_number == 7:
        return generate_Model_V7(input_dim, hidden_dim, output_dim)
    elif model_number == 8:
        return generate_Model_V8(input_dim, hidden_dim, output_dim)
    elif model_number == 9:
        return generate_Model_V9(input_dim, hidden_dim, output_dim)
    elif model_number == 10:
        return generate_Model_V10(input_dim, hidden_dim, output_dim)
    elif model_number == 11:
        return generate_Model_V11(input_dim, hidden_dim, output_dim)
    elif model_number == 12:
        return generate_Model_V12(node_features, global_features, hidden_dim, output_dim)
    elif model_number == 13:
        return generate_Model_V13(input_dim, node_features, global_features, hidden_dim, output_dim)
    
def generate_model_args(model_args):
    return generate_model(model_args)


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

def transform_data_to_model(data):
    # converting the data to torch
    torch_matrix_P = torch.from_numpy(data["Adjacency_matrix_P"]).to(torch.float32)
    torch_matrix_M = torch.from_numpy(data["Adjacency_matrix_M"]).to(torch.float32)
    
    feature_node_frequencies_P = torch.from_numpy(data["Activity_count_P"]).unsqueeze(dim=1).to(torch.float32)
    feature_node_frequencies_M = torch.from_numpy(data["Activity_count_M"]).unsqueeze(dim=1).to(torch.float32)
      
    # normalization, removed since it could have negative effects
    if torch_matrix_P.max() != 0:
        torch_matrix_P = torch_matrix_P / torch_matrix_P.max()
    if torch_matrix_M.max() != 0:
        torch_matrix_M = torch_matrix_M / torch_matrix_M.max()
    
    if feature_node_frequencies_P.max() != 0: 
        feature_node_frequencies_P = feature_node_frequencies_P / feature_node_frequencies_P.max()
        
    if feature_node_frequencies_M.max() != 0:
        feature_node_frequencies_M = feature_node_frequencies_M / feature_node_frequencies_M.max()
    
    # Create the feature matrix
    global_feature = torch.tensor([[data["Support"], data["Ratio"], data["Size_par"]] for _ in range(data["Adjacency_matrix_P"].shape[0])])
    
    return torch_matrix_P, torch_matrix_M, global_feature, feature_node_frequencies_P, feature_node_frequencies_M

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

def generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes):
    adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
    adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
    
    if False:
        adj_mat_M = add_epsilon_to_matrix(adj_mat_M, number_relevant_nodes)
        adj_mat_P = add_epsilon_to_matrix(adj_mat_P, number_relevant_nodes)
        torch_matrix_P = add_epsilon_to_matrix(torch_matrix_P, number_relevant_nodes)
        torch_matrix_M = add_epsilon_to_matrix(torch_matrix_M, number_relevant_nodes)
    return adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M

def get_node_degree(torch_matrix_P, torch_matrix_M):
    # Calculate the degree for each node
    node_out_degree_P = torch.sum(torch_matrix_P > 0, dim=1, dtype=torch.float32, keepdim=True)
    node_in_degree_P = torch.sum(torch_matrix_P > 0, dim=0, dtype=torch.float32, keepdim=True)
    node_in_degree_P_T = torch.transpose(node_in_degree_P, 0, 1)
    node_degree_P = torch.cat((node_out_degree_P, node_in_degree_P_T), dim=1)
    
    node_out_degree_M = torch.sum(torch_matrix_M > 0, dim=1, dtype=torch.float32, keepdim=True)
    node_in_degree_M = torch.sum(torch_matrix_M > 0, dim=0, dtype=torch.float32, keepdim=True)
    node_in_degree_M_T = torch.transpose(node_in_degree_M, 0, 1)
    node_degree_M = torch.cat((node_out_degree_M, node_in_degree_M_T), dim=1)
    

    return node_degree_P, node_degree_M

def get_model_outcome(model_number, model, data):
    torch_matrix_P, torch_matrix_M, global_feature, feature_node_frequencies_P, feature_node_frequencies_M = transform_data_to_model(data)
    number_relevant_nodes = sum(1 for element in data["Activity_count_P"] if element != 0)
    
    if model_number == 1:
        return model(torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 2:
        return model(torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 3:
        return model(torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 4:
        return model(torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 5:
        mapped_matrix = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        return model(mapped_matrix, torch_matrix_P)
    elif model_number == 6:
        adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M = generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes)
        
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 7:
        adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M = generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes)
        
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 8:
        adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M = generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes)
        
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 9:
        adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M = generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes)
        
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 10:
        adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M = generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes)
        
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, global_feature)
    elif model_number == 11:
        adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M = generate_adjacency_matrix(torch_matrix_P, torch_matrix_M, model_number, number_relevant_nodes)
        
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, feature_node_frequencies_P, feature_node_frequencies_M, global_feature)
    elif model_number == 12:
        node_degree_P, node_degree_M = get_node_degree(torch_matrix_P, torch_matrix_M)
        return model(torch_matrix_P, torch_matrix_M, node_degree_P, node_degree_M, global_feature)
    elif model_number == 13:
        node_degree_P, node_degree_M = get_node_degree(torch_matrix_P, torch_matrix_M)
        return model(torch_matrix_P, torch_matrix_M, node_degree_P, node_degree_M, global_feature)
        
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

def generate_data_from_log(logP, logM, sup, ratio, size_par):
    unique_node_P, adj_matrix_P = generate_adjacency_matrix_from_log(logP)
    unique_node_M, adj_matrix_M = generate_adjacency_matrix_from_log(logM)
    unique_nodeList, matrix_P, matrix_M = generate_union_adjacency_matrices(adj_matrix_P,unique_node_P,adj_matrix_M,unique_node_M)
    
    logP_art = artificial_start_end(logP.__deepcopy__())
    logM_art = artificial_start_end(logM.__deepcopy__())
    
    activity_count_P = get_activity_count(logP_art)
    activity_count_M = get_activity_count(logM_art)
    unique_activity_count_P = get_activity_count_list_from_unique_list(activity_count_P, unique_nodeList)
    unique_activity_count_M = get_activity_count_list_from_unique_list(activity_count_M, unique_nodeList)
    
    size_par = 1
    
    data = {"Adjacency_matrix_P": matrix_P,
            "Adjacency_matrix_M": matrix_M,
            "Support": sup,
            "Ratio": ratio,
            "Size_par": size_par,
            "Labels": unique_nodeList,
            "Activity_count_P": np.array(unique_activity_count_P).astype(int),
            "Activity_count_M": np.array(unique_activity_count_M).astype(int),
            "Number_nodes": matrix_P.shape[0],
            "PartitionA": [],
            "PartitionB": []
            }
    
    return data
    
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


def pad_data(data, max_node_size_in_dataset):
    diff_size = max_node_size_in_dataset - data["Adjacency_matrix_P"].shape[0]
    # Pad the adjacency matrix with zeros
    data["Adjacency_matrix_P"] = np.pad(data["Adjacency_matrix_P"], ((0, diff_size), (0, diff_size)), mode='constant')
    data["Adjacency_matrix_M"] = np.pad(data["Adjacency_matrix_M"], ((0, diff_size), (0, diff_size)), mode='constant')

    data["Activity_count_P"] = np.pad(data["Activity_count_P"], (0, diff_size), mode='constant')
    data["Activity_count_M"] = np.pad(data["Activity_count_M"], (0, diff_size), mode='constant')

    data["Number_nodes"] = max_node_size_in_dataset
    return data

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

def find_parent_folder(target_name, start_path=os.getcwd()):
    current_path = os.path.abspath(start_path)

    while current_path != "/":  # Stop at the root directory
        basename = os.path.basename(current_path)
        if basename == target_name:
            return current_path
        current_path = os.path.dirname(current_path)
    
    return None  # Target folder not found in the hierarchy

def get_partitions_from_gnn(root_file_path, gnn_file_path, logP, logM, sup, ratio, size_par, percentage_of_nodes = 0):
    model_setting_paths = []
    model_paths = []
    
    result = find_parent_folder(root_file_path)
    gnn_file_path = os.path.join(result, gnn_file_path)
    
    
    if os.path.exists(gnn_file_path):
        for root, _ , files in os.walk(gnn_file_path):
            for file in files:
                if file.endswith(".txt"):  # Filter for text files
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

    data = generate_data_from_log(logP, logM, sup, ratio, size_par)
    

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




