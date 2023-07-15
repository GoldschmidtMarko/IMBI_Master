import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj


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
        def __init__(self, input_features, hidden_dim, output_dim, number_global_features = 2):
            super(GNNClassifier, self).__init__()
            # + 2 for the global features sup and ratio
            self.gc1 = GraphConvolutionLayer(input_features + number_global_features, hidden_dim)
            # self.dense = nn.Linear(hidden_dim, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + number_global_features, output_dim)

        def forward(self, adjacency_matrix_P, adjacency_matrix_M, global_features):
            
            # Placeholder tensor when there are no node features
            features = torch.ones(adjacency_matrix_P.size(0), self.gc1.linear_P.in_features - 2)

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
            self.gc1 = GraphConvolutionLayer(64 + 2, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 2, output_dim)
            

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
            self.gc1 = GraphConvolutionLayer(input_dim + 2, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 2, output_dim)
            

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
            self.gc1 = GraphConvolutionLayer(input_dim + 2, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 2, output_dim)

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
            self.gc1 = GraphConvolutionLayer(input_dim + 2, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim + 2, hidden_dim)
            self.gc3 = GraphConvolutionLayer(hidden_dim + 2, hidden_dim)
            self.gc4 = GraphConvolutionLayer(hidden_dim + 2, hidden_dim)
            self.gc5 = GraphConvolutionLayer(hidden_dim + 2, hidden_dim)
            self.gc6 = GraphConvolutionLayer(hidden_dim + 2, output_dim)

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
            self.linear = nn.Linear(2 * (out_features + 2), out_features)

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
            self.dense1 = nn.Linear(input_dim*4 + 2, hidden_dim)
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
            # hidden3 = F.relu(self.dense3(hidden2))
            # hidden4 = F.relu(self.dense3(hidden3))
            output = self.output_layer(hidden2)
            
            return output
    
    # Create the GNN classifier model
    model = GCNClassifier(input_dim, hidden_dim, output_dim)
    
    return model

# GIN (Graph Isomorphism Networks)   , throws error
def generate_Model_V10(input_dim, hidden_dim, output_dim):
    class GINConv(nn.Module):
        def __init__(self, in_features, out_features, hidden_dim=32):
            super(GINConv, self).__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_features),
                nn.ReLU()
            )
            self.bn = nn.BatchNorm1d(out_features)
            self.eps = nn.Parameter(torch.zeros(1, out_features))
            
        def forward(self, x, edge_index):
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x))
            out = self.bn(out)
            return out
        
        def message(self, x_j):
            return x_j

        def propagate(self, edge_index, x):
            row, col = edge_index
            return self.message(x[col])

    class GIN(nn.Module):
        def __init__(self, in_features, hidden_dim, num_classes):
            super(GIN, self).__init__()
            self.conv1 = GINConv(in_features, in_features, hidden_dim)
            self.conv2 = GINConv(hidden_dim, in_features, hidden_dim)
            self.fc = nn.Linear(hidden_dim + 1, num_classes)  # +1 for global feature
            
        def forward(self, adjacency_matrix_1, adjacency_matrix_2, node_features_1, node_features_2, global_feature):
            x1 = F.relu(self.conv1(node_features_1, adjacency_matrix_1))
            x1 = F.relu(self.conv2(x1, adjacency_matrix_1))
            
            x2 = F.relu(self.conv1(node_features_2, adjacency_matrix_2))
            x2 = F.relu(self.conv2(x2, adjacency_matrix_2))
            
            x = torch.cat((x1, x2), dim=0)  # Concatenate node representations from both graphs
            x = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(x.device))
            x = torch.cat((x, global_feature), dim=1)  # Concatenate global feature with aggregated node representations
            x = self.fc(x)
            return x

    # Create the GNN classifier model
    model = GIN(input_dim, hidden_dim, output_dim)
    
    return model

# GAT (Graph Attention Networks)   , throws error
def generate_Model_V11(input_dim, hidden_dim, output_dim):
    class GAT(nn.Module):
        def __init__(self, in_features, hidden_dim, num_heads, num_classes):
            super(GAT, self).__init__()
            self.gatconv1 = GATConv(in_features, hidden_dim, heads=num_heads)
            self.gatconv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
            self.fc = nn.Linear(hidden_dim * num_heads + 1, num_classes)  # +1 for global feature
            
        def forward(self, adjacency_matrix_1, adjacency_matrix_2, node_features_1, node_features_2, global_feature):
            adjacency_matrix_1 = adjacency_matrix_1.long()
            adjacency_matrix_2 = adjacency_matrix_2.long()
            
            node_features_1 = node_features_1.long()
            node_features_2 = node_features_2.long()
            
            adj1 = to_dense_adj(adjacency_matrix_1).squeeze()
            adj2 = to_dense_adj(adjacency_matrix_2).squeeze()
            
            adj1 = adj1.long()
            
            temp = self.gatconv1(node_features_1, adj1.long())

            x1 = F.relu(temp.float())
            x1 = F.relu(self.gatconv2(x1, adj1).float())
            
            x2 = F.relu(self.gatconv1(node_features_2, adj2).float())
            x2 = F.relu(self.gatconv2(x2, adj2).float())
            
            x = torch.cat((x1, x2), dim=0)  # Concatenate node representations from both graphs
            x = torch.mean(x, dim=0, keepdim=True)  # Average the node representations across the graphs
            x = torch.cat((x, global_feature), dim=1)  # Concatenate global feature with node representations
            x = self.fc(x)
            return x

    # Create the GNN classifier model
    num_heads = 4  # Number of attention heads
    model = GAT(input_dim, hidden_dim, num_heads, output_dim)
    
    return model


def generate_model(model_number, input_dim, hidden_dim, output_dim):
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
    
def generate_model_args(args):
    return generate_model(*args)


def transform_data_to_model(model_number, data):
    # converting the data to torch
    torch_matrix_P = torch.from_numpy(data["adjacency_matrix_P"]).to(torch.float32)
    torch_matrix_M = torch.from_numpy(data["adjacency_matrix_M"]).to(torch.float32)
    
    if model_number == 2:
        epsilon = 1e-8
        
        torch_matrix_P += torch.eye(torch_matrix_P.size(0)) * epsilon
        torch_matrix_M += torch.eye(torch_matrix_M.size(0)) * epsilon

    # normalization, removed since it could have negative effects
    # torch_matrix_P = torch_matrix_P / torch_matrix_P.max()
    # torch_matrix_M = torch_matrix_M / torch_matrix_M.max()

    # Create the feature matrix
    tensor_feature_matrix = torch.tensor([[data["Support"], data["Ratio"]] for _ in range(data["Number_nodes"])])
    return torch_matrix_P, torch_matrix_M, tensor_feature_matrix

def get_ignore_mask(data):
    # 0
    ignoreValue = 0
    
    maskList = []
    for label in data["Labels"]:
        if label in data["partitionA"] or label in data["partitionB"]:
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
    torch_matrix_P, torch_matrix_M, tensor_feature_matrix = transform_data_to_model(model_number, data)
    
    if model_number == 1:
        return model(torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 2:
        return model(torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 3:
        return model(torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 4:
        return model(torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 5:
        mapped_matrix = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        return model(mapped_matrix, torch_matrix_P)
    elif model_number == 6:
        adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 7:
        adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 8:
        adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 9:
        adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 10:
        adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
    elif model_number == 11:
        adj_mat_P = torch.where(torch_matrix_P > 0, torch.tensor(1.0), torch_matrix_P)
        adj_mat_M = torch.where(torch_matrix_M > 0, torch.tensor(1.0), torch_matrix_M)
        return model(adj_mat_P, adj_mat_M, torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
        






