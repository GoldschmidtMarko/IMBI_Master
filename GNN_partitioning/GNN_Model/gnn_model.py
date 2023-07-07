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

def generate_Model(num_features, hidden_dim, output_dim):
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
            output = F.relu(output)
            
            return output
    
    class GNNClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNClassifier, self).__init__()
            self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
            self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

        def forward(self, adjacency_matrix_P, adjacency_matrix_M, features):
            # Perform first graph convolution
            hidden1 = self.gc1(adjacency_matrix_P, adjacency_matrix_M, features)
            hidden1 = F.relu(hidden1)
            
            
            # Perform second graph convolution
            hidden2 = self.gc2(adjacency_matrix_P, adjacency_matrix_M, hidden1)
            hidden2 = F.relu(hidden2)

            return hidden2

    # Create the GNN classifier model
    model = GNNClassifier(num_features, hidden_dim, output_dim)
    
    return model

def transform_data_to_model(data):
    # converting the data to torch
    torch_matrix_P = torch.from_numpy(data["adjacency_matrix_P"])
    torch_matrix_M = torch.from_numpy(data["adjacency_matrix_M"])

    # normalization
    torch_matrix_P = torch_matrix_P / torch_matrix_P.max()
    torch_matrix_M = torch_matrix_M / torch_matrix_M.max()

    # Create the feature matrix
    tensor_feature_matrix = torch.tensor([[data["support"], data["ratio"]] for _ in range(data["Number_nodes"])])
    return torch_matrix_P, torch_matrix_M, tensor_feature_matrix

def get_ignore_mask(data):
    maskList = []
    for label in data["Labels"]:
        if label in data["partitionA"] or label in data["partitionB"]:
            maskList.append(1)
        else:
            maskList.append(0)
            
    # ignore padded nodes
    if len(maskList) < data["Number_nodes"]:
        maskList = maskList + [0 for _ in range(data["Number_nodes"] - len(maskList))]
        
    # converting the mask to tensor
    mask = torch.tensor(maskList, dtype=torch.float32)
    return mask

def evaluate_model(model, test_data):

    # 2. Set the Model to Evaluation Mode
    model.eval()
    
    accuracy_list = []
    fully_accurate = 0
    
    # 3. Forward Pass
    with torch.no_grad():
        for data in test_data:
            # converting the data to torch
            torch_matrix_P, torch_matrix_M, tensor_feature_matrix = transform_data_to_model(data)
            # print(torch_matrix_P)
            # print(torch_matrix_M)
            # print(tensor_feature_matrix)

            # Forward pass to get predictions
            predictions = model(torch_matrix_P, torch_matrix_M, tensor_feature_matrix)
            
            # Apply softmax to obtain probability distributions over classes
            probs = F.sigmoid(predictions)
            
            binary_predictions = torch.round(probs)
            
            # Assuming you have a tensor `mask` that indicates which nodes to ignore
            mask = get_ignore_mask(data)
            mask = torch.unsqueeze(mask, dim=1)
            
            target_tensor = torch.tensor(data["Truth"])
            
            target_tensor_transposed = torch.unsqueeze(target_tensor, dim=1)
            
            # Apply the mask to the ground truth labels and logits tensors
            masked_ground_truth_labels = target_tensor_transposed * mask
            masked_logits = binary_predictions * mask
            
            # Apply the mask to the ground truth labels and logits tensors
            masked_ground_truth_labels = target_tensor.bool()
            masked_predicted_labels = masked_logits.bool()
            
            data_accuracy = (masked_ground_truth_labels == masked_predicted_labels).float().mean().item()
            accuracy_list.append(data_accuracy)
            
            equal_mask = (masked_ground_truth_labels == masked_predicted_labels)
            if torch.all(equal_mask):
                fully_accurate += 1


    
    # Compute evaluation metrics
    accuracy = sum(accuracy_list) / len(accuracy_list)
    # (e.g., precision, recall, F1-score) 
    
    return accuracy, fully_accurate/len(accuracy_list)

def train_model(model, num_epochs, training_data):
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define the batch size
    # TODO FOR PARALLEL
    # batch_size = 1

    # Create a data loader with the specified batch size
    # data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # for batch_data in data_loader:

    
    for epoch in range(num_epochs):
        for data in training_data:
            # Set model to training mode
            model.train()

            # Zero the gradients
            optimizer.zero_grad()

            # converting the data to torch
            torch_matrix_P, torch_matrix_M, tensor_feature_matrix = transform_data_to_model(data)

            # Forward pass
            logits = model(torch_matrix_P.to(torch.float32),
                        torch_matrix_M.to(torch.float32),
                        tensor_feature_matrix)
  
            # Assuming you have a tensor `mask` that indicates which nodes to ignore
            mask = get_ignore_mask(data)
            mask = torch.unsqueeze(mask, dim=1)
            
            target_tensor = torch.tensor(data["Truth"],dtype=torch.float32)
            
            target_tensor_transposed = torch.unsqueeze(target_tensor, dim=1)
            
            # Apply the mask to the ground truth labels and logits tensors
            masked_ground_truth_labels = target_tensor_transposed * mask
            masked_logits = logits * mask

            # Compute the loss
            loss = criterion(masked_logits, masked_ground_truth_labels)

            # Backward pass
            loss.backward()
            
            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradients for {name}:")
                    print(param.grad)

            # Update the weights
            optimizer.step()

        # Print the loss for monitoring
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")
        accuracy, full_accuracy = evaluate_model(model, training_data)
        print("Avg accuracy: " + str(accuracy))
        print("Full accuracy: " + str(full_accuracy))


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
                elif state == 1 and line == "\n":
                    data["adjacency_matrix_P"] = np.vstack(matrix_P_arrayList)
                    state += 1
                    continue
                elif state == 2 and line == "\n":
                    data["adjacency_matrix_M"] = np.vstack(matrix_M_arrayList)
                    state += 1
                    continue   
                elif state == 3:
                    data["support"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 4:
                    data["ratio"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 5:
                    data["partitionA"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 6:
                    data["partitionB"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                
                if state == 1:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_P_arrayList.append(np_array)
                if state == 2:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_M_arrayList.append(np_array)
    return data

def generate_ground_truth(data):
    partitionA = data["partitionA"]
    partitionB = data["partitionB"]
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

def setup_dataSet(dataSet, max_node_size_in_dataset):
    for data in dataSet:
        data["Truth"] = generate_ground_truth(data)
        # Calculate the difference in size
        diff_size = max_node_size_in_dataset - data["adjacency_matrix_P"].shape[0]
        # Pad the adjacency matrix with zeros
        data["adjacency_matrix_P"] = np.pad(data["adjacency_matrix_P"], ((0, diff_size), (0, diff_size)), mode='constant')
        data["adjacency_matrix_M"] = np.pad(data["adjacency_matrix_M"], ((0, diff_size), (0, diff_size)), mode='constant')
        data["Truth"] = data["Truth"] + [-1 for _ in range(max_node_size_in_dataset - len(data["Truth"]))]
        data["Number_nodes"] = max_node_size_in_dataset
    return dataSet

def read_all_data_for_cut_Type(file_path, cut_type):
    dataList = []
    max_node_size_in_dataset = 0
    currentPath = file_path
    
    if os.path.exists(currentPath):
        currentPath += "/" + cut_type
        if os.path.exists(currentPath):
            for root, _ , files in os.walk(file_path):
                for file in files:
                    if file.endswith(".txt"):  # Filter for text files
                        if len(dataList) > 1:
                            break
                        
                        file_path = os.path.join(root, file)
                        
                        data = read_data_from_path(file_path)
                        max_node_size_in_dataset = max(max_node_size_in_dataset, len(data["Labels"]))
                        dataList.append(data)
                        

                        
                        
    dataList = setup_dataSet(dataList,max_node_size_in_dataset)
    return dataList
                    
def run():
    relative_path = "GNN_partitioning/GNN_Data"

    print("Reading Data")
    dataList = read_all_data_for_cut_Type(relative_path, "seq")


    # Example usage
    hidden_dim = 32  # Hidden dimension in GNN layers
    output_dim = 1  # Number of output classes

    print("Generating Model")
    model = generate_Model(2, hidden_dim, output_dim)


    # Split the data and labels into training and test sets
    train_data, test_data = train_test_split(dataList, test_size=0.2, random_state=1996)

    print("Train data size: " + str(len(train_data)))
    print("Test data size: " + str(len(test_data)))

    print()
    print("INITIAL STATISTIC")
    accuracy, full_accuracy = evaluate_model(model, test_data)
    print("Avg accuracy: " + str(accuracy))
    print("Full accuracy: " + str(full_accuracy))

    print()
    print("Training Model")
    num_epochs = 20
    train_model(model, num_epochs, train_data)

    print()
    print("FINAL STATISTIC")
    print("Evaluating Model")
    accuracy, full_accuracy = evaluate_model(model, test_data)
    print("Avg accuracy: " + str(accuracy))
    print("Full accuracy: " + str(full_accuracy))




run()





