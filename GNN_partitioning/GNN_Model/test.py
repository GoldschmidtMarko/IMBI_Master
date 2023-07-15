import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
  def __init__(self, in_features, out_features):
      super(GraphConvolutionLayer, self).__init__()
      self.linear = nn.Linear(in_features, out_features)

  def forward(self, adjacency_matrix_1, adjacency_matrix_2, features):
      # First graph convolution
      output_1 = torch.matmul(adjacency_matrix_1, features)
      output_1 = self.linear(output_1)
      output_1 = F.relu(output_1)
      
      # Second graph convolution
      output_2 = torch.matmul(adjacency_matrix_2, features)
      output_2 = self.linear(output_2)
      output_2 = F.relu(output_2)

      return output_1 + output_2

class GCNClassifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
      super(GCNClassifier, self).__init__()
      self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
      self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)
      

  def forward(self, adjacency_matrix_1, adjacency_matrix_2, features):
      hidden1 = self.gc1(adjacency_matrix_1, adjacency_matrix_2, features)
      
      hidden2 = self.gc2(adjacency_matrix_1, adjacency_matrix_2, hidden1)

      return hidden2

# Example usage
input_dim = 10
hidden_dim = 16
output_dim = 1

# Create random adjacency matrices, features, and global variables
adjacency_matrix_1 = torch.randn(10, 10)
adjacency_matrix_2 = torch.randn(10, 10)
features = torch.randn(10, input_dim)
global_variables = torch.randn(10, 2)

# Create GCN model
model = GCNClassifier(input_dim, hidden_dim, output_dim)

# Forward pass
output = model(adjacency_matrix_1, adjacency_matrix_2, features, global_variables)

print("Output shape:", output.shape)
