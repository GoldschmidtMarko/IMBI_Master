import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

class GlobalGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GlobalGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.global_attention = nn.Linear(hidden_dim, 1)
        self.final_linear = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        
        # Global attention mechanism
        global_attention_scores = self.global_attention(x2)
        global_attention_weights = torch.softmax(global_attention_scores, dim=0)
        
        global_embedding = torch.sum(global_attention_weights * x2, dim=1)  # Sum along the node dimension
        
        pooled = torch.mean(x2, dim=0)  # Mean pooling
        
        combined = torch.cat([pooled.unsqueeze(0), global_embedding.unsqueeze(0)], dim=0)

        output = self.final_linear(combined)
        return output

# Create a toy graph for demonstration
edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 3, 1]], dtype=torch.long)
x = torch.randn(4, 16)  # Node features
data = Data(x=x, edge_index=edge_index)

# Set hyperparameters
input_dim = 16
hidden_dim = 32
num_classes = 2

# Instantiate the GlobalGraphSAGE model
model = GlobalGraphSAGE(input_dim, hidden_dim, num_classes)

# Forward pass
output = model(data)
print("Bisection Predictions Shape:", output.shape)  # Print the shape of the output
