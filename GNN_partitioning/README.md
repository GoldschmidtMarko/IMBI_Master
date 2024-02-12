# GNN_partitioning

This folder contains all graph neural network related code

## GNN_Analysing
- Runnable file: comparison_gnn.py
- Runs a comparision of the IM-Bi with and without the gnn framewore. The results are saved into a df and plotted.

## GNN_Data_Generation
- Runnable file: gnn_generation.py, gnn_data_distribution.py
- gnn_data_distribution.py shows the data distribution present for the gnns
- gnn_generation.py contains all code for the generation of the syntethic data.
   Console inputs with the program execution define unique_indentifier, number_new_data_instances_per_category, list_grap_node_sizes.
   Relevant codelines 482 - 495.
  After execution, a folder GNN_Data is created with all files.

## GNN_Model
- Contains the trained models  *.pt and required setting files *_settings.txt.
  The files *_data.txt contain information about the used training and test data.

## GNN_Model_Generation
- Runnable file: gnn_model_construction.py
- Grabs all data from the folder GNN_Data.
- Creates and trains the graph neural network models and saves them in the folder GNN_Model
