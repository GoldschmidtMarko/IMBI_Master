

import random
import string
import os
import multiprocessing
from pm4py.objects.log.importer.xes.importer import apply as import_apply
from pm4py.objects.log.exporter.xes.exporter import apply
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import Trace
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.process_tree.exporter import exporter as tree_exporter
import sys
import time
import json
root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_best_cut_with_cut_type
from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import artificial_start_end
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import view_dfg
from pm4py.objects.process_tree.obj import ProcessTree, Operator
import warnings
from pm4py.objects.log.exporter.xes.variants.etree_xes_exp import Parameters as Export_Parameter
from GNN_partitioning.GNN_Model_Generation.gnn_models import generate_adjacency_matrix_from_log
from GNN_partitioning.GNN_Model_Generation.bi.gnn_models import generate_union_adjacency_matrices

def get_log(file_name):
  warnings.filterwarnings("ignore")
  # Export the event log to a XES file
  parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  log = import_apply(file_name + ".xes", parameters=parameter)
  return log

# generating activity names
def generate_activity_name_list(number_activites, string_length):
  def generate_random_string(length):
      # Generate a random string of given length
      letters = string.ascii_letters
      return ''.join(random.choice(letters) for _ in range(length))

  def generate_unique_strings(num_strings, string_length):
      unique_strings = set()
      max_attempts = num_strings * 10  # Maximum number of attempts to prevent infinite loop

      while len(unique_strings) < num_strings and max_attempts > 0:
          new_string = generate_random_string(string_length)
          unique_strings.add(new_string)
          max_attempts -= 1

      return list(unique_strings)

  # Generate 10 random unique strings of length 8
  return generate_unique_strings(number_activites, string_length)

def generate_trace(activity_names, trace_length):
  trace = []
  while(len(trace) < trace_length):
    trace.append(random.choice(activity_names))
    
  return trace

def get_number_depending_on_deviation(avg_value, deviation_value):
  random_number = random.gauss(avg_value, deviation_value)
  
  return int(random_number)
  
def find_best_cut_type(lopP, logM, sup, ratio, cut_type):
  result, best_cut = get_best_cut_with_cut_type(lopP,logM, cut_type, sup,ratio)
  return result, best_cut
  
def view_log(log):
  start_act_cur_dfg = start_activities_get.get_start_activities(log)
  end_act_cur_dfg = end_activities_get.get_end_activities(log)
  cur_dfg = dfg_inst.apply(log)
  view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)

def combine_process_trees_from_list(process_tree_list, operator = None):
  random_items = random.sample(process_tree_list, 2)
  processTree1 = random_items[0]
  processTree2 = random_items[1]
  process_tree_list.remove(processTree1)
  process_tree_list.remove(processTree2)
  
  if operator == None:
    operatorList = [Operator.SEQUENCE,Operator.LOOP,Operator.PARALLEL,Operator.XOR]
    new_operator = random.choice(operatorList)
  else:
    new_operator = operator
  
  new_tree_node = ProcessTree(children=[processTree1, processTree2],operator=new_operator)
  processTree1.parent = new_tree_node
  processTree2.parent = new_tree_node
  
  process_tree_list.append(new_tree_node)
  
  return process_tree_list

def add_label_to_process_trees(process_tree, input_label, operator = None):
  def get_all_leaf_nodes(tree):
    leaf_nodes = []

    if len(tree.children) == 0:
        leaf_nodes.append(tree)
    else:
      for child in tree.children:
        leaf_nodes.extend(get_all_leaf_nodes(child))

    return leaf_nodes
  
  leaf_nodes = get_all_leaf_nodes(process_tree)
  random_leaf = random.choice(leaf_nodes)
  cur_label = random_leaf.label
  
  if operator == None:
    operatorList = [Operator.SEQUENCE,Operator.LOOP,Operator.PARALLEL,Operator.XOR]
    new_operator = random.choice(operatorList)
  else:
    new_operator = operator
  
  random_leaf.label = None
  random_leaf.operator = new_operator
  
  if random.random() < 0.5:
    child_tree_1 = ProcessTree(label=cur_label)
    child_tree_2 = ProcessTree(label=input_label)
  else:
    child_tree_1 = ProcessTree(label=input_label)
    child_tree_2 = ProcessTree(label=cur_label)
    
  random_leaf.children.append(child_tree_1)
  random_leaf.children.append(child_tree_2)
  child_tree_1.parent = random_leaf
  child_tree_2.parent = random_leaf
  
  return process_tree

def generate_random_process_tree_for_cut_type(activites_list, cut_type):
  if len(activites_list) == 1:
    return ProcessTree(label=activites_list[0])
  
  generated_ProcessTrees = []
  # generating leave node
  for activity_name in activites_list:
    generated_ProcessTrees.append(ProcessTree(label=activity_name))
    
  
  # generating tau node
  number_random_tau_labels = random.randint(0,len(activites_list))
  for i in range(number_random_tau_labels):
    generated_ProcessTrees.append(ProcessTree(label=None))
    
  basic_cuts = ["exc","seq", "par", "loop"]
  special_cuts = ["exc_tau","loop_tau"]
  
  if cut_type in basic_cuts:
    # combine 2 random process Trees and add the father back to the list, until only 2 remains
    while(len(generated_ProcessTrees) > 2):
      generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees)
      
    if cut_type == "seq":
      new_operator = Operator.SEQUENCE
    elif cut_type == "par":
      new_operator = Operator.PARALLEL
    elif cut_type == "loop":
      new_operator = Operator.LOOP
    elif cut_type == "exc":
      new_operator = Operator.XOR
    generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees, new_operator)
    return generated_ProcessTrees[0] 
  elif cut_type in special_cuts:
    if cut_type == "exc_tau":
      new_operator = Operator.XOR
    elif cut_type == "loop_tau":
      new_operator = Operator.LOOP
      
    # combine 2 random process Trees and add the father back to the list, until only 1 remains
    while(len(generated_ProcessTrees) > 1):
      generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees)

    # add tauTree to operant
    tauTree = ProcessTree(label=None)
    
    root_tree = ProcessTree(children=[generated_ProcessTrees[0] , tauTree],operator=new_operator)
    tauTree.parent = root_tree
    generated_ProcessTrees[0].parent = root_tree
    
    return root_tree
      
  else:
    # combine 2 random process Trees and add the father back to the list, until only 1 remains
    while(len(generated_ProcessTrees) > 1):
      generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees)
    return generated_ProcessTrees[0] 
  
  return generated_ProcessTrees[0] 

def mutate_process_tree(process_tree, mutation_rate):
  def mutate_process_tree_operator(process_tree):
    if process_tree.operator != None:
      operators = [Operator.SEQUENCE,Operator.LOOP,Operator.PARALLEL,Operator.XOR]
      operators.remove(process_tree.operator)
      new_operator = random.choice(operators)
      process_tree.operator = new_operator
  
  def get_tree_depth(tree):
    if len(tree.children) == 0:
      return 0

    max_child_depth = 0
    for child in tree.children:
        child_depth = get_tree_depth(child)
        max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth + 1
  
  def get_all_operator_nodes(tree):
    operator_nodes = []

    if tree.operator != None:
        operator_nodes.append(tree)
    
    for child in tree.children:
        operator_nodes.extend(get_all_operator_nodes(child))

    return operator_nodes
  
  operator_nodes = get_all_operator_nodes(process_tree)
  for node in operator_nodes:
    current_mutation_chance = mutation_rate / pow(2, get_tree_depth(node))
    if random.random() < current_mutation_chance:
      mutate_process_tree_operator(node)

def generate_mutated_process_tree_from_process_tree(activites_list, process_tree, mutation_rate = 0.5):
  def get_all_labels(tree):
    labels = []
    if tree.label != None:
      labels.append(tree.label)  # Add the current node's label

    for child in tree.children:
        labels.extend(get_all_labels(child))  # Recursively get labels from child nodes

    return labels
  
  def remove_labels(tree, label_list):
    if tree.label in label_list:
      tree.label = None

    for child in tree.children:
        remove_labels(child, label_list)  # Recursively

  
  def copy_process_tree(tree):
    if tree is None:
        return None

    new_tree = ProcessTree(label=tree.label, operator=tree.operator)
    for child in tree.children:
        new_child = copy_process_tree(child)
        new_tree.children.append(new_child)
        new_child.parent = new_tree

    return new_tree

  new_process_tree = copy_process_tree(process_tree)
  
  tree_labels = get_all_labels(new_process_tree)
  
  if len(tree_labels) < len(activites_list):
    additional_labels = set(activites_list) - set(tree_labels)
    for cur_label in additional_labels:
      new_process_tree = add_label_to_process_trees(new_process_tree, cur_label)
    
  else:
    # remove labels
    removing_labels = set(tree_labels) - set(activites_list)
    remove_labels(new_process_tree, removing_labels)
  
  # adding mutation to operators
  mutate_process_tree(new_process_tree, mutation_rate)
  
  return new_process_tree 

def get_percentage_of_noise():
  # Parameters
  mean = 0.2
  sigma = 0.1
  max_noise_factor = 0.4

  # Generate random samples
  samples = []
  while len(samples) < 10000:
      random_noise = random.gauss(mean, sigma)
      if 0 <= random_noise <= max_noise_factor:
          return random_noise
  
  return mean

def generate_log_from_process_tree_for_cut_type(activites_list, process_tree, seed):
  random.seed(seed)
  log = pm4py.play_out(process_tree)
  
  noise_Factor = get_percentage_of_noise()
  
  number_noise_traces = int(noise_Factor * len(log))
  
  number_avg_trace_length = get_avg_trace_length_from_log(log)
  number_avg_trace_length_deviation = (number_avg_trace_length * 0.2)
  
  # noise added
  for i in range (number_noise_traces):
    trace = Trace()
    cur_trace_length = get_number_depending_on_deviation(number_avg_trace_length, number_avg_trace_length_deviation)
    generated_trace = generate_trace(activites_list, trace_length=cur_trace_length)
    for event in generated_trace:
      event1 = {"concept:name": str(event)}
      trace.append(event1)

    # Add the trace to the event log
    log.append(trace)
    
  # save_log_as_dfg(log, "log2")
  return log

def get_avg_trace_length_from_log(log):
  # Calculate the average trace length
  total_trace_length = 0
  total_traces = len(log)

  for trace in log:
      trace_length = len(trace)
      total_trace_length += trace_length

  average_trace_length = total_trace_length / total_traces
  return average_trace_length

def save_tree(tree, file_name):
  def serialize_tree(node):
        if node is None:
            return None
        serialized_node = {
            "label": node.label,
            "operand": str(node.operator),
            "children": [serialize_tree(child) for child in node.children],
        }
        return serialized_node

  serialized_tree = serialize_tree(tree)

  with open(file_name, "w") as file:
      json.dump(serialized_tree, file, indent=4)
  
  
def save_data(file_name, adj_matrix_P, adj_matrix_M, unique_node,cut_type, sup, ratio, datapiece, partitionA, partitionB, score, unique_activity_count_P, unique_activity_count_M, size_par,random_seed_P,random_seed_M):
  with open(file_name + ".txt", "w") as file:
    file.write("# unique_node | unique_activity_count_P | unique_activity_count_M | adj_matrix_P | adj_matrix_M | cut_type | sup | ratio | size_par | dataitem | partitionA | partitionB | score | random_seed_P | random_seed_M " + "\n")
    # unique_node
    outputString = ""
    for value in unique_node:
      outputString += str(value) + " "
    file.write(outputString + "\n")
    # unique_activity_count_P
    outputString = ""
    for value in unique_activity_count_P:
      outputString += str(value) + " "
    file.write(outputString + "\n")
    # unique_activity_count_M
    outputString = ""
    for value in unique_activity_count_M:
      outputString += str(value) + " "
    file.write(outputString + "\n")
    # adj_matrix_P
    for row in adj_matrix_P:
      outputString = ""
      for value in row:
        outputString += str(value) + " "
      file.write(outputString + "\n")
    file.write("\n")
    # adj_matrix_M
    for row in adj_matrix_M:
      outputString = ""
      for value in row:
        outputString += str(value) + " "
      file.write(outputString + "\n")
    file.write("\n")
    # cut_type
    file.write(str(cut_type) + "\n")
    # sup
    file.write(str(sup) + "\n")
    # ratio
    file.write(str(ratio) + "\n")
    # size_par
    file.write(str(size_par) + "\n")
    # datapiece
    file.write(str(datapiece) + "\n")
    # partitionA
    outputString = ""
    for value in partitionA:
      outputString += str(value) + " "
    file.write(outputString + "\n")
    # partitionB
    outputString = ""
    for value in partitionB:
      outputString += str(value) + " "
    file.write(outputString + "\n")
    # score
    file.write(str(score) + "\n")
    # random_seed_P
    file.write(str(random_seed_P) + "\n")
    # random_seed_M
    file.write(str(random_seed_M) + "\n")
  
def get_partial_activity_names(activity_names, percentage_is_subset, min_percentage_subset = 0.5):
  random_number = random.random()
  
  if random_number < percentage_is_subset:
    # subset
    min = int(len(activity_names) * min_percentage_subset)
    max = len(activity_names)
    number_subset_activities = random.randint(min,max)    
    new_activity_list = random.sample(activity_names, number_subset_activities)
    return new_activity_list
    
  else:
    # superset
    def generate_random_string(length):
      # Generate a random string of given length
      letters = string.ascii_letters
      return ''.join(random.choice(letters) for _ in range(length))
    
    activity_string_length = len(activity_names[0])
    number_additional_activities = 1
    if random.random() < 0.5:
      number_additional_activities = 2

    max_attempts = number_additional_activities * 20  # Maximum number of attempts to prevent infinite loop
    new_activity_names = set(activity_names)

    while(len(new_activity_names) < len(activity_names) + number_additional_activities) and max_attempts > 0:
      newAct = generate_random_string(activity_string_length)
      new_activity_names.add(newAct)
      max_attempts -= 1

    return list(new_activity_names)
      
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
  
def is_log_consistent(log1, log2):
  variants1 = pm4py.get_variants(log1)
  variants2 = pm4py.get_variants(log2)
  
  if len(variants1) != len(variants2):
    return False
  if len(log1) != len(log2):
    return False
  for variant in variants1:
    if variant not in variants2:
      return False
  return True
             
def generate_data_piece_for_cut_type(file_path, number_of_activites, support, ratio, data_piece_index, cut_type):
  if number_of_activites <= 0:
    return

  warnings.filterwarnings("ignore")
  
  folder_name = file_path + "/" + cut_type
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
  
  folder_name += "/Data_" + str(number_of_activites)
  ending_File_string = str(number_of_activites) + "_Sup_"+ str(support) + "_Ratio_" + str(ratio) + "_" + str(data_piece_index)

  
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
     
  folder_name += "/Sup_" + str(support) + "_Ratio_" + str(ratio)
  
  # Check if the folder with param already exists
  if not os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
      
  # Check if files already exist
  checking_files_names = [folder_name + "/treeP_" + ending_File_string,
                          folder_name + "/treeM_" + ending_File_string,
                          folder_name + "/Cut_" + ending_File_string]
  for files in checking_files_names:
    if os.path.exists(files):
        # Delete the file
        os.remove(files)
  
  amount_tries = 0
  max_amount_tries = 5
  while(amount_tries < max_amount_tries):
    amount_tries += 1
    
    activity_list = generate_activity_name_list(number_of_activites,
                                                random.randint(5,8))
    
    process_tree_P = generate_random_process_tree_for_cut_type(activity_list, cut_type)
    
    random_seed_P = random.randint(100000, 999999)
    logP = generate_log_from_process_tree_for_cut_type(activity_list, process_tree_P, random_seed_P)
    
    # random.seed(random_seed_P)
    # logP_ = generate_log_from_process_tree_for_cut_type(activity_list, process_tree_P)
    
    # if not is_log_consistent(logP, logP_):
    #   print("Inconsistent logP")
    
    percentage_is_subset = 0.8
    activity_list_near = get_partial_activity_names(activity_list, percentage_is_subset, min_percentage_subset = 0.5)
    process_tree_M = generate_mutated_process_tree_from_process_tree(activity_list_near, process_tree_P, mutation_rate=0.5)

    random_seed_M = random.randint(100000, 999999)
    logM = generate_log_from_process_tree_for_cut_type(activity_list_near, process_tree_M, random_seed_M)
    
    result, cut = find_best_cut_type(logP, logM, support, ratio, cut_type)

    if result == True:
      # result, cut = find_best_cut_type(logP,logM,support,ratio, cut_type)
      try:
        save_tree(process_tree_P, folder_name + "/treeP_" + ending_File_string  + ".json")
      except:
        print("Error saving treeP:")
        print(process_tree_P)
        continue
        
      try:
        save_tree(process_tree_M, folder_name + "/treeM_" + ending_File_string  + ".json")
      except:
        
        if os.path.exists(folder_name + "/treeP_" + ending_File_string + ".json"):
          # Delete the process_tree_P
          os.remove(folder_name + "/treeP_" + ending_File_string  + ".json")
          
        print("Error saving treeM:")
        print(process_tree_M)
        continue

      unique_node_P, adj_matrix_P = generate_adjacency_matrix_from_log(logP)
      unique_node_M, adj_matrix_M = generate_adjacency_matrix_from_log(logM)
      unique_nodeList, matrix_P, matrix_M = generate_union_adjacency_matrices(adj_matrix_P,unique_node_P,adj_matrix_M,unique_node_M)
      
      logP_art = artificial_start_end(logP.__deepcopy__())
      logM_art = artificial_start_end(logM.__deepcopy__())
      
      activity_count_P = get_activity_count(logP_art)
      activity_count_M = get_activity_count(logM_art)
      unique_activity_count_P = get_activity_count_list_from_unique_list(activity_count_P, unique_nodeList)
      unique_activity_count_M = get_activity_count_list_from_unique_list(activity_count_M, unique_nodeList)
      
      size_par = len(logP) / len(logM)

      save_data(folder_name + "/Data_" + str(data_piece_index), matrix_P, matrix_M,unique_nodeList,cut_type,support,ratio,data_piece_index,cut[0][0],cut[0][1], cut[4],unique_activity_count_P,unique_activity_count_M,size_par,random_seed_P,random_seed_M)
      return 1

      
    return 0
   
if __name__ == '__main__':
  print("Error, not supported to run as main")

