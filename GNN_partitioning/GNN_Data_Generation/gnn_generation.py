

import random
import string
import os
import multiprocessing
from pm4py.objects.log.exporter.xes.exporter import apply
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import Trace
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import sys
sys.path.append('c:\\Users\\Marko\\Desktop\\GIt\\IMBI_Master')
from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_best_cut
from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_best_cut_with_cut_type
from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import artificial_start_end
from local_pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import view_dfg
from pm4py.objects.process_tree.obj import ProcessTree, Operator
import shutil
import numpy as np
import logging
from tqdm import tqdm
import warnings
from pm4py.objects.log.exporter.xes.variants.etree_xes_exp import Parameters as Export_Parameter
from pm4py.objects.log.importer.xes.variants.iterparse import Parameters as Import_Parameter


random_seed = 1996

number_activities = 6

number_avg_trace_length = 6
number_avg_trace_length_deviation = 2

number_avg_traces = 10
number_avg_traces_deviation = 2

relative_path = "GNN_partitioning/GNN_Data"


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
  return random.randint(avg_value - deviation_value, avg_value + deviation_value)
  
# Todo trace frequency distribution (e.g. 80/20 or 40/20/10/5/3/3... )
def generate_Random_Log(number_avg_traces, number_avg_traces_deviation, number_avg_trace_length, number_avg_trace_length_deviation, file_name):
  activity_names = generate_activity_name_list(number_activities,4)
  cur_case_id = 0
  cur_number_of_traces = get_number_depending_on_deviation(number_avg_traces, number_avg_traces_deviation)
  
  # Create an empty event log
  log = EventLog()

  # Create a trace
  while(cur_case_id < cur_number_of_traces):
    trace = Trace()
    trace.attributes["concept:name"] = "Trace" + str(cur_case_id)
    generated_trace = generate_trace(activity_names, trace_length=get_number_depending_on_deviation(number_avg_trace_length, number_avg_trace_length_deviation))
    for event in generated_trace:
      event1 = {"concept:name": str(event), "time:timestamp": "2023-06-24T10:00:00"}
      trace.append(event1)
    
    # Add the trace to the event log
    log.append(trace)
    cur_case_id += 1
    
  # Export the event log to a XES file
  apply(log, file_name + ".xes")

def log_statistic(log_path):
  log = xes_importer.apply(log_path + ".xes")
  # log = pm4py.read_xes(log_path + ".xes")
  variants = pm4py.get_variants(log)
  print("Number of variants: " + str(len(variants)))
  print("Number of traces: " + str(len(log)))
  
  unique_activities = set()
  for trace in log:
      for event in trace:
          unique_activities.add(event["concept:name"])

  # Get the number of unique activities
  num_unique_activities = len(unique_activities)

  # Print the number of unique activities
  print("Number of unique activities:", num_unique_activities)
  
  # Initialize variables for longest and smallest trace length
  longest_length = 0
  smallest_length = float("inf")

  # Iterate over the traces
  for trace in log:
      trace_length = len(trace)
      
      # Update longest length if necessary
      if trace_length > longest_length:
          longest_length = trace_length
      
      # Update smallest length if necessary
      if trace_length < smallest_length:
          smallest_length = trace_length

  # Print the longest and smallest trace length
  print("Longest trace length:", longest_length)
  print("Smallest trace length:", smallest_length)

def find_best_cut(lopP, logM, sup, ratio, pruning_threshold):
  best_cut = get_best_cut(lopP,logM,sup,ratio,pruning_threshold)
  return best_cut

def find_best_cut_type(lopP, logM, sup, ratio, pruning_threshold, cut_type):
  result, best_cut = get_best_cut_with_cut_type(lopP,logM, cut_type, sup,ratio,pruning_threshold)
  return result, best_cut
  
def view_log(log):
  start_act_cur_dfg = start_activities_get.get_start_activities(log)
  end_act_cur_dfg = end_activities_get.get_end_activities(log)
  cur_dfg = dfg_inst.apply(log)
  view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)

def generate_random_process_tree(activites_list):
  generated_ProcessTrees = []
  # generating leave node
  for activity_name in activites_list:
    generated_ProcessTrees.append(ProcessTree(label=activity_name))
    
  # generating tau node
  number_random_tau_labels = random.randint(0,len(activites_list))
  for i in range(number_random_tau_labels):
    generated_ProcessTrees.append(ProcessTree(label=None))
    
  # combine 2 random process Trees and add the father back to the list, until only 1 remains
  while(len(generated_ProcessTrees) > 1):
    generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees)
    
    
  # print("Generated Tree")
  # generated_ProcessTrees[0]._print_tree()
  return generated_ProcessTrees[0] 


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
  
  process_tree_list.append(ProcessTree(children=[processTree1, processTree2],operator=new_operator))
  
  return process_tree_list

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
  special_cuts = ["exc-tau","loop_tau"]
  
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
    if cut_type == "exc-tau":
      new_operator = Operator.XOR
    elif cut_type == "loop_tau":
      new_operator = Operator.LOOP
      
    # combine 2 random process Trees and add the father back to the list, until only 1 remains
    while(len(generated_ProcessTrees) > 1):
      generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees)

    # add tauTree to operant
    tauTree = ProcessTree(label=None)
    
    return ProcessTree(children=[generated_ProcessTrees[0] , tauTree],operator=new_operator)
      
  else:
    # combine 2 random process Trees and add the father back to the list, until only 1 remains
    while(len(generated_ProcessTrees) > 1):
      generated_ProcessTrees = combine_process_trees_from_list(generated_ProcessTrees)
    return generated_ProcessTrees[0] 
  
  return generated_ProcessTrees[0] 


# TODO make noise similar to traces by small local mutation
def generate_log_from_process_tree(activites_list, noise_factor = 0):
  process_tree = generate_random_process_tree(activites_list)
  log = pm4py.play_out(process_tree)
  number_noise_traces = int(noise_factor * len(log))
  
  number_avg_trace_length = int(get_avg_trace_length_from_log(log))
  number_avg_trace_length_deviation = int((number_avg_trace_length / 8))
  
  # noise added
  for i in range (number_noise_traces):
    trace = Trace()
    generated_trace = generate_trace(activites_list, trace_length=get_number_depending_on_deviation(number_avg_trace_length, number_avg_trace_length_deviation))
    for event in generated_trace:
      event1 = {"concept:name": str(event)}
      trace.append(event1)

    # Add the trace to the event log
    log.append(trace)
    
  return log

# TODO make noise similar to traces by small local mutation
def generate_log_from_process_tree_for_cut_type(activites_list, cut_type, noise_factor = 0):
  process_tree = generate_random_process_tree_for_cut_type(activites_list, cut_type)
  log = pm4py.play_out(process_tree)
  number_noise_traces = int(noise_factor * len(log))
  
  number_avg_trace_length = int(get_avg_trace_length_from_log(log))
  number_avg_trace_length_deviation = int((number_avg_trace_length / 8))
  
  # noise added
  for i in range (number_noise_traces):
    trace = Trace()
    generated_trace = generate_trace(activites_list, trace_length=get_number_depending_on_deviation(number_avg_trace_length, number_avg_trace_length_deviation))
    for event in generated_trace:
      event1 = {"concept:name": str(event)}
      trace.append(event1)

    # Add the trace to the event log
    log.append(trace)
    
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

  # print("P")
  # print(nodeListP)
  # print(matrix_P)

  # print("")
  # print(common_labels)
  # print(expanded_matrix1)
  # print("")
  
  # print("M")
  # print(nodeListM)
  # print(matrix_M)

  # print("")
  # print(common_labels)
  # print(expanded_matrix2)
  
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

  
def get_min_trace_length_from_log(log):
   # Initialize variables for longest and smallest trace length
  smallest_length = float("inf")

  # Iterate over the traces
  for trace in log:
      trace_length = len(trace)
      
      # Update smallest length if necessary
      if trace_length < smallest_length:
          smallest_length = trace_length
  return smallest_length

def get_max_trace_length_from_log(log):
   # Initialize variables for longest and smallest trace length
  longest_length = 0

  # Iterate over the traces
  for trace in log:
      trace_length = len(trace)
      
      # Update longest length if necessary
      if trace_length > longest_length:
          longest_length = trace_length
  return longest_length

def get_avg_trace_length_from_log(log):
  # Calculate the average trace length
  total_trace_length = 0
  total_traces = len(log)

  for trace in log:
      trace_length = len(trace)
      total_trace_length += trace_length

  average_trace_length = total_trace_length / total_traces
  return average_trace_length

def save_log(log, file_name):
  # Export the event log to a XES file
  parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  apply(log, file_name + ".xes", parameters=parameter)
  
def save_data(file_name, adj_matrix_P, adj_matrix_M, unique_node, sup, ratio, partitionA, partitionB):
  with open(file_name + ".txt", "w") as file:
    file.write("# unique_node | adj_matrix_P | adj_matrix_M | sup | ratio | partitionA | partitionB" + "\n")
    # adj_matrix_P
    outputString = ""
    for value in unique_node:
      outputString += str(value) + " "
    file.write(outputString + "\n")
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
    # sup
    file.write(str(sup) + "\n")
    # ratio
    file.write(str(ratio) + "\n")
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
    
def get_partial_activity_names(activity_names, number_activites, activity_string_length, number_activity_intersections):
  input_activity_names = activity_names
  new_activity_names = set()
  for i in range(number_activity_intersections):
    if len(input_activity_names) > 0:
      act = random.choice(input_activity_names)
      input_activity_names.remove(act)
      new_activity_names.add(act)
      
  def generate_random_string(length):
      # Generate a random string of given length
      letters = string.ascii_letters
      return ''.join(random.choice(letters) for _ in range(length))
  
  max_attempts = number_activites * 10  # Maximum number of attempts to prevent infinite loop
  
  while(len(new_activity_names) < number_activites) and max_attempts > 0:
    newAct = generate_random_string(activity_string_length)
    new_activity_names.add(newAct)
    max_attempts -= 1
    
  return list(new_activity_names)
      
def generate_data_piece_for_cut_type(file_path, number_of_activites, support, ratio, pruning_threshold, data_piece_index, cut_type):
  if number_of_activites <= 0:
    return

  warnings.filterwarnings("ignore")
  
  folder_name = file_path + "/" + cut_type
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
  
  folder_name += "/Data_" + str(number_of_activites)
  ending_File_string = str(number_of_activites) + "_Sup_"+ str(support) + "_Ratio_" + str(ratio) + "_Pruning_" + str(pruning_threshold) + "_Data_" + str(data_piece_index)

  
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
     
  folder_name += "/Sup_" + str(support) + "_Ratio_" + str(ratio) + "_Pruning_" + str(pruning_threshold)
  
  # Check if the folder with param already exists
  if not os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
      
  # Check if files already exist
  checking_files_names = [folder_name + "/logP_" + ending_File_string,
                          folder_name + "/logM_" + ending_File_string,
                          folder_name + "/Cut_" + ending_File_string]
  for files in checking_files_names:
    if os.path.exists(files):
        # Delete the file
        os.remove(files)
  
  activity_list = generate_activity_name_list(number_of_activites,
                                              random.randint(5,8))
  
  logP = generate_log_from_process_tree_for_cut_type(activity_list,cut_type, 0)
  
  number_activity_intersections = random.randint(0,number_of_activites)
  
  activity_list_near = get_partial_activity_names(activity_list,number_of_activites,
                                                  random.randint(5,8),
                                                  number_activity_intersections)
  
  logM = generate_log_from_process_tree_for_cut_type(activity_list_near, cut_type, 0)
  

  # save_log(logP, folder_name + "/logP_" + ending_File_string)
  # save_log(logM, folder_name + "/logM_" + ending_File_string)
  
  result, cut = find_best_cut_type(logP,logM,support,ratio, pruning_threshold, cut_type)
  
  if result == True:
    unique_node_P, adj_matrix_P = generate_adjacency_matrix_from_log(logP)
    unique_node_M, adj_matrix_M = generate_adjacency_matrix_from_log(logM)
    unique_nodeList, matrix_P, matrix_M = generate_union_adjacency_matrices(adj_matrix_P,unique_node_P,adj_matrix_M,unique_node_M)
    
    save_data(folder_name + "/Data_" + str(data_piece_index), matrix_P, matrix_M,unique_nodeList,support,ratio,cut[0][0],cut[0][1])
    
    
  
def generate_data_piece_star_function_cut_Type(args):
    return generate_data_piece_for_cut_type(*args)
    
def generate_data(file_path, sup_step, ratio_step, pruning_threshold, use_parallel = True):
  if os.path.exists(file_path):
    shutil.rmtree(file_path)
  
  # Check if the folder already exists
  if not os.path.exists(file_path):
      # Create the folder
      os.makedirs(file_path)
      
  
      
  num_processors_available = multiprocessing.cpu_count()
  print("Number of available processors:", num_processors_available)
  num_processors = max(1,round(num_processors_available/2))
  
  
  max_number_activites = 5
  number_of_data_pieces_per_variation = 30
  
  # Setup sup list
  if sup_step == 0:
    sup_list = [0]
  else:
    sup_list = np.round(np.arange(0,1 + sup_step,sup_step),1)
    
  # Setup ratio list
  if ratio_step == 0:
    ratio_list = [0]
  else:
    ratio_list = np.round(np.arange(0,1 + ratio_step,ratio_step),1)
    
  # excluded for now: single_activity, none
  # TODO add loop1
  cut_types = ["exc", "exc-tau", "seq", "par", "loop_tau", "loop"]
    
  # def generate_data_piece(file_path, number_of_activites, support, ratio):
  list_data_pool = [(file_path, number_activ, sup, ratio, pruning_threshold, data_piece_index, cut_type)
              for number_activ in range(2,max_number_activites + 1)
              for sup in sup_list
              for ratio in ratio_list
              for data_piece_index in range(1,number_of_data_pieces_per_variation + 1)
              for cut_type in cut_types]

  if use_parallel:
    print("Number of used processors:", num_processors)
    # Create a pool of workers
    with multiprocessing.Pool(num_processors) as pool:
      list(tqdm(pool.imap(generate_data_piece_star_function_cut_Type, list_data_pool), total=len(list_data_pool)))
      
  else:
    for it in list_data_pool:
      print("Running: " + str(it))
      generate_data_piece_star_function_cut_Type(it)


   
def get_labeled_data_cut_type_distribution(file_path, sup_step, ratio_step, pruning_threshold):
  result_dic = dict()
  
  # Setup sup list
  if sup_step == 0:
    sup_list = [0]
  else:
    sup_list = np.round(np.arange(0,1 + sup_step,sup_step),1)
    
  # Setup ratio list
  if ratio_step == 0:
    ratio_list = [0]
  else:
    ratio_list = np.round(np.arange(0,1 + ratio_step,ratio_step),1)
    
  cut_types = ["exc", "exc-tau", "seq", "par", "loop_tau", "loop"]
    
  for cut in cut_types:
    current_path_cut = file_path + "/" + cut
    if os.path.exists(current_path_cut):
      result_dic[cut] = dict()
      for integer in range(2,30):
        current_path_cut_Data = current_path_cut + "/Data_" + str(integer)
        if os.path.exists(current_path_cut_Data):
          if integer not in result_dic[cut]:
            result_dic[cut][integer] = 0
          for sup in sup_list:
            for ratio in ratio_list:
              sup_ratio_string = "Sup_" + str(sup) + "_Ratio_" + str(ratio) + "_Pruning_" + str(pruning_threshold)
              current_path_variant = current_path_cut_Data + "/" + sup_ratio_string
              if os.path.exists(current_path_variant):
                for data_integer in range(1,50):
                  current_path_variant_data = current_path_variant + "/" + "Data_" + str(data_integer) + ".txt"
                  if os.path.exists(current_path_variant_data):
                    if integer not in result_dic[cut]:
                      result_dic[cut][integer] = 1
                    else:
                      result_dic[cut][integer] += 1
        else:
          break
  
  for key, dic_data in result_dic.items():
    print(key)
    for key, value in dic_data.items():
      print("|" + str(key) + ": " + str(value) + "| ", end="")
    print("")
    
   
def manual_run(file_path, number_of_activites, support, ratio, pruning_threshold, data_piece_index):
  if number_of_activites <= 0:
    return

  warnings.filterwarnings("ignore")
  
  folder_name = file_path + "/Data_" + str(number_of_activites)
  ending_File_string = str(number_of_activites) + "_Sup_"+ str(support) + "_Ratio_" + str(ratio) + "_Pruning_" + str(pruning_threshold) + "_Data_" + str(data_piece_index)

  
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
     
  folder_name += "/Sup_" + str(support) + "_Ratio_" + str(ratio) + "_Pruning_" + str(pruning_threshold)
  
  # Check if the folder with param already exists
  if not os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
      
  cut = find_best_cut(folder_name + "/logP_" + ending_File_string, folder_name + "/logM_" + ending_File_string,support,ratio, pruning_threshold)
  
   
if __name__ == '__main__':
  random.seed(random_seed)
  
  generate_data(relative_path,0.2,0.2,0,True)
  
  
  # get_labeled_data_cut_type_distribution(relative_path,0.2,0.2,0)

  # manual_run('GNN_partitioning/GNN_Data', 4, 0.2, 0.6, 0, 5)

  # log_statistic(relative_path + "data1")

