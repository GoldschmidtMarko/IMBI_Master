

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

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_best_cut_with_cut_type
from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import artificial_start_end
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import view_dfg
from pm4py import save_vis_dfg
from pm4py.objects.process_tree.obj import ProcessTree, Operator
import warnings
from pm4py.objects.log.exporter.xes.variants.etree_xes_exp import Parameters as Export_Parameter
from GNN_partitioning.GNN_Model_Generation.gnn_models import generate_adjacency_matrix_from_log
import json

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
  
  
def save_log_as_dfg(log, file_name):
  start_act_cur_dfg = start_activities_get.get_start_activities(log)
  end_act_cur_dfg = end_activities_get.get_end_activities(log)
  cur_dfg = dfg_inst.apply(log)
  save_vis_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg, file_name + ".png")
  
  
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
    operatorChance = [0.5,0.1,0.2,0.2]
    
    new_operator = random.choices(operatorList,operatorChance)[0]
  else:
    new_operator = operator
  
  new_tree_node = ProcessTree(children=[processTree1, processTree2],operator=new_operator)
  processTree1.parent = new_tree_node
  processTree2.parent = new_tree_node
  
  process_tree_list.append(new_tree_node)
  
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

def get_percentage_of_noise():
  return 0
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
  
  # save_log_as_dfg(log, "log1")
  
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
  from statistics import mean
  # Calculate the average trace length
  trace_lengths = []

  for trace in log:
      trace_length = len(trace)
      trace_lengths.append(trace_length)

  average_trace_length = mean(trace_lengths)
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
  
  
  # Export the event log to a XES file
  # parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  # apply(log, file_name + ".xes", parameters=parameter)
  # tree_exporter.apply(tree, file_name + ".ptml")

  
def save_log(log, file_name):
  # Export the event log to a XES file
  parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  apply(log, file_name + ".xes", parameters=parameter)

def save_data(file_name, adj_matrix_P, unique_node,cut_type, sup, ratio, datapiece, partitionA, partitionB, score, unique_activity_count_P, size_par, seed_P, tree):
  with open(file_name + ".txt", "w") as file:
    file.write("# unique_node | unique_activity_count_P | adj_matrix_P | cut_type | sup | ratio | size_par | dataitem | partitionA | partitionB | score | random_seed_P | tree " + "\n")
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
    # adj_matrix_P
    for row in adj_matrix_P:
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
    file.write(str(seed_P) + "\n")
    # tree
    file.write(str(tree) + "\n")
   
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
     
def generate_data_piece_for_cut_type(file_path, number_of_activites, support, data_piece_index, cut_type):
  if number_of_activites <= 0:
    return

  warnings.filterwarnings("ignore")
  
  folder_name = file_path + "/" + cut_type
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
  
  folder_name += "/Data_" + str(number_of_activites)
  ending_File_string = str(number_of_activites) + "_Sup_"+ str(support) + "_" +  str(data_piece_index)

  
  
  # Check if the folder already exists
  if os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
     
  folder_name += "/Sup_" + str(support)
  
  # Check if the folder with param already exists
  if not os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name, exist_ok=True)
      
  # Check if files already exist
  checking_files_names = [folder_name + "/treeP_" + ending_File_string,
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

    logM = logP.__deepcopy__()
    
    result, cut = find_best_cut_type(logP, logM, support, 0, cut_type)

    if result == True:
      # result, cut = find_best_cut_type(logP,logM,support,ratio, cut_type)
      try:
        save_tree(process_tree_P, folder_name + "/treeP_" + ending_File_string + ".json")
      except:
        print("Error saving treeP:")
        print(process_tree_P)
        continue

      # save_log(logP, folder_name + "/logP_" + ending_File_string)

      unique_nodeList, matrix_P = generate_adjacency_matrix_from_log(logP)
      
      logP_art = artificial_start_end(logP.__deepcopy__())
      
      activity_count_P = get_activity_count(logP_art)
      unique_activity_count_P = get_activity_count_list_from_unique_list(activity_count_P, unique_nodeList)
      
      size_par = len(logP) / len(logM)

      save_data(folder_name + "/Data_" + str(data_piece_index), matrix_P,unique_nodeList,cut_type,support, 0,data_piece_index,cut[0][0],cut[0][1], cut[4],unique_activity_count_P,size_par, random_seed_P, process_tree_P)
      return 1

      
    return 0

 
if __name__ == '__main__':
  print("Error, not supported to run as main")
  

