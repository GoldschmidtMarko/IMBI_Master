

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

def find_best_cut(path_lopP, path_logM, sup, ratio):
  
  parameter = {Import_Parameter.SHOW_PROGRESS_BAR: False}
  log = xes_importer.apply(path_lopP + ".xes", parameters=parameter)
  logM = xes_importer.apply(path_logM + ".xes", parameters=parameter)

  best_cut = get_best_cut(log,logM,sup,ratio)
  return best_cut
  
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
    random_items = random.sample(generated_ProcessTrees, 2)
    processTree1 = random_items[0]
    processTree2 = random_items[1]
    generated_ProcessTrees.remove(processTree1)
    generated_ProcessTrees.remove(processTree2)
    
    operatorList = [Operator.SEQUENCE,Operator.LOOP,Operator.PARALLEL,Operator.XOR]
    new_operator = random.choice(operatorList)
    
    generated_ProcessTrees.append(ProcessTree(children=[processTree1, processTree2],operator=new_operator))
    
    
  # print("Generated Tree")
  # generated_ProcessTrees[0]._print_tree()
  return generated_ProcessTrees[0] 

def generate_log_from_process_tree(activites_list, noise_factor = 0):
  process_tree = generate_random_process_tree(activites_list)
  log = pm4py.play_out(process_tree)
  number_noise_traces = int(noise_factor * len(log))
  
  number_avg_trace_length = int(get_avg_trace_length_from_log(log))
  number_avg_trace_length_deviation = int((number_avg_trace_length / 8))
  
  for i in range (number_noise_traces):
    trace = Trace()
    generated_trace = generate_trace(activites_list, trace_length=get_number_depending_on_deviation(number_avg_trace_length, number_avg_trace_length_deviation))
    for event in generated_trace:
      event1 = {"concept:name": str(event)}
      trace.append(event1)

    # Add the trace to the event log
    log.append(trace)
    
  return log
  
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
  
def save_cut(cut, file_name):
  with open(file_name + ".txt", "w") as file:
    file.write("# partitionA | partitionB | cut_type" + "\n")
    outputString = ""
    for act in cut[0][0]:
      outputString += str(act) + " "
    file.write(outputString + "\n")
    
    outputString = ""
    for act in cut[0][1]:
      outputString += str(act) + " "
    file.write(outputString + "\n")
    
    file.write(str(cut[1]) + "\n")
    

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
      
  
def generate_data_piece(file_path, number_of_activites, support, ratio, data_piece_index):
  if number_of_activites <= 0:
    return

  warnings.filterwarnings("ignore")
  
  folder_name = file_path + "/Data_" + str(number_of_activites)
  ending_File_string = str(number_of_activites) + "_Sup_"+ str(support) + "_Ratio_" + str(ratio) + "_Data_" + str(data_piece_index)
  
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
  checking_files_names = [folder_name + "/logP_" + ending_File_string,
                          folder_name + "/logM_" + ending_File_string,
                          folder_name + "/Cut_" + ending_File_string]
  for files in checking_files_names:
    if os.path.exists(files):
        # Delete the file
        shutil.rmtree(files)
  
  activity_list = generate_activity_name_list(number_of_activites,
                                              random.randint(5,8))
  
  logP = generate_log_from_process_tree(activity_list, 0)
  
  number_activity_intersections = random.randint(0,number_of_activites)
  
  activity_list_near = get_partial_activity_names(activity_list,number_of_activites,
                                                  random.randint(5,8),
                                                  number_activity_intersections)
  
  logM = generate_log_from_process_tree(activity_list_near, 0)
  

  save_log(logP, folder_name + "/logP_" + ending_File_string)
  save_log(logM, folder_name + "/logM_" + ending_File_string)
  
  cut = find_best_cut(folder_name + "/logP_" + ending_File_string, folder_name + "/logM_" + ending_File_string,support,ratio)
  
  save_cut(cut,folder_name + "/Cut_" + ending_File_string)
  
def generate_data_piece_star_function(args):
    return generate_data_piece(*args)
    
def generate_data(file_path):
  
  
  if os.path.exists(file_path):
    shutil.rmtree(file_path)
  
  # Check if the folder already exists
  if not os.path.exists(file_path):
      # Create the folder
      os.makedirs(file_path)
      
  max_number_activites = 6
      
  num_processors_available = multiprocessing.cpu_count()
  print("Number of available processors:", num_processors_available)
  num_processors = max(1,round(num_processors_available/2))
  
  number_of_data_pieces_per_variation = 2
  
  # def generate_data_piece(file_path, number_of_activites, support, ratio):
  list_data_pool = [(file_path, number_activ, sup, ratio, data_piece_index)
              for number_activ in range(max_number_activites + 1)
              for sup in np.round(np.arange(0,1.2,0.2),1)
              for ratio in np.round(np.arange(0,1.2,0.2),1)
              for data_piece_index in range(1,number_of_data_pieces_per_variation + 1)]

  # Create a pool of workers
  with multiprocessing.Pool(num_processors) as pool:
    list(tqdm(pool.imap(generate_data_piece_star_function, list_data_pool), total=len(list_data_pool)))


    
    
if __name__ == '__main__':
  random.seed(random_seed)
  
  generate_data(relative_path)
  # generate_data_piece(relative_path,4,0,0)
  

# log_statistic(relative_path + "data1")

