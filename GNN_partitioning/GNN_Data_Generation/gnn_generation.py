
import random
import string
import os
import multiprocessing
from pm4py.objects.log.importer.xes.importer import apply as import_apply
from pm4py.objects.log.exporter.xes.exporter import apply
import sys
import time
import datetime

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

import numpy as np
from tqdm import tqdm
import warnings
from pm4py.objects.log.exporter.xes.variants.etree_xes_exp import Parameters as Export_Parameter
import psutil

from GNN_partitioning.GNN_Data_Generation.uni.gnn_generation import generate_data_piece_for_cut_type as uni_generate_data_piece_for_cut_type
from GNN_partitioning.GNN_Data_Generation.bi.gnn_generation import generate_data_piece_for_cut_type as bi_generate_data_piece_for_cut_type
import json



random_start_seed = 1996

relative_path = root_path + "/GNN_partitioning/GNN_Data"



def get_log(file_name):
  warnings.filterwarnings("ignore")
  # Export the event log to a XES file
  parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  log = import_apply(file_name + ".xes", parameters=parameter)
  return log

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
  
def get_available_disk_space(path):
    disk_usage = psutil.disk_usage(path)
    available_space = disk_usage.free / (1024 ** 3)  # Convert bytes to gigabytes
    return available_space
          
def generate_data_piece_for_cut_type(file_path, number_of_activites, support, ratio, data_piece_index, cut_type, consider_ratio):
  if consider_ratio == False:
    return uni_generate_data_piece_for_cut_type(file_path,number_of_activites,support,data_piece_index,cut_type)
  else:
    return bi_generate_data_piece_for_cut_type(file_path,number_of_activites,support,ratio,data_piece_index,cut_type)

def generate_data_piece_star_function_cut_Type(args):
    return generate_data_piece_for_cut_type(*args)
       
def check_if_data_piece_exists(file_path, number_of_activites, support, data_piece_index, cut_type):
  folder_name = file_path + "/" + cut_type
  folder_name += "/Data_" + str(number_of_activites)
  folder_name += "/Sup_" + str(support)
  
  ending_File_string = str(number_of_activites) + "_Sup_"+ str(support) + "_" + str(data_piece_index)
  
  test_file = folder_name + "/treeP_" + ending_File_string + ".pmt"
  
  # Check if the folder already exists
  if os.path.exists(test_file):
    return True
  else:
    return False
  
  
def get_distribution_dictionary(file_path, sup_step, ratio_step = 0.0):
  print("Getting labeled data cut type distribution in: " + file_path)
  result_dic = dict()
  
  # Setup sup list
  if sup_step == 0:
    sup_list = [0]
  else:
    sup_list = np.round(np.arange(0,1 + sup_step,sup_step),1)
    
  # Setup ratio list
  if ratio_step == 0.0:
    ratio_list = [0]
    consider_ratio = False
  else:
    ratio_list = np.round(np.arange(0,1 + ratio_step,ratio_step),1)
    consider_ratio = True
    print("Ratio list: " + str(ratio_list))
    
  cut_types = ["exc", "exc_tau", "seq", "par", "loop_tau", "loop"]
    
  def get_txt_files(directory):
    txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    return txt_files
    
  for cut in cut_types:
    current_path_cut = file_path + "/" + cut
    if os.path.exists(current_path_cut):
      result_dic[cut] = dict()
      for integer in range(2,30):
        current_path_cut_Data = current_path_cut + "/Data_" + str(integer)
        if os.path.exists(current_path_cut_Data):
          if integer not in result_dic[cut]:
            result_dic[cut][integer] = {"Total": 0}
          for sup in sup_list:
            if consider_ratio:
              for ratio in ratio_list:
                sup_ratio_string = "Sup_" + str(sup) + "_Ratio_" + str(ratio)
                current_path_variant = current_path_cut_Data + "/" + sup_ratio_string
                if os.path.exists(current_path_variant):
                  txt_files = get_txt_files(current_path_variant)

                  result_dic[cut][integer]["Total"] += len(txt_files)
                  result_dic[cut][integer][sup_ratio_string] = len(txt_files)
            else:
              sup_ratio_string = "Sup_" + str(sup)
              current_path_variant = current_path_cut_Data + "/" + sup_ratio_string
              if os.path.exists(current_path_variant):
                txt_files = get_txt_files(current_path_variant)
                result_dic[cut][integer]["Total"] += len(txt_files)
                result_dic[cut][integer] = {sup_ratio_string: len(txt_files)}
  return result_dic

def extract_sup_and_ratio(input_string):
  parts = input_string.split('_')

  if len(parts) == 4 and parts[0] == 'Sup' and parts[2] == 'Ratio':
      try:
          sup = float(parts[1])
          ratio = float(parts[3])
          return sup, ratio
      except ValueError:
          # Handle conversion errors
          return None, None

  return None, None

def get_deviating_categories_per_graph_per_category(result_dic, graph_node, cut_types, workitems):
  res_categories = []
  for cut_type in cut_types:
    max_v = 0
    for key, value in result_dic[cut_type][graph_node].items():
      if key != "Total":
        if value > max_v:
          max_v = value
    for key, value in result_dic[cut_type][graph_node].items():
      if key != "Total":
        sup, ratio = extract_sup_and_ratio(key)
        if value < max_v * 0.9:
          res_categories.append({"Graph_Node_Size": graph_node, "Cut_Type": cut_type, "Support": sup, "Ratio": ratio, "Work": max_v - value + workitems})
        else:
          res_categories.append({"Graph_Node_Size": graph_node, "Cut_Type": cut_type, "Support": sup, "Ratio": ratio, "Work": workitems})
        
  return res_categories


def get_number_work_per_graph_size_per_category(result_dic, graph_node_size, cut_types, workitems):
  res_work = []
  for graph_node in graph_node_size:
    res_work = res_work + get_deviating_categories_per_graph_per_category(result_dic, graph_node, cut_types, workitems)
  return res_work
    
def generate_data(file_path, sup_step, ratio_step, unique_identifier, number_new_data_instances_per_category, list_grap_node_sizes, use_parallel = True):
  
  # Delete the folder if it already exists
  # if os.path.exists(file_path):
  #   shutil.rmtree(file_path)
  
  print("Generating GNN Data in: " + file_path)
  print("Current time: " + str(datetime.datetime.now()))
  
  # Check if the folder already exists
  if not os.path.exists(file_path):
      # Create the folder
      os.makedirs(file_path)
      

  num_processors_available = multiprocessing.cpu_count()
  print("Number of available processors:", num_processors_available)
  if num_processors_available > 20:
    num_processors = max(1,round(num_processors_available))
  else:
    num_processors = max(1,round(num_processors_available/2))
  
  
  # Setup sup list
  if sup_step == 0:
    sup_list = [0]
  else:
    sup_list_temp = np.round(np.arange(0,1 + sup_step,sup_step),1)
    sup_list = [float(x) for x in sup_list_temp]
    
  # Setup ratio list
  if ratio_step == 0.0:
    ratio_list = [0]
    consider_ratio = False
  else:
    ratio_list = np.round(np.arange(0,1 + ratio_step,ratio_step),1)
    consider_ratio = True
    print("Ratio list: " + str(ratio_list))
    

  cut_types = ["exc", "seq", "par", "loop"]
  # cut_types = ["seq"]
  

  list_data_pool = []
  balancing_work = True
  if balancing_work:
    result_distribution = get_distribution_dictionary(relative_path, 0.2, 0.2)
    worklist = get_number_work_per_graph_size_per_category(result_distribution, list_grap_node_sizes, cut_types, number_new_data_instances_per_category)
    for work in worklist:
      for data_piece_name in range(1,work["Work"] + 1):
        list_data_pool.append((file_path, work["Graph_Node_Size"], work["Support"], work["Ratio"], f"{unique_identifier}{data_piece_name}", work["Cut_Type"], consider_ratio))
  else:
    list_data_pool = [(file_path, number_activ, sup, ratio, f"{unique_identifier}{data_piece_name}", cut_type, consider_ratio)
                for number_activ in list_grap_node_sizes
                for sup in sup_list
                for ratio in ratio_list
                for data_piece_name in range(1,number_new_data_instances_per_category + 1)
                for cut_type in cut_types]
  
  
  # Check if the folder already exists
  if len(list_data_pool) > 0:
    if check_if_data_piece_exists(file_path, list_data_pool[0][1], list_data_pool[0][2], list_data_pool[0][3], list_data_pool[0][4]):
        # Error
        print("Error, data already exists for unique identifier: " + str(unique_identifier))
        sys.exit()
      
  
  space_Approx = 4000/(5400 * 1000)
  
  if get_available_disk_space(file_path) < space_Approx * len(list_data_pool):
    print("Error, not enough space available. Approx. space needed: " + str(space_Approx * len(list_data_pool)) + " GB")
    print("Available space: " + str(get_available_disk_space(file_path)) + " GB")
    sys.exit()

  sum_results = 0
  if use_parallel:
    print("Number of used processors:", num_processors)
    # Create a pool of workers
    with multiprocessing.Pool(num_processors) as pool:
      results = list(tqdm(pool.imap(generate_data_piece_star_function_cut_Type, list_data_pool), total=len(list_data_pool)))
      
    sum_results = sum(results)
  else:
    for it in list_data_pool:
      print("Running: " + str(it))
      sum_results += generate_data_piece_star_function_cut_Type(it)

  print("Number of generated data pieces: " + str(sum_results) + " of " + str(len(list_data_pool)))
  print("Percentage of generated data pieces: " + str(round((sum_results / len(list_data_pool) * 100),2)) + "%")
   
def get_labeled_data_cut_type_distribution(file_path, sup_step, ratio_step = 0.0):
  print("Getting labeled data cut type distribution in: " + file_path)
  result_dic = dict()
  
  # Setup sup list
  if sup_step == 0:
    sup_list = [0]
  else:
    sup_list = np.round(np.arange(0,1 + sup_step,sup_step),1)
    
  # Setup ratio list
  if ratio_step == 0.0:
    ratio_list = [0]
    consider_ratio = False
  else:
    ratio_list = np.round(np.arange(0,1 + ratio_step,ratio_step),1)
    consider_ratio = True
    print("Ratio list: " + str(ratio_list))
    
  cut_types = ["exc", "exc_tau", "seq", "par", "loop_tau", "loop"]
    
  def get_txt_files(directory):
    txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    return txt_files
    
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
            if consider_ratio:
              for ratio in ratio_list:
                sup_ratio_string = "Sup_" + str(sup) + "_Ratio_" + str(ratio)
                current_path_variant = current_path_cut_Data + "/" + sup_ratio_string
                if os.path.exists(current_path_variant):
                  txt_files = get_txt_files(current_path_variant)
                  result_dic[cut][integer] += len(txt_files)
            else:
              sup_ratio_string = "Sup_" + str(sup)
              current_path_variant = current_path_cut_Data + "/" + sup_ratio_string
              if os.path.exists(current_path_variant):
                txt_files = get_txt_files(current_path_variant)
                result_dic[cut][integer] += len(txt_files)
  
  for key, dic_data in result_dic.items():
    if key != "loop_tau" and key != "exc_tau":
      print("Cut: " + key)
      total = 0
      for key, value in dic_data.items():
        print("|" + str(key) + ": " + str(value) + "| ", end="")
        total += value
      print("")
      print("Total: " + str(total))
      print("")
    
def get_input_arguments(list_inputs):
  if len(list_inputs) != 4:
    print("Error, wrong number of input arguments, expected 4, got: " + str(len(list_inputs)))
    print("Expected input: unique_indentifier number_new_data_instances_per_category list_grap_node_sizes")
    sys.exit()
  
  unique_indentifier = list_inputs[1] + "_"
  number_new_data_instances_per_category = int(list_inputs[2])
  list_grap_node_sizes = map(int, list_inputs[3].strip('[]').split(','))
  
  print("unique_indentifier: " + str(unique_indentifier))
  print("number_new_data_instances_per_category: " + str(number_new_data_instances_per_category))
  print("list_grap_node_sizes: " + str(list_grap_node_sizes))
  
  return unique_indentifier, number_new_data_instances_per_category, list_grap_node_sizes
   
   
   
def log_runtime_of_imbi():
  overall_runtime = time.time()
  from GNN_partitioning.GNN_Data_Generation.bi import gnn_generation as gnn_generation_bi
  from local_pm4py.algo.discovery.inductive import algorithm as inductive_miner
  from local_pm4py.algo.analysis import custom_enum
  from local_pm4py.algo.discovery.inductive.variants.im_bi.data_structures.subtree_plain import get_best_cut_with_cut_type
  
  ratio = 1
  support = 0.2 
  
  max_activites = 18
  
  for number_of_activites in range(2,max_activites + 1):
    print("Number of activites: " + str(number_of_activites))
    # print("Generating data...")
    activity_list = gnn_generation_bi.generate_activity_name_list(number_of_activites,
                                                  random.randint(5,8))
      
    process_tree_P = gnn_generation_bi.generate_random_process_tree_for_cut_type(activity_list, "seq")
    
    random_seed_P = random.randint(100000, 999999)
    logP = gnn_generation_bi.generate_log_from_process_tree_for_cut_type(activity_list, process_tree_P, random_seed_P)
    
    percentage_is_subset = 0.8
    activity_list_near = gnn_generation_bi.get_partial_activity_names(activity_list, percentage_is_subset, min_percentage_subset = 0.5)
    process_tree_M = gnn_generation_bi.generate_mutated_process_tree_from_process_tree(activity_list_near, process_tree_P, mutation_rate=0.5)

    random_seed_M = random.randint(100000, 999999)
    logM = gnn_generation_bi.generate_log_from_process_tree_for_cut_type(activity_list_near, process_tree_M, random_seed_M)
    
    # print("Running IMBI...")
    cur_time = time.time()
    # net, initial_marking, final_marking = inductive_miner.apply_bi(logP,logM, variant= inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, use_gnn=False)
    get_best_cut_with_cut_type(logP,logM,sup=support, ratio=ratio, cost_Variant=custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE)
    
    # print("Runtime of IMBI: " + str(time.time() - cur_time))


  print("Runtime overall: " + str(time.time() - overall_runtime))
  
def run_generate_data():
  unique_indentifier, number_new_data_instances_per_category, list_grap_node_sizes = get_input_arguments(sys.argv)
  # unique_indentifier, number_new_data_instances_per_category, list_grap_node_sizes = "test", 10, [7]
  generate_data(relative_path, 0.2, 0.2, unique_indentifier, number_new_data_instances_per_category, list_grap_node_sizes, True)
  
  
   
if __name__ == '__main__':
  random.seed(random_start_seed)
  cur_time = time.time()

  print()
  run_generate_data()
  # log_runtime_of_imbi()
  # get_labeled_data_cut_type_distribution(relative_path, 0.2, 0.2)
  
  print("Runtime: " + str(time.time() - cur_time))
  

