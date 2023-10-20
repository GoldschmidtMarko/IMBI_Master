import os
import sys
import numpy as np

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)


list_grap_node_sizes = [2,3,4,5,6,7,8]
number_new_data_instances_per_category = 30
relative_path = root_path + "/GNN_partitioning/GNN_Data"
cut_types = ["exc", "seq", "par", "loop"]

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
    print(max_v)
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

result_distribution = get_distribution_dictionary(relative_path, 0.2, 0.2)

work = get_number_work_per_graph_size_per_category(result_distribution, list_grap_node_sizes, cut_types, number_new_data_instances_per_category)

print(work)
print(len(work))