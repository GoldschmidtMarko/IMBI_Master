
import random
import os
import sys
import time

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

import numpy as np



random_start_seed = 1996

relative_path = root_path + "/GNN_partitioning/GNN_Data"
 
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
    
  cut_types = ["exc", "seq", "par", "loop"]
    
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
  
def get_category_balance(dic_categories):
  max_value = 0
  smallest_ratio = 1
  for key, value in dic_categories.items():
    if key != "Total":
      if value > max_value:
        max_value = value

  for key, value in dic_categories.items():
      if key != "Total":
        ratio = value / max_value
        if ratio < smallest_ratio:
          smallest_ratio = ratio
  return round(smallest_ratio,2)
      
  
  
def print_distribution_dictionary(result_dic):
  for key, dic_data in result_dic.items():
    
    print("Cut: " + key)
    total = 0
    for key, value in dic_data.items():
      balance = get_category_balance(value)
      if balance <= 0.7:
        balance_string = "\033[91m" + str(balance) + "\033[0m"
      else:
        balance_string = str(balance)
      print("|" + str(key) + ": " + str(value["Total"]) + " B: " + balance_string + "| ", end="")
      total += value["Total"]
    print("")
    print("Total: " + str(total))
    print("")
    
if __name__ == '__main__':
  random.seed(random_start_seed)
  cur_time = time.time()

  print()
  result_distribution = get_distribution_dictionary(relative_path, 0.2, 0.2)
  print_distribution_dictionary(result_distribution)
  
  
  print("Runtime: " + str(time.time() - cur_time))
  

