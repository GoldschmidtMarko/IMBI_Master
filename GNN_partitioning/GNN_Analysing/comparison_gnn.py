import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

import random
from local_pm4py.algo.discovery.inductive import algorithm as inductive_miner
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import warnings
import json
import datetime
from pm4py.objects.log.importer.xes.variants.iterparse import Parameters as Export_Parameter
from pm4py.objects.log.importer.xes.importer import apply as apply_import_xes
from local_pm4py.algo.analysis import custom_enum
from pm4py import view_petri_net
from pm4py import save_vis_petri_net
from pm4py import play_out
import pandas as pd
from local_pm4py.algo.analysis import Optimzation_Goals
from pm4py.objects.process_tree.obj import ProcessTree, Operator
import matplotlib.pyplot as plt
import numpy as np
import math
import fpdf
import time
from PIL import Image
import multiprocessing
from tqdm import tqdm
import pm4py
from pebble import ProcessPool
from concurrent.futures import TimeoutError

def visualize_cuts(fileName):
  pdf = fpdf.FPDF(format='letter') #pdf format
  pdf.add_page() #create new page
  pdf.set_font("Arial", size=8) # font and textsize

  depth = 0
  currentIteration = 1
  file_path = "imbi_cuts/depth_" + str(depth) + "_It_" + str(currentIteration)
  
  folder_name = "imbi_cuts"
  # Check if the folder already exists
  if not os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name)
    
  # for depth
  while(os.path.isfile(file_path + ".png")):
    # for iteration
    while(os.path.isfile(file_path + ".png")):
      with open(file_path + ".txt") as f:
        pdf.cell(100, 4, txt="Cut: " + str(depth + 1) + " it: " + str(currentIteration), ln=1, align="C")
        pdf.cell(1000, 4, txt="cut | type | cost_p | cost_m | cost_ratio | fitP", ln=1, align="L")
        pdf.cell(1000, 4, txt="", ln=1, align="L")
        lines = f.readlines()
        readLines = 0
        for line in lines:
          if readLines == 0:
            outputLine = line.replace(" ", " | ")
          else:
            outputLine = line
          pdf.cell(1000, 4, txt=outputLine, ln=1, align="L")
          readLines += 1
          if readLines == 3:
            readLines = 0
            pdf.cell(1000, 4, txt="", ln=1, align="L")
      img = Image.open(file_path + ".png")
      width,height = img.size
      # print(width, height)
      pdf.image(file_path + ".png",w=min(150,width/3),h=min(150,height/3))
      pdf.add_page()
      currentIteration += 1
      file_path = "imbi_cuts/depth_" + str(depth) + "_It_" + str(currentIteration)
      
    depth += 1
    currentIteration = 1
    file_path = "imbi_cuts/depth_" + str(depth) + "_It_" + str(currentIteration)
  pdf.output(fileName + ".pdf")

def visualize_petriNet(df, miner, logPName, logMName = "", use_gnn = False):
  df_temp = df[df["miner"] == miner]
  df_temp = df_temp[df_temp["logM_Name"] == logMName]
  df_temp = df_temp[df_temp["logP_Name"] == logPName]
  df_temp = df_temp[df_temp["use_gnn"] == use_gnn]
  for net, im, fm, sup, ratio in zip(df_temp.net, df_temp.im, df_temp.fm, df_temp.im_bi_sup, df_temp.im_bi_ratio):
    print("Displaying: " + str(miner) + " " + str(logPName) + " " + str(logMName) + " Sup: " + str(sup) + " Ratio: " + str(ratio) + " GNN:" + str(use_gnn))
    view_petri_net(net, im, fm)

def visualize_petriNet_Advanced(df, miner, logPName, logMName = "", sup = 0, ratio = 0, use_gnn = False):
  df_temp = df[df["miner"] == miner]
  df_temp = df_temp[df_temp["logM_Name"] == logMName]
  df_temp = df_temp[df_temp["logP_Name"] == logPName]
  df_temp = df_temp[df_temp["use_gnn"] == use_gnn]
  df_temp = df_temp[df_temp["im_bi_sup"] == sup]
  df_temp = df_temp[df_temp["im_bi_ratio"] == ratio]
  for net, im, fm, sup, ratio in zip(df_temp.net, df_temp.im, df_temp.fm, df_temp.im_bi_sup, df_temp.im_bi_ratio):
    print("Displaying: " + str(miner) + " " + str(logPName) + " " + str(logMName) + " Sup: " + str(sup) + " Ratio: " + str(ratio) + " GNN:" + str(use_gnn))
    view_petri_net(net, im, fm)


def save_petriNet(fileName, df, miner, logPName, logMName = "", use_gnn = False):
  df_temp = df[df["miner"] == miner]
  df_temp = df_temp[df_temp["logM_Name"] == logMName]
  df_temp = df_temp[df_temp["logP_Name"] == logPName]
  df_temp = df_temp[df_temp["use_gnn"] == use_gnn]
  for net, im, fm, sup, ratio in zip(df_temp.net, df_temp.im, df_temp.fm, df_temp.im_bi_sup, df_temp.im_bi_ratio):
    print("Saving: " + str(miner) + " " + str(logPName) + " " + str(logMName) + " Sup: " + str(sup) + " Ratio: " + str(ratio) + " GNN:" + str(use_gnn))
    save_vis_petri_net(net, im, fm,file_path=fileName)

def save_petriNet_Advanced(fileName, df, miner, logPName, logMName = "", sup = 0, ratio = 0, use_gnn = False):
  df_temp = df[df["miner"] == miner]
  df_temp = df_temp[df_temp["logM_Name"] == logMName]
  df_temp = df_temp[df_temp["logP_Name"] == logPName]
  df_temp = df_temp[df_temp["use_gnn"] == use_gnn]
  df_temp = df_temp[df_temp["im_bi_sup"] == sup]
  df_temp = df_temp[df_temp["im_bi_ratio"] == ratio]
  for net, im, fm, sup, ratio in zip(df_temp.net, df_temp.im, df_temp.fm, df_temp.im_bi_sup, df_temp.im_bi_ratio):
    print("Displaying: " + str(miner) + " " + str(logPName) + " " + str(logMName) + " Sup: " + str(sup) + " Ratio: " + str(ratio) + " GNN:" + str(use_gnn))
    save_vis_petri_net(net, im, fm,file_path=fileName)

def visualize_All_petriNet(df, miner):
  df_temp = df[df["miner"] == miner]
  for logPName, logMName in zip(df_temp.logP_Name, df_temp.logM_Name):
    visualize_petriNet(df,miner,logPName,logMName)

def runDoubleLogEvaluation(df,cut_Type, log,logM, name,net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, use_gnn = False):
  mes = Optimzation_Goals.apply_petri_silent(log,logM,net,im,fm)

  df = pd.concat([df, pd.DataFrame.from_records([{
    "miner" : name,
    "cut_type" : cut_Type,
    "logP_Name": logPName[:logPName.rfind(".")],
    "logM_Name": "",
    "imf_noise_thr" : imf_noiseThreshold,
    "hm_depen_thr" : hm_dependency_threshold,
    "im_bi_sup" : im_bi_sup,
    "im_bi_ratio" : im_bi_ratio,
    "use_gnn" : use_gnn,
    # "acc_logs": mes['acc'],
    "fitP" : mes['fitP'],
    # "fitM" : mes['fitM'],
    # "f1_fit_logs": mes['F1'],
    "precision" : mes['precision'],
    "net": net,
    "im" : im,
    "fm" : fm
  }])])
  return df

def add_Model_To_Database(df,cut_Type, log,logM, name,net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, use_gnn = False):
  df = runDoubleLogEvaluation(df,cut_Type, log,logM, name,net, im, fm, logPName,logMName, imf_noiseThreshold, hm_dependency_threshold,im_bi_sup, im_bi_ratio, use_gnn=use_gnn)
  return df

def getF1Value(df, miner, logPName, logMName, support, ratio, use_gnn):
  dftemp = df[df["miner"] == miner]
  dftemp = dftemp[dftemp["logP_Name"] == logPName[:logPName.rfind(".")]]
  dftemp = dftemp[dftemp["logM_Name"] == logMName[:logPName.rfind(".")]]
  dftemp = dftemp[dftemp["im_bi_sup"] == support]
  dftemp = dftemp[dftemp["im_bi_ratio"] == ratio]
  dftemp = dftemp[dftemp["use_gnn"] == use_gnn]
  if len(dftemp.index) > 1:
    raise Exception("Error, too many rows.")
  return dftemp["f1_fit_logs"].iloc[0]

def get_log(file_name):
  warnings.filterwarnings("ignore")
  # Export the event log to a XES file
  parameter = {Export_Parameter.SHOW_PROGRESS_BAR: False}
  log = apply_import_xes(file_name, parameters=parameter)
  return log

def applyMinerToLogForGNN(df, cut_Type, logP, logPName, logM, logMName, consider_ratio = False, noiseThreshold = 0.0, dependency_threshold=0.0, support = 0, ratio = 0, use_gnn = False):
  
  if consider_ratio == False:
    logM = logP.__deepcopy__()
    logMName = ""
  
  
  cur_time = time.time()
    
  # imbi_ali
  cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant,use_gnn=use_gnn)

  
  df = add_Model_To_Database(df=df,cut_Type=cut_Type, log=logP, logM=logM,net=net,im=im,fm=fm,name="IMbi_ali",logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  time_in_bi = time.time() - cur_time

  time_measurement = time.time() - cur_time
  cur_time = time.time()
  
  save_cuts = False
  if save_cuts == True:
    fileName_cuts_ali = "cuts_IMbi_ali_sup_" + str(support) + "_ratio_" + str(ratio) + "_logP_" + logPName[:logPName.rfind(".")] + "_logM_" + logMName[:logMName.rfind(".")]
    visualize_cuts(fileName_cuts_ali)
  
  # print("Load: " + str(round(time_loading,2)) + " | Imbi: " + str(round(time_in_bi,2)) + " | Measure: " + str(round(time_measurement,2)))
  
  return df

def setupYTickList(minValue, step):
  res = []
  cur = 1
  while cur > minValue:
    res.append(cur)
    cur -= step
  res.append(cur)
  return res
    
def displayDoubleLog(df, saveFig = False):
  df_grouped = df.groupby(by=["logP_Name",	"logM_Name", "im_bi_sup", "im_bi_ratio"], group_keys=True).apply(lambda x : x)
  numberOfPlotPerRow = 4
  rows = math.ceil(float(len(df_grouped.index.unique()))/numberOfPlotPerRow)
  cols = min(len(df_grouped.index.unique()),numberOfPlotPerRow)

  fig, axs = plt.subplots(rows, cols, figsize=(15 * (cols / numberOfPlotPerRow), 4 * rows), squeeze=False)
  fig.tight_layout(pad=10.0)
  cur_Row = 0
  cur_Col = 0
  
  for logGroup in df_grouped.index.unique():
    df_log_grouped = df_grouped.loc[logGroup]

    axs[cur_Row,cur_Col].set_title("LogP: " + logGroup[0] + " LogM: " + logGroup[1] + "\n" + "Sup: " + str(df_log_grouped.im_bi_sup[0]) + " ratio: " + str(df_log_grouped.im_bi_ratio[0]) )
    axs[cur_Row,cur_Col].set_xlabel("Miners")
    j = 0
    xTickLabel = []
    idx = []
    minValue = 0
    for miner, use_gnn, precision, acc, fitP, fitM, f1_fit in zip(df_log_grouped.miner, df_log_grouped.use_gnn, df_log_grouped.precision, df_log_grouped.acc_logs, df_log_grouped.fitP, df_log_grouped.fitM, df_log_grouped.f1_fit_logs):
      minValue = min([minValue, acc, fitP, fitM, f1_fit])
      axs[cur_Row,cur_Col].bar(j,precision, color="r", label="precision")
      axs[cur_Row,cur_Col].bar(j+1,acc, color="black", label="acc")
      axs[cur_Row,cur_Col].bar(j+2,fitP, color="g", label="fitP")
      axs[cur_Row,cur_Col].bar(j+3,fitM, color="b", label="fitM")
      axs[cur_Row,cur_Col].bar(j+4,f1_fit, color="orange", label="f1_fit")
      xTickLabel.append(miner + "\nGNN: " + str(use_gnn))
      idx.append(j + 2.5)
      j += 6
      
    
    axs[cur_Row,cur_Col].set_yticks(setupYTickList(minValue, 0.25))
    axs[cur_Row,cur_Col].set_xticks(idx)
    axs[cur_Row,cur_Col].set_xticklabels(xTickLabel)
    axs[cur_Row,cur_Col].legend(loc='center left', ncols=1, labels=["precP","acc", "fitP", "fitM", "f1_fit"], bbox_to_anchor=(1, 0.5))
    cur_Col += 1
    
    if cur_Col == numberOfPlotPerRow:
      cur_Row += 1
      cur_Col = 0
  
  plt.show()

  if saveFig:
    fig.savefig("plot" + ".pdf")
    
def create_df():
  columns = ["miner", "logP_Name", "logM_Name","imf_noise_thr","hm_depen_thr","im_bi_sup","im_bi_ratio", "use_gnn","acc_logs", "fitP", "fitM", "f1_fit_logs", "precision", "net", "im", "fm"]

  df = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=None)
  df['use_gnn'] = df['use_gnn'].astype(bool)
  return df

def convert_tree_path_to_text_path(input_str):
  # Split the input string on "tree"
  parts1 = input_str.split("tree")
  
  # Get the part after "tree" and split it on the last underscore
  
  parts3 = parts1[1].rsplit('_', 2)
  
  # Extract the numeric part
  numeric_part = parts3[2].split('.')[0]
  
  # Construct the output string
  # output_str = parts1[0] + "Data_" + parts3[1] + "_" + numeric_part + ".txt"
  output_str = parts1[0] + "Data_" + parts3[1] + "_" + numeric_part + ".txt"
  return output_str

def read_data_from_text_path(file_path):
  data = {}
  if os.path.exists(file_path):
      # Open the text file in read mode
      with open(file_path, 'r') as file:
          # Iterate over each line in the file
          matrix_P_arrayList = []
          matrix_M_arrayList = []
          state = -1
          for line in file:
                if state == -1:
                    state += 1
                    continue
                elif state == 0 :
                    data["Labels"] = line.split(" ")[:-1]
                    state += 1
                    continue
                elif state == 1:
                    data["Activity_count_P"] = np.array(line.split(" ")[:-1]).astype(int)
                    state += 1
                    continue
                elif state == 2:
                    data["Activity_count_M"] = np.array(line.split(" ")[:-1]).astype(int)
                    state += 1
                    continue
                elif state == 3 and line == "\n":
                    data["Adjacency_matrix_P"] = np.vstack(matrix_P_arrayList)
                    state += 1
                    continue
                elif state == 4 and line == "\n":
                    data["Adjacency_matrix_M"] = np.vstack(matrix_M_arrayList)
                    state += 1
                    continue   
                elif state == 5:
                    data["Cut_type"] = line[:-1]
                    state += 1
                    continue 
                elif state == 6:
                    data["Support"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 7:
                    data["Ratio"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 8:
                    data["Size_par"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 9:
                    data["Dataitem"] = line[:-1]
                    state += 1
                    continue 
                elif state == 10:
                    data["PartitionA"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 11:
                    data["PartitionB"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 12:
                    data["Score"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 13:
                    data["random_seed_P"] = int(line[:-1])
                    state += 1
                    continue 
                elif state == 14:
                    data["random_seed_M"] = int(line[:-1])
                    state += 1
                    continue 
                
                if state == 3:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_P_arrayList.append(np_array)
                if state == 4:
                    lineList = line.split(" ")[:-1]
                    np_array = np.array(lineList, dtype=int)
                    matrix_M_arrayList.append(np_array)
  return data


def get_data_paths(use_synthetic, dataPath, consider_ratio = False, max_node_size = 100, num_data_per_category = 1, min_number_node_size = 0):

  def is_string_present(string, list):
    if list == None:
      return False
    if len(list) == 0:
      return False
    for item in list:
      if string in item[0] or string in item[1]:
        return True
    return False
  
  def get_last_Data_number(string):
    parts = string.split('Data_')
    if len(parts) > 1:
        rest = parts[-1]
        number_part = rest.split('/')[0]
        if number_part.isdigit():
            return int(number_part)
        number_part = rest.split('\\')[0]
        if number_part.isdigit():
            return int(number_part)
    return 101
  
  if use_synthetic:
    cut_types = ["par", "exc","loop", "seq"]
    
    foundFiles = set()
    
    # cut_types = ["loop", "exc","par", "seq"]
    pathFiles = {key: [] for key in cut_types}
    currentPath = dataPath
    if os.path.exists(dataPath):
      for cut_type in cut_types:
        currentPath = os.path.join(dataPath, cut_type)
        continue_running = True
        dirs_and_files = []
        if os.path.exists(currentPath) and continue_running:
          for root, _ , files in os.walk(currentPath):
            dirs_and_files.append((root, _, files))
          
          dirs_and_files.sort(reverse=True)
          for root, _, files in dirs_and_files:
            if min_number_node_size <= get_last_Data_number(root) <= max_node_size:
              for file in files:
                if not continue_running:
                  break
                
                if file.endswith(".json"):
                  found_file = os.path.join(root, file)
                  if "treeP" in found_file:
                    file_M = found_file.replace("treeP", "treeM")
                    file_P = found_file
                  else:
                    file_M = found_file
                    file_P = found_file.replace("treeM", "treeP")
                  data_txt_file = convert_tree_path_to_text_path(found_file)
                  
                  if os.path.exists(data_txt_file):   
                    if file_P not in foundFiles and file_M not in foundFiles:
                      data = read_data_from_text_path(data_txt_file)
                      pathFiles[cut_type].append((file_P, file_M, data))
                  if len(pathFiles[cut_type]) >= num_data_per_category:
                    continue_running = False
                    break

  else:
    pathFiles = []
    currentPath = dataPath
    if os.path.exists(dataPath):
      for root, _, files in os.walk(currentPath):
          for file in files:
            if file.endswith(".xes"):  # Filter for text files
              found_file = os.path.join(root, file)
              if consider_ratio:
                if "lp" in found_file:
                  file_M = found_file.replace("lp", "lm")
                  file_P = found_file
                else:
                  file_M = found_file
                  file_P = found_file.replace("lm", "lp")
              else:
                file_M = found_file
                file_P = found_file
                
              if not is_string_present(file_P, pathFiles) and not is_string_present(file_M, pathFiles):
                pathFiles.append((file_P, file_M))
  if use_synthetic:
    for key in pathFiles:
      pathFiles[key] = sorted(pathFiles[key], key=lambda x: x[0], reverse=True)

    for cut_type, dataList in pathFiles.items():
      print("Data")
      print("Cut type: " + cut_type + " Number of files: " + str(len(dataList)))

    return pathFiles
  else:
    print("Data")
    print("Number of files: " + str(len(pathFiles)))
    return pathFiles
  
  
def run_evaluation_delta_synthetic(df, dataPath, num_data_per_category, using_gnn, max_node_size = 100, parallel = False):
  pathFiles = get_data_paths(True, dataPath, max_node_size, num_data_per_category)

  for cut_type, dataList in pathFiles.items():
    print("Running: " + cut_type)
    if parallel:
      num_processors_available = multiprocessing.cpu_count()
      print("Number of available processors:", num_processors_available)
      if num_processors_available > 20:
        num_processors = max(1,round(num_processors_available))
      else:
        num_processors = max(1,round(num_processors_available/2))
      print("Number of used processors:", num_processors)


      batch_size = math.ceil(len(dataList) / num_processors)
      # batch_size = 1
      input_data = []
      offset = 0
      for i in range(num_processors):
          batch_data = dataList[offset:offset + batch_size]
          df_temp = create_df()
          input_data.append((df_temp, cut_type, batch_data, [False, True]))
          offset += batch_size
          
      TIMEOUT_SECONDS = 60 * batch_size
      # TIMEOUT_SECONDS = 60 * 10
      print("Timeout: " + str(TIMEOUT_SECONDS))
      
      pool_res = []
      '''
      pool_res = tqdm(pool.imap(run_evaluation_category_star, input_data),total=len(input_data))
      '''
      number_timesouts = 0
      futures = []
      progress_bar = None
      
      # with ThreadPoolExecutor(max_workers=num_processors) as executor:
      with ProcessPool(max_workers=num_processors) as pool:
        
        futures = [pool.schedule(run_evaluation_category_star, args=(one,), timeout=TIMEOUT_SECONDS) for one in input_data]
        # futures = [executor.submit(run_with_timeout, run_evaluation_category_star, (one,), TIMEOUT_SECONDS) for one in input_data]
        
        # Create a tqdm progress bar to track the progress of futures
        progress_bar = tqdm(total=len(futures), desc="Processing", unit="future")

        
      for future in futures:
        try:
          result = future.result()
          pool_res.append(result)
        except TimeoutError:
          number_timesouts += 1
        except Exception as exc:
          import traceback
          traceback.print_exc()  # Print the full traceback
          print('%r generated an exception: %s' % (future, exc))
          number_timesouts += 1
        progress_bar.update(1)

      # Close the progress bar after the loop finishes
      progress_bar.close()

      for result in pool_res:
          # Process individual evaluation result
          df = pd.concat([df, result])
            
      print("Timeout percentage: " + str(number_timesouts/len(input_data)))
    else: 
      df = run_evaluation_category(df, cut_type, dataList, using_gnn)
  
  return df

def run_evaluation_trace_variants_synthetic(dataPath, num_data_per_category, max_node_size = 100, parallel = False):
  columns = ["miner","cut_type", "logP_Name", "logM_Name","trace_variants_P","trace_variants_M"]
  df = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=None)
  df['trace_variants_P'] = df['trace_variants_P'].astype(float)
  df['trace_variants_M'] = df['trace_variants_M'].astype(float)
  
  pathFiles = get_data_paths(True, dataPath, max_node_size, num_data_per_category)
  
  for cut_type, dataList in pathFiles.items():
    print("Running: " + cut_type)
    if parallel:
      num_processors_available = multiprocessing.cpu_count()
      print("Number of available processors:", num_processors_available)
      if num_processors_available > 20:
        num_processors = max(1,round(num_processors_available))
      else:
        num_processors = max(1,round(num_processors_available/2))
      print("Number of used processors:", num_processors)

      batch_size = math.ceil(len(dataList) / num_processors)
      input_data = []
      offset = 0
      for i in range(num_processors):
          batch_data = dataList[offset:offset + batch_size]
          df_temp = create_df()
          input_data.append((df_temp, cut_type, batch_data))
          offset += batch_size
 
      pool_res = None
      with multiprocessing.Pool(num_processors) as pool:
          pool_res = tqdm(pool.imap(run_evaluation_category_trace_variants_star, input_data),total=len(input_data))
          
          for result in pool_res:
              # Process individual evaluation result
              df = pd.concat([df, result])
    else: 
      for data in dataList:
        df = run_evaluation_category_trace_variants(df, cut_type, data)
  
  return df

def run_evaluation_trace_variants_real(dataPath, parallel = False):
  columns = ["miner","cut_type", "logP_Name", "logM_Name","trace_variants_P","trace_variants_M"]
  df = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=None)
  df['trace_variants_P'] = df['trace_variants_P'].astype(float)
  df['trace_variants_M'] = df['trace_variants_M'].astype(float)
  
  pathFiles = get_data_paths(False, dataPath)
  
  if parallel:
    num_processors_available = multiprocessing.cpu_count()
    print("Number of available processors:", num_processors_available)
    if num_processors_available > 20:
      num_processors = max(1,round(num_processors_available))
    else:
      num_processors = max(1,round(num_processors_available/2))
    print("Number of used processors:", num_processors)

    input_data = []
    iterator = 0
    for data in pathFiles:
      iterator += 1
      df_temp = create_df()
      input_data.append((df_temp, iterator, data))
        
    pool_res = None
    with multiprocessing.Pool(num_processors) as pool:
        pool_res = tqdm(pool.imap(run_evaluation_category_trace_variants_star, input_data),total=len(input_data))
        
        for result in pool_res:
            # Process individual evaluation result
            df = pd.concat([df, result])
  else: 
    iterator = 0
    for data in pathFiles:
      iterator += 1
      df = run_evaluation_category_trace_variants(df, iterator, data)

  return df


def run_evaluation_delta_real(df, dataPath, sup_list, ratio_list, using_gnn, consider_ratio = False, parallel = False):
  pathFiles = get_data_paths(False, dataPath, consider_ratio=consider_ratio)

  enriched_pathFiles = []
  for dataList in pathFiles:
    for sup in sup_list:
      for ratio in ratio_list:
        enriched_pathFiles.append(
          {"logP": dataList[0],
           "logM": dataList[1],
           "Support": sup,
           "Ratio": ratio})


  if parallel:
    num_processors_available = multiprocessing.cpu_count()
    print("Number of available processors:", num_processors_available)
    if num_processors_available > 20:
      num_processors = max(1,round(num_processors_available))
    else:
      num_processors = max(1,round(num_processors_available/2))
    print("Number of used processors:", num_processors)

    input_data = []
    iterator = 0
    for data in enriched_pathFiles:
      iterator += 1
      for gnn in using_gnn:
        df_temp = create_df()
        input_data.append((df_temp, iterator, [data], [gnn], consider_ratio))
    
    pool_res = None
    '''
    pool_res = tqdm(pool.imap(run_evaluation_category_star, input_data),total=len(input_data))
    '''
    with multiprocessing.Pool(num_processors) as pool:
      pool_res = tqdm(pool.imap(run_evaluation_category_star, input_data),total=len(input_data))

      for result in pool_res:
          # Process individual evaluation result
          df = pd.concat([df, result])
     
  else: 
    iterator = 0
    for data in enriched_pathFiles:
      iterator += 1
      df = run_evaluation_category(df, iterator, [data],using_gnn,consider_ratio)

  return df

def run_evaluation_category_trace_variants_star(args):
    return run_evaluation_category_trace_variants(*args)    

def run_evaluation_category_trace_variants(df, cut_type, data):
  logP_path = data[0]
  logM_path = data[1]
  logP = get_log(logP_path)
  logM = get_log(logM_path)
  variantsP = pm4py.get_variants(logP)
  variantsM = pm4py.get_variants(logM)
  
  df = pd.concat([df, pd.DataFrame.from_records([{
    "miner" : logP_path,
    "cut_type" : cut_type,
    "logP_Name": logP_path[:logP_path.rfind(".")],
    "logM_Name": logM_path[:logM_path.rfind(".")],
    "trace_variants_P": len(variantsP),
    "trace_variants_M": len(variantsM)
  }])])

  return df

def run_with_timeout(func, args, timeout):
  result = None
  exception = None

  def worker():
      nonlocal result, exception
      try:
          result = func(*args)
      except Exception as e:
          exception = e

  thread = threading.Thread(target=worker)
  thread.start()
  thread.join(timeout)

  if thread.is_alive():
      thread.join()  # Ensure the thread terminates
      raise TimeoutError
  elif exception is not None:
      raise exception
  else:
      return result
  
def run_evaluation_category_star(args):
  return run_evaluation_category(*args)    

def get_log_from_tree(tree, seed):
  random.seed(seed)
  logP = play_out(tree)
  return logP

def load_tree(file_name):
  def get_Operator_from_string(operator_string):
    if operator_string == "->":
        return Operator.SEQUENCE
    elif operator_string == "X":
        return Operator.XOR
    elif operator_string == "+":
        return Operator.PARALLEL
    elif operator_string == "*":
        return Operator.LOOP
    return None
 
  def deserialize_tree(serialized_node):
      if serialized_node is None:
          return None
      
      node = ProcessTree()
      node.label = serialized_node["label"]
      node.operator = get_Operator_from_string(serialized_node["operand"])
      node.children = [deserialize_tree(child) for child in serialized_node["children"]]
      return node

  with open(file_name, "r") as file:
      serialized_tree = json.load(file)

  return deserialize_tree(serialized_tree)
  
def run_evaluation_category(df, cut_type, dataList, using_gnn, consider_ratio):
  for data in dataList:
    if len(data) == 4:
      support = data["Support"]
      ratio = data["Ratio"]

      logP = get_log(data["logP"])
      logP_name = data["logP"]
      
      logM = get_log(data["logM"])
      logM_name = data["logM"]
      
    else:
      support = data[1]["Support"]
      ratio = data[1]["Ratio"]
      
      tree_path = data[0]
      treeP = load_tree(tree_path)
      logP = get_log_from_tree(treeP, data[1]["random_seed_P"])
      logP_name = tree_path
      
      if consider_ratio:
        raise Exception("Not implemented")
        treeMpath = data[0]
        treeM = load_tree(tree_path)
        logM = get_log_from_tree(treeP, data[1]["random_seed_P"])
        logM_name = tree_path
    
    for use_gnn in using_gnn:
      try:
        df = applyMinerToLogForGNN(df, cut_type, logP, logP_name, logM, logM_name, consider_ratio, 0.2, 0.99, support, ratio,use_gnn)
      except Exception as exc:
        import traceback
        traceback.print_exc()  # Print the full traceback
        print('%r generated an exception: %s' % (exc))
  return df

def get_measurement_delta(df, columnList, column_feature):
  
  columns_to_keep = columnList + column_feature +  ['use_gnn']

  # Trim the DataFrame to include only specific columns
  df = df[columns_to_keep]
  df = df.fillna("")
  # common_prefix = "C:\\Users\\Marko\\Desktop\\GIt\\IMBI_Master\\GNN_partitioning\\GNN_Data"
  # df['logP_Name'] = df['logP_Name'].str.replace(common_prefix, '', regex=False)
  
  # pd.set_option("display.max_colwidth", 200)
  # print(df.head(10))
  # print(columnList)
  
  def custom_agg(group):
    precision_diff = group.loc[group['use_gnn'] == False, 'precision'].mean() - \
                    group.loc[group['use_gnn'] == True, 'precision'].mean()

    fitP_diff = group.loc[group['use_gnn'] == False, 'fitP'].mean() - \
                group.loc[group['use_gnn'] == True, 'fitP'].mean()

    return pd.Series({'precision': precision_diff, 'fitP': fitP_diff})

  # Apply the custom aggregation function and reset the index
  df_measurement = df.groupby(columnList).apply(custom_agg).reset_index()
  
  # print()
  # pd.set_option("display.max_colwidth", 10)
  # print(df_measurement.head(10))
  return df_measurement

def show_duplicates(df):
  filtered_rows = df[
    (df["logP_Name"] == "GNN_partitioning\GNN_Data\par\Data_8\Sup_1.0_Ratio_1.0_Pruning_0\logP_8_Sup_1.0_Ratio_1.0_Pruning_0_Data_11") &
    (df["logM_Name"] == "GNN_partitioning\GNN_Data\par\Data_8\Sup_1.0_Ratio_1.0_Pruning_0\logM_8_Sup_1.0_Ratio_1.0_Pruning_0_Data_11") &
    (df["cut_type"] == "par")
  ]

  print(filtered_rows)  
  print("Number of duplicates: " + str(len(filtered_rows)))  

def visualize_measurement(df_measurement, column_feature, use_synthetic, title = "", column_prefix = "", file_name = None):
  y_min = df_measurement[column_feature].min().min()
  y_max = df_measurement[column_feature].max().max()
  if use_synthetic:
    df_grouped = df_measurement.groupby("cut_type")
    number_columns = 4
  else:
    df_grouped = df_measurement.groupby("logP_Name")
    number_columns = df_grouped.ngroups
    
  import matplotlib.font_manager as fm
  # print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
  custom_font_bold = fm.FontProperties(family='Arial', size=16, weight='bold')
  custom_font = fm.FontProperties(family='Arial', size=16)
    
  fig, axes = plt.subplots(nrows=1, ncols=number_columns, figsize=(16, 10))
  fig.suptitle(title, fontproperties=custom_font_bold)
  


  for i, (cut_type, group) in enumerate(df_grouped):
    ax = axes[i]  # Select the specific axis for this subplot
    if use_synthetic:
      ax.set_title("Data folder: " + str(cut_type) + "\nDatasize: " + str(len(group)))  # Set title for the subplot
    else:
      ax.set_title("Data P: " + str(os.path.basename(group.logP_Name.iloc[0])) + "\nData M: " + str(os.path.basename(group.logM_Name.iloc[0])) + "\nDatasize: " + str(len(group)), fontproperties=custom_font)  # Set title for the subplot
    ax.set_ylabel("Delta percentage", fontproperties=custom_font)  # Set ylabel for the subplot
    
    x_ticks = []
    x_labels = []
    
    for j, col in enumerate(column_feature):
      group.boxplot(column=col,positions=[j],widths=0.5, ax=ax, patch_artist=True,    # Fill boxes with color
                        showfliers=False,     # Hide outliers
                        boxprops={'facecolor': 'lightblue', 'linewidth': 2},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        medianprops={'color': 'orange', 'linewidth': 2})
      
      # Modify x-axis tick labels
      x_ticks.append(j)
      x_labels.append(column_prefix + col)
      
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, fontproperties=custom_font)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # TODO MAKE SMALLER AROUND MIN AND MAX
    y_interval = max(0.2, (y_max - y_min) / 10)
    y_ticks = np.arange(np.floor(y_min / y_interval) * y_interval,
                                np.ceil(y_max / y_interval) * y_interval + y_interval, y_interval)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_ticks[0], y_ticks[-1])  # Expand limits slightly
    
    # ax.set_ylim(-1, 1)
    # ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

  plt.tight_layout()
  if file_name != None:
    plt.savefig(file_name)
  else:
    plt.show()
      
   
def validate_data_results(df, columns_to_check, interval):
  for index, row in df.iterrows():
      for column in columns_to_check:
          value = row[column]
          if not (interval[0] <= value <= interval[1]):
              print(f"Row {index}: Value {value} in column '{column}' is not between -1 and 1.")

def sanity_check_input(synthetic_path, real_path):
  if synthetic_path != None and real_path != None:
    print("Error, only one path can be specified.")
    return -1
  if synthetic_path == None and real_path == None:
    print("Error, no path specified.")
    return -1
  if synthetic_path != None and not os.path.exists(synthetic_path):
    print("Error, synthetic path does not exist.")
    return -1
  if real_path != None and not os.path.exists(real_path):
    print("Error, real path does not exist.")
    return -1
  if synthetic_path != None and os.path.isdir(synthetic_path):
    return 0
  if real_path != None and os.path.isdir(real_path):
    return 1
  

def get_dataframe_trace_Variants(data_path_csv, synthetic_path = None, real_path = None):
  progress = sanity_check_input(synthetic_path, real_path)
  if progress == -1:
    sys.exit()
  if progress == 0:
    csv_filename = os.path.join(data_path_csv, "output_trace_variant_synthetic.csv")
  else:
    csv_filename = os.path.join(data_path_csv, "output_trace_variant_real.csv")
    
  if not os.path.exists(csv_filename):
    if progress == 0:
      df = run_evaluation_trace_variants_synthetic(synthetic_path, 50, max_node_size=8, parallel = False)
      df.to_csv(csv_filename, index=False)
    elif progress == 1:
      df = run_evaluation_trace_variants_real(real_path, parallel = True)
      df.to_csv(csv_filename, index=False)
  else:
    df = pd.read_csv(csv_filename)
  return df
      
def get_dataframe_delta(data_path_csv, synthetic_path = None, real_path = None):
  progress = sanity_check_input(synthetic_path, real_path)
  if progress == -1:
    sys.exit()
  if progress == 0:
    csv_filename = os.path.join(data_path_csv, "output_delta_synthetic.csv")
  else:
    csv_filename = os.path.join(data_path_csv, "output_delta_real.csv")
    
  if not os.path.exists(csv_filename):
    df = create_df()
    
    if progress == 0:
      using_gnn = [False, True]
      num_data_per_category = 50
      df = run_evaluation_delta_synthetic(df, synthetic_path, num_data_per_category, using_gnn, max_node_size=8, parallel = True)
      # Save the DataFrame to a CSV file
      df.to_csv(csv_filename, index=False)
    elif progress == 1:
      using_gnn = [False, True]
      sup_list = [0, 0.2, 0.4, 0.6]
      ratio_list = [0, 0.5, 1]
      consider_ratio = False
      if len(ratio_list) > 1:
        consider_ratio = True
      df = run_evaluation_delta_real(df, real_path, sup_list, ratio_list, using_gnn,consider_ratio=consider_ratio, parallel = True)
      # Save the DataFrame to a CSV file
      df.to_csv(csv_filename, index=False)
      
    else:
      sys.exit()
      
  else:
    df = pd.read_csv(csv_filename)
    
  return df
    
def run_comparison():
  time_start = time.time()
  
  quasi_identifiers = ["logP_Name",	"logM_Name", "cut_type"]
    
  # data_path_real = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  data_path_real = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  data_path_synthetic = os.path.join(root_path, "GNN_partitioning", "GNN_Data")
  data_path_csv = os.path.join(root_path, "GNN_partitioning", "GNN_Analysing", "Results")
  
  if not os.path.exists(data_path_csv):
    os.makedirs(data_path_csv)
  
  delta_measurement = True
  if delta_measurement:
    use_synthetic = False
    column_feature = ["precision","fitP"]
    
    title = 'Data Delta Measurement\nDelta = (No GNN) - (GNN)'
    column_prefix = "Δ "
    if use_synthetic:
      df = get_dataframe_delta(data_path_csv, synthetic_path=data_path_synthetic) 
      df_measurement = get_measurement_delta(df, quasi_identifiers, column_feature)
      
      output_path = os.path.join(data_path_csv, "df_delta_measurement_synthetic.png")
      visualize_measurement(df_measurement, column_feature, use_synthetic, title, column_prefix, output_path)
    else:
      df = get_dataframe_delta(data_path_csv, real_path=data_path_real) 
      df_measurement = get_measurement_delta(df, quasi_identifiers, column_feature)
      
      output_path = os.path.join(data_path_csv, "df_delta_measurement_real.png")
      visualize_measurement(df_measurement, column_feature, use_synthetic, title, column_prefix, output_path)
      
  trace_variant_measurement = False
  if trace_variant_measurement:
    use_synthetic = True
    column_feature = ["trace_variants_P", "trace_variants_M"]
    
    itle = 'Data Trace Variants Measurement'
    column_prefix = "# "
    if use_synthetic:
      df = get_dataframe_trace_Variants(data_path_csv, synthetic_path=data_path_synthetic)
      
      output_path = os.path.join(data_path_csv, "df_measurement_trace_variants_synthetic.png")
      visualize_measurement(df, column_feature, use_synthetic, title, column_prefix, output_path)
    else:
      df = get_dataframe_trace_Variants(data_path_csv, real_path=data_path_real)
      
      output_path = os.path.join(data_path_csv, "df_measurement_trace_variants_real.png")
      visualize_measurement(df, column_feature, use_synthetic, title, column_prefix, output_path)
      
  print("Time elapsed: " + str(time.time() - time_start) + " seconds")
  
  
def manual_run():
  df = create_df()
  
  found_problem_path = "/loop/Data_6/Sup_0.6/treeP_6_Sup_0.6_Data_run1_79.json"
  data_path_synthetic = os.path.join(root_path, "GNN_partitioning", "GNN_Data")
  pathFiles = get_data_paths(True, data_path_synthetic, 1000, 100)
  
  
  data = None
  for file in pathFiles["loop"]:
    if found_problem_path in file[0]:
      data = file
      break

  if data == None:
    print("Error, tree path not found.")
    return
  
  treeP = load_tree(data[0])
  print(treeP)
  logP = get_log_from_tree(treeP, data[1]["random_seed_P"])
  df = applyMinerToLogForGNN(df, "loop", logP, data[0], 0.2, 0.99, 0.6, 0,True)
  print(df)
  
def get_data_from_logName(tree_path):
  if os.path.exists(tree_path):
    data_txt_file = convert_tree_path_to_text_path(tree_path)
    if os.path.exists(data_txt_file):     
      data = read_data_from_text_path(data_txt_file)
      return (tree_path, data)
    else:
      print("Error, data path does not exist.")
      print(data_txt_file)
  else:
    print("Error, tree path does not exist.")
    print(tree_path)
  return None
  
  
def run_show_data():
  data_path_real = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  data_path_synthetic = os.path.join(root_path, "GNN_partitioning", "GNN_Data")
  data_path_csv = os.path.join(root_path, "GNN_partitioning", "GNN_Analysing", "Results")
  
  
  os.system('clear')
  input_synthetic = input("Enter Datatype (1 - synthethic, 2 -  real): ")
  os.system('clear')
  print("DataList loaded:")
  if input_synthetic == "1":
    dataList = get_data_paths(True, data_path_synthetic,100, 100, 7)
  elif input_synthetic == "2":
    dataList = None
  else:
    print("Error, invalid input.")
    return
  
  print()
  while(True):
    input_synthetic = input("Options: \n 1 - exit \n 2 - get random data \n 3 - get specific data (TODO)\n")
    os.system('clear')
    if input_synthetic == "1":
      return
    if input_synthetic == "2":
      
      tree_path = random.choice(dataList[random.choice(list(dataList))])[0]
      
      while(True):
        common_prefix = "/rwthfs/rz/cluster/work/xm454523/IMBI_Master/GNN_partitioning/GNN_Data"
        print("Current Data:")
        print("LogName: " + tree_path[len(common_prefix):])
        print("Cut type: " + tree_path)
        data_txt_file = convert_tree_path_to_text_path(tree_path)
        data = None
        if os.path.exists(data_txt_file):     
          data = read_data_from_text_path(data_txt_file)
    
        print("Support: " + str(data["Support"]))
        print("Data_node_size: " + str(len(data["PartitionA"]) + len(data["PartitionB"])))
        print()
        input_option = input("Options: \n 1 - back \n 2 - show petrinets \n 3 - show processtree \n 4 - show dfg\n")
        os.system('clear')
        if input_option == "1":
          break
        elif input_option == "2":
          print("Mining Petrinet:")
          data = get_data_from_logName(tree_path)
          
          tree_path = data[0]
          tree = load_tree(tree_path)
          log = get_log_from_tree(tree, data[1]["random_seed_P"])
          support = data[1]["Support"]
          ratio = data[1]["Ratio"]
          cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE
          use_gnn = False
          net, im, fm = inductive_miner.apply_bi(log,log, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(log)/len(log), cost_Variant=cost_Variant,use_gnn=use_gnn)
          use_gnn = True
          net_gnn, im_gnn, fm_gnn = inductive_miner.apply_bi(log,log, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(log)/len(log), cost_Variant=cost_Variant,use_gnn=use_gnn)
          save_vis_petri_net(net,im,fm,os.path.join(data_path_csv, "petri1.png"))
          save_vis_petri_net(net_gnn,im_gnn,fm_gnn,os.path.join(data_path_csv, "petri2.png"))
          
          import matplotlib.image as mpimg
          # Load the first PNG image
          image1 = mpimg.imread(os.path.join(data_path_csv, "petri1.png"))

          # Load the second PNG image
          image2 = mpimg.imread(os.path.join(data_path_csv, "petri2.png"))

          # Calculate the maximum width between the two images
          max_width = max(image1.shape[1], image2.shape[1])

          # Calculate the width difference for centering
          width_diff1 = (max_width - image1.shape[1]) // 2
          width_diff2 = (max_width - image2.shape[1]) // 2

          # Create a figure and two subplots
          fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

          # Plot the first image on the first subplot
          ax1.imshow(np.pad(image1, ((0, 0), (width_diff1, width_diff1), (0, 0)), mode='constant'))
          ax1.axis('off')  # Turn off axes
          ax1.set_title("PetriNet use_gnn: False", fontsize=12)  # Add a title above the first image

          # Plot the second image on the second subplot
          ax2.imshow(np.pad(image2, ((0, 0), (width_diff2, width_diff2), (0, 0)), mode='constant'))
          ax2.axis('off')  # Turn off axes
          ax2.set_title("PetriNet use_gnn: True", fontsize=12)  # Add a title above the second image

          # Adjust spacing between subplots
          plt.subplots_adjust(hspace=0.0)  # You can adjust the vertical spacing as needed

          # Save the combined image
          plt.savefig(os.path.join(data_path_csv, "file.png"), bbox_inches='tight')  # bbox_inches='tight' to remove extra whitespace

          
          
        elif input_option == "3":
          import networkx as nx
          def create_graph_from_json(data):
            G = nx.DiGraph()
            
            def add_nodes_and_edges(node_data, parent=None):
              G.add_node(node_data, label=node_data.label)
              if parent is not None:
                  G.add_edge(parent, node_data)
              
              children = node_data.children
              for child_data in children:
                  add_nodes_and_edges(child_data, node_data)
            
            add_nodes_and_edges(data)
            
            return G
          
          data = get_data_from_logName(tree_path)
          tree_path = data[0]
          tree = load_tree(tree_path)
          # Create a NetworkX graph from the JSON data
          graph = create_graph_from_json(tree)

          # Define the layout (optional)
          layout = nx.spring_layout(graph)
          
          def custom_layout(G):
            pos = {}
            level_height = 1.0  # Vertical distance between levels
            sibling_distance = 5.0  # Horizontal distance between siblings

            def position_tree(node, x, y, dx):
                children = list(G.successors(node))
                if len(children) > 0:
                    dx /= len(children)
                    next_x = x - (len(children) - 1) * dx / 2
                    for child in children:
                        pos[child] = (next_x, y - level_height)
                        next_x += dx
                        position_tree(child, pos[child][0], pos[child][1], dx)

            # Set the root node position
            root = [node for node, in_degree in G.in_degree() if in_degree == 0][0]
            pos[root] = (0, 0)

            # Recursively calculate positions for the rest of the nodes
            position_tree(root, 0, 0, sibling_distance)

            return pos
          
          
          # Use the "dot" layout from Graphviz
          pos = custom_layout(graph)
          
          # Create a plot of the graph with labels
          plt.figure(figsize=(8, 6))
          nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='skyblue')
          # Draw edges
          nx.draw_networkx_edges(graph, pos, edge_color='gray')
          
          node_labels = {node: node.label if node.label is not None else node.operator for node in graph.nodes()}
          nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, font_color='black', verticalalignment='bottom')




          # Save the graph as an image (e.g., PNG)
          plt.savefig(os.path.join(data_path_csv, "file.png"))
          
          
          print("Saved processTree.\n")
          
        elif input_option == "4":
          def save_dfg(log, file_name):
            from pm4py.statistics.end_activities.log import get as end_activities_get
            from pm4py.statistics.start_activities.log import get as start_activities_get
            from pm4py.algo.discovery.dfg.variants import native as dfg_inst
            from pm4py import save_vis_dfg
            parameters = {}
            start_act_cur_dfg = start_activities_get.get_start_activities(log, parameters=parameters)
            end_act_cur_dfg = end_activities_get.get_end_activities(log, parameters=parameters)
            cur_dfg = dfg_inst.apply(log, parameters=parameters)
            save_vis_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg, file_name + ".png")
          
          data = get_data_from_logName(tree_path)
          
          tree_path = data[0]
          tree = load_tree(tree_path)
          log = get_log_from_tree(tree, data[2]["random_seed_P"])
          save_dfg(log,os.path.join(data_path_csv, "file"))
          print("Saved dfg.\n")
        
def compare_gnn_time(logP,logM,support,ratio):
  start_time = time.time()
  use_gnn = False
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE,use_gnn=use_gnn)
  time_no_gnn = time.time() - start_time
  
  use_gnn = True
  start_time = time.time()
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE,use_gnn=use_gnn)
  time_gnn = time.time() - start_time
  return (time_no_gnn, time_gnn)
  
          
  
def run_time_comparison_artificial():
  data_path_synthetic = os.path.join(root_path, "GNN_partitioning", "GNN_Data")
  dataDict = get_data_paths(True, data_path_synthetic,True,100, 100, 2)
  
  total_time_no_gnn = 0
  total_time_gnn = 0
  runs = 1
  totalRuns = 0
  for dataList in dataDict.values():
    totalRuns += len(dataList)
    
  for dataList in dataDict.values():
    for dataPair in dataList:
      print("Run: " + str(runs) + "/" + str(totalRuns))
      runs += 1
      data = dataPair[2]
      treeP = load_tree(dataPair[0])
      treeM = load_tree(dataPair[1])
      logP = get_log_from_tree(treeP, data["random_seed_P"])
      logM = get_log_from_tree(treeM, data["random_seed_M"])
      time_no_gnn, time_gnn = compare_gnn_time(logP,logM,data["Support"],data["Ratio"])
      total_time_no_gnn += time_no_gnn
      total_time_gnn += time_gnn
      
  print("Total time no gnn: " + str(total_time_no_gnn))
  print("Total time gnn: " + str(total_time_gnn))
  


  
def run_runtime_comparison():
  data_path_real = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  data_path_real = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  pathFiles = get_data_paths(False, data_path_real, consider_ratio=True)
  for files in pathFiles:
    logP = get_log(files[0])
    logM = get_log(files[1])
    support = 0.3
    ratio = 1
    time_no_gnn, time_gnn = compare_gnn_time(logP,logM,support,ratio)
    print("Time no gnn: " + str(time_no_gnn))
    print("Time gnn: " + str(time_gnn))
  
  
if __name__ == '__main__':
  run_time_comparison_artificial()
  # run_runtime_comparison()
  # run_comparison()
  # run_show_data()
  # manual_run()
  

    

  

  

  
    
    
    
    
