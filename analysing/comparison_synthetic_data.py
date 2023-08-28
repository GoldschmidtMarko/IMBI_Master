import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/Marko/Desktop/GIt/IMBI_Master')
from local_pm4py.algo.discovery.inductive import algorithm as inductive_miner
import warnings
from pm4py.objects.log.importer.xes.variants.iterparse import Parameters as Export_Parameter
from pm4py.objects.log.importer.xes.importer import apply as apply_import_xes
from local_pm4py.algo.analysis import custom_enum
from pm4py import view_petri_net
from pm4py import save_vis_petri_net
import pandas as pd
from local_pm4py.algo.analysis import Optimzation_Goals
import matplotlib.pyplot as plt
import numpy as np
import math
import fpdf
from PIL import Image
import multiprocessing
from tqdm import tqdm
import os

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

def runDoubleLogEvaluation(df,cut_Type, log,logM, name,net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, pruning_threshold = 0, use_gnn = False):
  mes = Optimzation_Goals.apply_petri_silent(log,logM,net,im,fm)

  df = pd.concat([df, pd.DataFrame.from_records([{
    "miner" : name,
    "cut_type" : cut_Type,
    "logP_Name": logPName[:logPName.rfind(".")],
    "logM_Name": logMName[:logMName.rfind(".")],
    "imf_noise_thr" : imf_noiseThreshold,
    "hm_depen_thr" : hm_dependency_threshold,
    "im_bi_sup" : im_bi_sup,
    "im_bi_ratio" : im_bi_ratio,
    "pruning_threshold" : pruning_threshold,
    "use_gnn" : use_gnn,
    "acc_logs": mes['acc'],
    "fitP" : mes['fitP'],
    "fitM" : mes['fitM'],
    "f1_fit_logs": mes['F1'],
    "precision" : mes['precision'],
    "net": net,
    "im" : im,
    "fm" : fm
  }])])
  return df

def add_Model_To_Database(df,cut_Type, log,logM, name,net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, pruning_threshold = 0, use_gnn = False):
  df = runDoubleLogEvaluation(df,cut_Type, log,logM, name,net, im, fm, logPName,logMName, imf_noiseThreshold, hm_dependency_threshold,im_bi_sup, im_bi_ratio, pruning_threshold, use_gnn=use_gnn)
  return df

def getF1Value(df, miner, logPName, logMName, support, ratio, pruning_threshold, use_gnn):
  dftemp = df[df["miner"] == miner]
  dftemp = dftemp[dftemp["logP_Name"] == logPName[:logPName.rfind(".")]]
  dftemp = dftemp[dftemp["logM_Name"] == logMName[:logPName.rfind(".")]]
  dftemp = dftemp[dftemp["im_bi_sup"] == support]
  dftemp = dftemp[dftemp["im_bi_ratio"] == ratio]
  dftemp = dftemp[dftemp["pruning_threshold"] == pruning_threshold]
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

def applyMinerToLogForGNN(df, cut_Type, logPathP, logPathM,logPName, logMName = "", noiseThreshold = 0.0, dependency_threshold=0.0, support = 0, ratio = 0, pruning_threshold = 0, use_gnn = False):
  logP = get_log(logPathP)
  logM = get_log(logPathM)
    
  # imbi_ali
  # print("Running IMbi_ali, GNN: " + str(use_gnn))
  cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, pruning_threshold = pruning_threshold, size_par=len(logP)/len(logM), cost_Variant=cost_Variant,use_gnn=use_gnn)
  
  df = add_Model_To_Database(df=df,cut_Type=cut_Type, log=logP, logM=logM,net=net,im=im,fm=fm,name="IMbi_ali",logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio,pruning_threshold = pruning_threshold, use_gnn = use_gnn)
  
  save_cuts = False
  if save_cuts == True:
    fileName_cuts_ali = "cuts_IMbi_ali_sup_" + str(support) + "_ratio_" + str(ratio) + "_logP_" + logPName[:logPName.rfind(".")] + "_logM_" + logMName[:logMName.rfind(".")]
    visualize_cuts(fileName_cuts_ali)
  
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
  df_grouped = df.groupby(by=["logP_Name",	"logM_Name", "im_bi_sup", "im_bi_ratio", "pruning_threshold"], group_keys=True).apply(lambda x : x)
  numberOfPlotPerRow = 4
  rows = math.ceil(float(len(df_grouped.index.unique()))/numberOfPlotPerRow)
  cols = min(len(df_grouped.index.unique()),numberOfPlotPerRow)

  fig, axs = plt.subplots(rows, cols, figsize=(15 * (cols / numberOfPlotPerRow), 4 * rows), squeeze=False)
  fig.tight_layout(pad=10.0)
  cur_Row = 0
  cur_Col = 0
  
  for logGroup in df_grouped.index.unique():
    df_log_grouped = df_grouped.loc[logGroup]

    axs[cur_Row,cur_Col].set_title("LogP: " + logGroup[0] + " LogM: " + logGroup[1] + "\n" + "Sup: " + str(df_log_grouped.im_bi_sup[0]) + " ratio: " + str(df_log_grouped.im_bi_ratio[0]) + " Pruning: " + str(df_log_grouped.pruning_threshold[0]))
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
  columns = ["miner", "logP_Name", "logM_Name","imf_noise_thr","hm_depen_thr","im_bi_sup","im_bi_ratio", "pruning_threshold", "use_gnn","acc_logs", "fitP", "fitM", "f1_fit_logs", "precision", "net", "im", "fm"]

  df = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=None)
  df['use_gnn'] = df['use_gnn'].astype(bool)
  return df

def run_evaluation_synthetic(df, dataPath, num_data_per_category, using_gnn, max_node_size = 100, parallel = False):
  
  def read_data_from_path(file_path):
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
                    data["Pruning"] = int(line[:-1])
                    state += 1
                    continue 
                elif state == 9:
                    data["Size_par"] = float(line[:-1])
                    state += 1
                    continue 
                elif state == 10:
                    data["Dataitem"] = int(line[:-1])
                    state += 1
                    continue 
                elif state == 11:
                    data["PartitionA"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 12:
                    data["PartitionB"] = line.split(" ")[:-1]
                    state += 1
                    continue 
                elif state == 13:
                    data["Score"] = float(line[:-1])
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

  def get_last_number(string_input):
    last_underscore_index = string_input.rfind("_")
    if last_underscore_index != -1:
      substring_after_underscore = string_input[last_underscore_index + 1:]
      # Find the index of the first non-digit character in the substring
      index_of_non_digit = next((i for i, c in enumerate(substring_after_underscore) if not c.isdigit()), None)

      if index_of_non_digit is not None:
        # Extract the numbers before the non-digit character
        extracted_number = int(substring_after_underscore[:index_of_non_digit])
        return extracted_number
      else:
        print("Error, No numbers found after the last underscore.")
        return -1
    else:
      print("Error, No underscores found in the file name.")
      return -1
  
  def get_last_Data_number(string):
    parts = string.split('Data_')
    if len(parts) > 1:
        rest = parts[-1]
        number_part = rest.split('\\')[0]
        if number_part.isdigit():
            return int(number_part)
    return 101
  
  def is_string_present(string, list):
    if list == None:
      return False
    if len(list) == 0:
      return False
    for item in list:
      if string in item[0] or string in item[1]:
        return True
    return False
  
  cut_types = ["par", "exc","loop", "seq"]
  # cut_types = ["par", "exc", "seq"]
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
        
        for root, _, files in reversed(dirs_and_files):
          if get_last_Data_number(root) <= max_node_size:
            for file in files:
              if not continue_running:
                break
              if file.endswith(".xes"):  # Filter for text files
                found_file = os.path.join(root, file)
                if "logP" in found_file:
                  file_M = found_file.replace("logP", "logM")
                  file_P = found_file
                else:
                  file_M = found_file
                  file_P = found_file.replace("logM", "logP")
                
                
                input_detail_file = os.path.join(root, "Data_" + str(get_last_number(file_P)) + ".txt")
                if os.path.exists(input_detail_file):     
                  if os.path.exists(file_M):
                    if not is_string_present(file_P, pathFiles[cut_type]) and not is_string_present(file_M, pathFiles[cut_type]):
                      data = read_data_from_path(input_detail_file)
                      pathFiles[cut_type].append((file_P, file_M, data["Support"], data["Ratio"]))
                  if len(pathFiles[cut_type]) >= num_data_per_category:
                    continue_running = False
                    break

  for cut_type, dataList in pathFiles.items():
    print("Running: " + cut_type)
    if parallel:
      num_processors_available = multiprocessing.cpu_count()
      print("Number of available processors:", num_processors_available)
      num_processors = max(1,round(num_processors_available/2))
      print("Number of used processors:", num_processors)

      input_data = []
      for data in dataList:
        for gnn in using_gnn:
          df_temp = create_df()
          input_data.append((df_temp, cut_type, [data], [gnn], [0]))
      
      pool_res = None
      with multiprocessing.Pool(num_processors) as pool:
          pool_res = tqdm(pool.imap(run_evaluation_category_star, input_data),total=len(input_data))
          
          for result in pool_res:
              # Process individual evaluation result
              df = pd.concat([df, result])
    else: 
      df = run_evaluation_category(df, cut_type, dataList, using_gnn, [0])
  
  return df

def run_evaluation_real(df, dataPath, sup_list, ratio_list, using_gnn, parallel = False):
  pathFiles = []
  
  def is_string_present(string, list):
    if list == None:
      return False
    if len(list) == 0:
      return False
    for item in list:
      if string in item[0] or string in item[1]:
        return True
    return False
  
  
  currentPath = dataPath
  if os.path.exists(dataPath):
    for root, _, files in os.walk(currentPath):
        for file in files:
          if file.endswith(".xes"):  # Filter for text files
            found_file = os.path.join(root, file)
            if "lp" in found_file:
              file_M = found_file.replace("lp", "lm")
              file_P = found_file
            else:
              file_M = found_file
              file_P = found_file.replace("lm", "lp")
              
            if not is_string_present(file_P, pathFiles) and not is_string_present(file_M, pathFiles):
              pathFiles.append((file_P, file_M))

  enriched_pathFiles = []
  for dataList in pathFiles:
    for sup in sup_list:
      for ratio in ratio_list:
        enriched_pathFiles.append((dataList[0], dataList[1], sup, ratio))


  if parallel:
    num_processors_available = multiprocessing.cpu_count()
    print("Number of available processors:", num_processors_available)
    num_processors = max(1,round(num_processors_available/2))
    print("Number of used processors:", num_processors)

    input_data = []
    iterator = 0
    for data in enriched_pathFiles:
      iterator += 1
      for gnn in using_gnn:
        df_temp = create_df()
        input_data.append((df_temp, iterator, [data], [gnn], [0]))
    
    pool_res = None
    with multiprocessing.Pool(num_processors) as pool:
        pool_res = tqdm(pool.imap(run_evaluation_category_star, input_data),total=len(input_data))
        
        for result in pool_res:
            # Process individual evaluation result
            df = pd.concat([df, result])
  else: 
    df = run_evaluation_category(df, df_temp[0], enriched_pathFiles, using_gnn, [0])

  return df


def run_evaluation_category_star(args):
    return run_evaluation_category(*args)    

def run_evaluation_category(df, cut_type, dataList, using_gnn, pruning_thresholds):
  for data in dataList:
    support = data[2]
    ratio = data[3]
    for pruning_threshold in pruning_thresholds:
      for use_gnn in using_gnn:
        # print("Running: " + str(runs) + "/" + str(totalRuns))
        # print("Ratio: " + str(ratio) + " sup: " + str(support) + " pruning: " + str(pruning_threshold) + " gnn: " + str(use_gnn))
        df = applyMinerToLogForGNN(df, cut_type, data[0], data[1], data[0], data[1], 0.2, 0.99, support, ratio, pruning_threshold,use_gnn)
  return df

# Define a custom aggregation function
def custom_agg(group):
  if len(group) > 2:
    print(f"Aggregating {len(group)} rows for group: {group.iloc[0]}")

  use_gnn_sum = group.loc[group['use_gnn']]
  not_use_gnn_sum = group.loc[~group['use_gnn']]
  
  result = {
      'precision': not_use_gnn_sum['precision'].sum() - use_gnn_sum['precision'].sum(),
      'fitP': not_use_gnn_sum['fitP'].sum() - use_gnn_sum['fitP'].sum(),
      'fitM': not_use_gnn_sum['fitM'].sum() - use_gnn_sum['fitM'].sum(),
      'f1_fit_logs': not_use_gnn_sum['f1_fit_logs'].sum() - use_gnn_sum['f1_fit_logs'].sum(),
      'acc_logs': not_use_gnn_sum['acc_logs'].sum() - use_gnn_sum['acc_logs'].sum()
  }
  
  return pd.Series(result)

def get_measurement_delta(df, columnList, column_feature):
  
  columns_to_keep = columnList + column_feature +  ['use_gnn']

  # Trim the DataFrame to include only specific columns
  df = df[columns_to_keep]
  
  # pd.set_option("display.max_colwidth", 10)
  # print(df.head(10))
  # print(columnList)
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

def visualize_measurement(df_measurement, column_feature, use_synthetic, file_name = None):
  y_min = df_measurement[column_feature].min().min()
  y_max = df_measurement[column_feature].max().max()

  if use_synthetic:
    df_grouped = df_measurement.groupby("cut_type")
  else:
    df_grouped = df_measurement.groupby("logP_Name")

  fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 10))
  fig.suptitle('Synthetic Data Delta Measurement\nDelta = (No GNN) - (GNN)')

  for i, (cut_type, group) in enumerate(df_grouped):
    ax = axes[i]  # Select the specific axis for this subplot
    if use_synthetic:
      ax.set_title("Data folder: " + str(cut_type) + "\nDatasize: " + str(len(group)))  # Set title for the subplot
    else:
      ax.set_title("Data: " + str(os.path.basename(group.logP_Name.iloc[0])) + "\nDatasize: " + str(len(group)))  # Set title for the subplot
    ax.set_ylabel("Values")  # Set ylabel for the subplot
    
    x_ticks = []
    x_labels = []
    
    for j, col in enumerate(column_feature):
      group.boxplot(column=col,positions=[j],widths=0.5, ax=ax)
      
      # Modify x-axis tick labels
      x_ticks.append(j)
      x_labels.append("Δ " + col)
      
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)

    # TODO MAKE SMALLER AROUND MIN AND MAX
    y_interval = 0.2
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

      
def get_dataframe(synthetic_path = None, real_path = None):
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
  
  progress = sanity_check_input(synthetic_path, real_path)
  if progress == -1:
    sys.exit()
  if progress == 0:
    csv_filename = "output_synthetic.csv"
  else:
    csv_filename = "output_real.csv"
    
  if not os.path.exists(csv_filename):
    df = create_df()
    
    if progress == 0:
      using_gnn = [False, True]
      num_data_per_category = 300

      df = run_evaluation_synthetic(df, synthetic_path, num_data_per_category, using_gnn, max_node_size=5, parallel = True)
      # Save the DataFrame to a CSV file
      df.to_csv(csv_filename, index=False)
    elif progress == 1:
      using_gnn = [False, True]
      sup_list = [0, 0.2, 0.3]
      ratio_list = [0.5, 0.8, 1.0]
      df = run_evaluation_real(df, real_path, sup_list, ratio_list, using_gnn, parallel = True)
      # Save the DataFrame to a CSV file
      df.to_csv(csv_filename, index=False)
      
    else:
      sys.exit()
      
  else:
    df = pd.read_csv(csv_filename)
    
  return df
      
if __name__ == '__main__':
  
  column_feature = ["precision","acc_logs", "fitP", "fitM", "f1_fit_logs"]
  quasi_identifiers = ["logP_Name",	"logM_Name", "cut_type"]
    
  synthetic_path = os.path.join("GNN_partitioning", "GNN_Data")
  real_path = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  
  use_synthetic = True
  
  if use_synthetic:
    df = get_dataframe(synthetic_path=synthetic_path) 
    df_measurement = get_measurement_delta(df, quasi_identifiers, column_feature)
    visualize_measurement(df_measurement, column_feature, use_synthetic, "df_measurement_synthetic.png")
  else:
    df = get_dataframe(real_path=real_path) 
    df_measurement = get_measurement_delta(df, quasi_identifiers, column_feature)
    visualize_measurement(df_measurement, column_feature, use_synthetic, "df_measurement_real.png")

  
  

  
    
    
    
    
