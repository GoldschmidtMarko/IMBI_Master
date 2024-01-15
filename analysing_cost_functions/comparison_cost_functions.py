import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

from local_pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.importer.xes import importer as xes_importer
from local_pm4py.algo.analysis import custom_enum
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py import view_petri_net
from pm4py import write_pnml
from pm4py import save_vis_petri_net
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py import precision_alignments
from pm4py import precision_token_based_replay
from pm4py import discover_petri_net_heuristics
from pm4py.convert import convert_to_petri_net
from pm4py.algo.discovery.inductive import algorithm as pm4py_algorithm
from pm4py.algo.discovery.inductive.variants import imf as pm4py_imf
import pandas as pd
from pm4py.algo.discovery.inductive.algorithm import Variants as ind_Variants
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from local_pm4py.algo.analysis import Optimzation_Goals
import matplotlib.pyplot as plt
import numpy as np
import math
import fpdf
import time
from PIL import Image
import warnings
import multiprocessing
from tqdm import tqdm
import upperBoundCalculation.upperBoundCalculation as ubc
from matplotlib.lines import Line2D
import shutil

def get_original_log_paths(subString):
  pathP, pathM = get_data_paths()
  for p in pathP:
    if subString in p[1]:
      return p[1]
  for m in pathM:
    if subString in m[1]:
      return m[1]
  return None
  
def get_data_paths():
  # rootPath = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  # rootPath = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  # lpNames = ["2012_O_lp.xes", "2017_O_lp.xes"]
  # lMNames = ["2012_O_lm.xes", "2017_O_lm.xes"]
  rootPath = "C:/Users/Marko/Desktop/IMbi_Data/new-data/"
  lpNames = ["RTFM-LP.xes","BPIC-2012-LP.xes","BPIC-2017-LP.xes"]
  lMNames = ["RTFM-LM.xes","BPIC-2012-LM.xes","BPIC-2017-LM.xes"]
  
  
  lpPaths = []
  lmPaths = []

  for lp in lpNames:
    lpPaths.append((lp,rootPath + lp))
  for lm in lMNames:
    lmPaths.append((lm,rootPath + lm))
    
  return lpPaths, lmPaths

def f1_score(a, b):
  # try: catch
  if math.isclose(a+b,0):
    return 0
  return 2 * (a * b) / (a + b)

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

def isRowPresent(df, miner, logPName, logMName, imf_noiseThreshold, hm_dependency_threshold, im_bi_sup, im_bi_ratio):
  dftemp = df[df["miner"] == miner]
  dftemp = dftemp[dftemp["logP_Name"] == logPName]
  dftemp = dftemp[dftemp["logM_Name"] == logMName]
  dftemp = dftemp[dftemp["imf_noise_thr"] == imf_noiseThreshold]
  dftemp = dftemp[dftemp["hm_depen_thr"] == hm_dependency_threshold]
  dftemp = dftemp[dftemp["im_bi_sup"] == im_bi_sup]
  dftemp = dftemp[dftemp["im_bi_ratio"] == im_bi_ratio]
  if len(dftemp.index) > 0:
    return True
  return False

def runDoubleLogEvaluation(df,log,logM, name,net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, use_gnn = False):
  mes = Optimzation_Goals.apply_petri_silent(log,logM,net,im,fm)

  df = pd.concat([df, pd.DataFrame.from_records([{
    "miner" : name,
    "logP_Name": logPName[:logPName.rfind(".")],
    "logM_Name": logMName[:logMName.rfind(".")],
    "imf_noise_thr" : imf_noiseThreshold,
    "hm_depen_thr" : hm_dependency_threshold,
    "im_bi_sup" : im_bi_sup,
    "im_bi_ratio" : im_bi_ratio,
    "use_gnn" : use_gnn,
    "acc_align": mes['acc'],
    "acc_trace": mes['acc_perf'],
    "precP" : mes['precision'],
    "net": net,
    "im" : im,
    "fm" : fm
  }])])
  return df

def runSingleLogEvaluation(df,log,logM, name, net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, use_gnn = False):
  
  # if isRowPresent(df, name, logPName, logMName, imf_noiseThreshold, hm_dependency_threshold, im_bi_sup, im_bi_ratio) == True:
  #   print ("Skipped because present")
  #   return df
  
  parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT:"pdf"}
  gviz = pn_visualizer.apply(net, im, fm, parameters=parameters)

  try:
    fitness_token = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)["log_fitness"]
  except:
    fitness_token = 0
  try:
    fitness_align = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
  except:
    fitness_align = 0
    
  try:
    prec_token = precision_token_based_replay(log, net, im, fm)
  except:
    prec_token = 0
    
  try:
    prec_alignment = precision_alignments(log, net, im, fm)
  except:
    prec_alignment = 0
    

  df = pd.concat([df, pd.DataFrame.from_records([{
      "miner" : name,
      "logP_Name": logPName,
      "logM_Name": logMName,
      "imf_noise_thr" : imf_noiseThreshold,
      "hm_depen_thr" : hm_dependency_threshold,
      "im_bi_sup" : im_bi_sup,
      "im_bi_ratio" : im_bi_ratio,
      "use_gnn" : use_gnn,
      "fit_tok": fitness_token,
      "fit_alig": fitness_align,
      "prec_tok": prec_token,
      "prec_alig": prec_alignment,
      "f1_tok": f1_score(fitness_token, prec_token),
      "f1_alig": f1_score(fitness_align, prec_alignment),
      "net": net,
      "im" : im,
      "fm" : fm
  }])])
  return df

def add_Model_To_Database(df,log,logM, name,net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0, hm_dependency_threshold = 0,im_bi_sup = 0, im_bi_ratio = 0, use_gnn = False):
  if logMName == "":
    df = runSingleLogEvaluation(df,log,logM, name,net, im, fm, logPName,logMName, imf_noiseThreshold, hm_dependency_threshold,im_bi_sup, im_bi_ratio, use_gnn = use_gnn)
  else:
    df = runDoubleLogEvaluation(df,log,logM, name,net, im, fm, logPName,logMName, imf_noiseThreshold, hm_dependency_threshold,im_bi_sup, im_bi_ratio, use_gnn=use_gnn)
  return df

def get_df_value(df, miner, logPName, logMName, support, ratio, use_gnn, target_column):
  dftemp = df[df["miner"] == miner]
  dftemp = dftemp[dftemp["logP_Name"] == logPName[:logPName.rfind(".")]]
  dftemp = dftemp[dftemp["logM_Name"] == logMName[:logPName.rfind(".")]]
  dftemp = dftemp[dftemp["im_bi_sup"] == support]
  dftemp = dftemp[dftemp["im_bi_ratio"] == ratio]
  dftemp = dftemp[dftemp["use_gnn"] == use_gnn]
  if len(dftemp.index) > 1:
    raise Exception("Error, too many rows.")
  return dftemp[target_column].iloc[0]

def visualize_cuts(file_dir, file_name):
  file_dir_cuts = os.path.join(root_path,"imbi_cuts")
  if not os.path.exists(file_dir_cuts):
    return
  if len(os.listdir(file_dir_cuts)) == 0:
    return
  
  pdf = fpdf.FPDF(format='letter') #pdf format
  pdf.add_page() #create new page
  pdf.set_font("Arial", size=8) # font and textsize

  depth = 0
  currentIteration = 1
  
  file_dir_cuts_file = os.path.join(file_dir_cuts,"depth_" + str(depth) + "_It_" + str(currentIteration))
  
  # for depth
  while(os.path.isfile(file_dir_cuts_file + ".png")):
    # for iteration
    while(os.path.isfile(file_dir_cuts_file + ".png")):
      with open(file_dir_cuts_file + ".txt") as f:
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
      img = Image.open(file_dir_cuts_file + ".png")
      width,height = img.size
      # print(width, height)
      pdf.image(file_dir_cuts_file + ".png",w=min(150,width/3),h=min(150,height/3))
      pdf.add_page()
      currentIteration += 1
      file_dir_cuts_file = os.path.join(file_dir_cuts,"depth_" + str(depth) + "_It_" + str(currentIteration))
      
    depth += 1
    currentIteration = 1
    file_dir_cuts_file = os.path.join(file_dir_cuts, "depth_" + str(depth) + "_It_" + str(currentIteration))
    
  output_path = os.path.join(file_dir, file_name)
  pdf.output(output_path + ".pdf")
  
def run_imbi_miner(cost_Variant, cost_variant_name,logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn,imbi_cuts_path):
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant,use_gnn=use_gnn)
  df = add_Model_To_Database(df=df, log=logP, logM=logM,net=net,im=im,fm=fm,name=cost_variant_name,logPName=logPName, logMName=logMName, im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  
  fileName_cuts_mar = "cuts_IMbi_mar_sup_" + str(support) + "_ratio_" + str(ratio) + "_logP-" + logPName[:logPName.rfind(".")] + "_logM-" + logMName[:logMName.rfind(".")]
  visualize_cuts(result_path, fileName_cuts_mar)
  
  if os.path.exists(imbi_cuts_path):
    for f in os.listdir(imbi_cuts_path):
      os.remove(os.path.join(imbi_cuts_path, f))
      
  return df

def find_max_difference_tuple(tuples):
  max_values = None
  iterator = None

  for i, (a, b) in enumerate(tuples):
    if max_values == None:
      max_values = (a, b)
      iterator = i
      continue
    if a > max_values[0] and b > max_values[1]:
      max_values = (a, b)
      iterator = i
      continue
    
    if (a - max_values[0]) + (b - max_values[1]) > 0:
      max_values = (a, b)
      iterator = i
      continue

  return iterator, max_values
  
def applyMinerToLog_on_list(list_input):
  warnings.filterwarnings("ignore")
  res_df = create_df()
  mar_better_runs = 0
  ali_better_runs = 0
  aprox_better_runs = 0
  
  for i, input in enumerate(list_input):
    print("Running: ", i, " of ", len(list_input))
    df, result = applyMinerToLog(*input)
    res_df = pd.concat([res_df, df])
    if result == 0:
      ali_better_runs += 1
    if result == 1:
      mar_better_runs += 1
    if result == 2:
      aprox_better_runs += 1
    
  return res_df, (ali_better_runs, mar_better_runs, aprox_better_runs)
    
def applyMinerToLog(df, result_path, logPathP, logPathM,logPName, logMName = "", noiseThreshold = 0.0, dependency_threshold=0.0, support = 0, ratio = 0, use_gnn = False):
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  logP = xes_importer.apply(logPathP, parameters=parameter)
  if logMName == "":
    logM = logP
    # inductive miner
    print("Running IM")
    pt = pm4py_algorithm.apply(logP,variant=ind_Variants.IM)
    net, im, fm = convert_to_petri_net(pt)
    df = add_Model_To_Database(df=df, log=logP, logM=logM,net=net,im=im,fm=fm,name="IM",logPName=logPName, im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
    
    #imf 
    print("Running IMF")
    parameters = {pm4py_imf.IMFParameters.NOISE_THRESHOLD : noiseThreshold}
    pt = pm4py_algorithm.apply(logP,variant=ind_Variants.IMf, parameters=parameters)
    net, im, fm = convert_to_petri_net(pt)
    df = add_Model_To_Database(df=df, log=logP, logM=logM,net=net,im=im,fm=fm,name="IMF",logPName=logPName,imf_noiseThreshold=noiseThreshold, im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
    
    #hm
    print("Running HM")
    net, im, fm = discover_petri_net_heuristics(logP,dependency_threshold=dependency_threshold)
    df = add_Model_To_Database(df=df, log=logP, logM=logM,net=net,im=im,fm=fm,name="HM",logPName=logPName,hm_dependency_threshold=dependency_threshold, im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  else:
    parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
    logM = xes_importer.apply(logPathM, parameters=parameter)
    
  # imbi_ali
  # print("Running IMbi_cost: ", logPName)
  
  imbi_cuts_path = os.path.join(root_path,"imbi_cuts")
  
  df = run_imbi_miner(custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, "IMbi_freq",logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn,imbi_cuts_path)

  # imbi_mar  
  # print("Running IMbi_rel: ", logPName)
  df = run_imbi_miner(custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE, "IMbi_rel",logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn, imbi_cuts_path)
  
  # imbi_mar  
  # print("Running IMbi_aprox: ", logPName)
  df = run_imbi_miner(custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE, "IMbi_aprox",logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn, imbi_cuts_path)
         
  result = 0
  return df, result

def applyMinerToLogForGNN(df, logPathP, logPathM,logPName, logMName = "", noiseThreshold = 0.0, dependency_threshold=0.0, support = 0, ratio = 0, use_gnn = False):
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  logP = xes_importer.apply(logPathP, parameters=parameter)
  logM = xes_importer.apply(logPathM, parameters=parameter)
    
  # imbi_ali
  print("Running IMbi_ali")
  cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant,use_gnn=use_gnn)
  df = add_Model_To_Database(df=df,log=logP, logM=logM,net=net,im=im,fm=fm,name="IMbi_ali",logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  
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

    axs[cur_Row,cur_Col].set_title("LogP: " + logGroup[0] + " LogM: " + logGroup[1] + "\n" + "Sup: " + str(df_log_grouped.im_bi_sup[0]) + " ratio: " + str(df_log_grouped.im_bi_ratio[0]))
    axs[cur_Row,cur_Col].set_xlabel("Miners")
    j = 0
    xTickLabel = []
    idx = []
    minValue = 0
    for miner, use_gnn, precP, acc, fitP, fitM, f1_fit in zip(df_log_grouped.miner, df_log_grouped.use_gnn, df_log_grouped.precP, df_log_grouped.acc_logs, df_log_grouped.fitP, df_log_grouped.fitM, df_log_grouped.f1_fit_logs):
      minValue = min([minValue, acc, fitP, fitM, f1_fit])
      axs[cur_Row,cur_Col].bar(j,precP, color="r", label="precP")
      axs[cur_Row,cur_Col].bar(j+1,acc, color="black", label="acc")
      axs[cur_Row,cur_Col].bar(j+2,fitP, color="g", label="fitP")
      axs[cur_Row,cur_Col].bar(j+3,fitM, color="b", label="fitM")
      axs[cur_Row,cur_Col].bar(j+4,f1_fit, color="orange", label="f1_fit")
      # xTickLabel.append(miner + "\nGNN: " + str(use_gnn))
      xTickLabel.append(miner)
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
      
def displayDoubleLogSplit(df, saveFig = False, file_path = ""):
  df_group = df.groupby(by=["logP_Name",	"logM_Name"], group_keys=True).apply(lambda x : x)
  
  import matplotlib.font_manager as fm
  # print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
  custom_font = fm.FontProperties(family='Arial', size=24)
  
  for logGroup_index in df_group.index.unique():
    logP_name = logGroup_index[df_group.index.names.index('logP_Name')]
    logM_name = logGroup_index[df_group.index.names.index('logM_Name')]
    
    df_log_cur = df_group.loc[logGroup_index]

    df_grouped = df_log_cur.groupby(by=["im_bi_sup", "im_bi_ratio"], group_keys=True).apply(lambda x : x)
    numberOfPlotPerRow = 3
    rows = math.ceil(float(len(df_grouped.index.unique()))/numberOfPlotPerRow)
    cols = min(len(df_grouped.index.unique()),numberOfPlotPerRow)

    fig, axs = plt.subplots(rows, cols, figsize=(17 * (cols / numberOfPlotPerRow), 6 * rows), squeeze=False)
    fig.tight_layout(pad=18.0)
    cur_Row = 0
    cur_Col = 0
    
    logP_name_org = get_original_log_paths(logP_name)
    logM_name_org = get_original_log_paths(logM_name)
    ubs_align = None
    ub_trace = None
    
    if logP_name_org != None and logM_name_org != None:
      ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
      ub_trace = ubc.run_upper_bound_traces_on_logs(logP_name_org, logM_name_org)
    

    for logGroup in df_grouped.index.unique():
      df_log_grouped = df_grouped.loc[logGroup]
      
      im_bi_sup = logGroup[df_grouped.index.names.index('im_bi_sup')]
      im_bi_ratio = logGroup[df_grouped.index.names.index('im_bi_ratio')]
      
      axs[cur_Row,cur_Col].set_title("LogP: " + logP_name + " LogM: " + logM_name + "\n" + "Sup: " + str(im_bi_sup) + " ratio: " + str(im_bi_ratio), fontproperties=custom_font)
      axs[cur_Row,cur_Col].set_xlabel("Miners", fontproperties=custom_font)
      j = 0
      xTickLabel = []
      idx = []
      minValue = 0
      best_miner = ""
      
      for miner, use_gnn, precP, acc_align, acc_trace in zip(df_log_grouped.miner, df_log_grouped.use_gnn, df_log_grouped.precP, df_log_grouped.acc_align, df_log_grouped.acc_trace):
        minValue = min([minValue, precP, acc_align, acc_trace])
        axs[cur_Row,cur_Col].bar(j,precP, color="r", label="$prec(L^{+},M)$")
        axs[cur_Row,cur_Col].bar(j+1,acc_align, color="g", label="$acc_{align}(L^{+},L^{-},M)$")
        axs[cur_Row,cur_Col].bar(j+2,acc_trace, color="b", label="$acc_{trace}(L^{+},L^{-},M)$")
        
        if ubs_align != None and ub_trace != None:
          # ubs trace
          # Add a horizontal dotted line above the specific bar
          axs[cur_Row,cur_Col].hlines(ub_trace, xmin=j+1.5, xmax=j+2.5, colors='b', linestyles='dotted', linewidth=2)
          
          for enum, ub_align in ubs_align.items():
            # ubs align
            # Add a horizontal dotted line above the specific bar
            axs[cur_Row,cur_Col].hlines(ub_align, xmin=j+0.5, xmax=j+1.5, colors='g', linestyles='dotted', linewidth=2)
            
        if acc_trace > ub_trace:
          print("Trace: " + str(acc_trace) + " > " + str(ub_trace))
          print("Data: " + str(logP_name) + " " + str(logM_name) + " " + str(im_bi_sup) + " " + str(im_bi_ratio) + " " + str(miner) + " " + str(use_gnn))
        
        # xTickLabel.append(miner + "\nGNN: " + str(use_gnn))
        miner_text = ""
        if miner == "IMbi_freq":
          miner_text = "Cost-Func"
        elif miner == "IMbi_rel":
          miner_text = "Reward-Func"
        elif miner == "IMbi_aprox":
          miner_text = "Approx-Func"
          
        if miner == best_miner:
          miner_text = "$\\bf{" + miner_text + "}$"
        xTickLabel.append(miner_text)
          
        idx.append(j + 1.5)
        j += 4
        
      
      axs[cur_Row,cur_Col].set_yticks(setupYTickList(minValue, 0.25))
      axs[cur_Row,cur_Col].set_xticks(idx)
      axs[cur_Row,cur_Col].set_xticklabels(xTickLabel, rotation=0, fontsize=20)
      axs[cur_Row,cur_Col].tick_params(axis='x', labelsize=20)
      axs[cur_Row,cur_Col].tick_params(axis='y', labelsize=20)
      
      legend_elements = [
          Line2D([0], [0], color='r', lw=2, label="$prec(L^{+},M)$"),
          Line2D([0], [0], color='g', lw=2, label="$acc_{align}(L^{+},L^{-},M)$"),
          Line2D([0], [0], color='b', lw=2, label="$acc_{trace}(L^{+},L^{-},M)$")
      ]

      # Add the legend to the subplot
      axs[cur_Row, cur_Col].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.2, 0.5))
      cur_Col += 1
      
      if cur_Col == numberOfPlotPerRow:
        cur_Row += 1
        cur_Col = 0
    
    if saveFig:
      fig.savefig(os.path.join(file_path,"plot_" + logP_name + "_" + logM_name  + ".pdf"))
    else:
      plt.show()
    
def displayDoubleLogSplitSingleBest(df, saveFig = False, file_path = ""):
  df.reset_index(drop=True, inplace=True)
  df_group = df.groupby(by=["logP_Name",	"logM_Name"], group_keys=True)
  use_upper_bound = True
  
  numberOfPlotPerRow = 2
  rows = math.ceil(float(df_group.ngroups)/numberOfPlotPerRow)
  cols = min(df_group.ngroups,numberOfPlotPerRow)
  # fig, axs = plt.subplots(rows, cols, figsize=(17 * (cols / numberOfPlotPerRow), 10 * rows), squeeze=False)
  fig, axs = plt.subplots(rows, cols, figsize=(17 , 10), squeeze=False)
  fig.tight_layout(pad=18.0)
  cur_Row = 0
  cur_Col = 0
  
  output_file_name = "plot_Summary_logs"
  
  if os.path.exists(os.path.join(file_path, output_file_name +".txt")):
    os.remove(os.path.join(file_path, output_file_name +".txt"))
    
  with open(os.path.join(file_path, output_file_name +".txt"), "a") as util_para_file:
    for group_keys, group_df in df_group:
      logP_name = group_df['logP_Name'].iloc[0]
      logM_name = group_df['logM_Name'].iloc[0]

      logP_name_org = get_original_log_paths(logP_name)
      logM_name_org = get_original_log_paths(logM_name)
      ubs_align = None
      ub_trace = None
      
      if use_upper_bound and logP_name_org != None and logM_name_org != None:
        ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
        ub_trace = ubc.run_upper_bound_traces_on_logs(logP_name_org, logM_name_org)
        
      axs[cur_Row,cur_Col].set_title("LogP: " + logP_name + " LogM: " + logM_name)
      axs[cur_Row,cur_Col].set_xlabel("Miners")
      j = 0
      xTickLabel = []
      idx = []
      minValue = 0
      best_miner = ""

      for miner, use_gnn, precP, acc_align, acc_trace, sup, ratio in zip(group_df.miner, group_df.use_gnn, group_df.precP, group_df.acc_align, group_df.acc_trace, group_df.im_bi_sup, group_df.im_bi_ratio):
        
        util_para_file.write("Miner: " + miner + " | LogP: " + logP_name + " | LogM: " + logM_name + " | Sup: " + str(sup) + " | Ratio: " + str(ratio) + " | GNN:" + str(use_gnn) + "\n")
        
        
        
        minValue = min([minValue, precP, acc_align, acc_trace])
        axs[cur_Row,cur_Col].bar(j,precP, color="r", label="$prec(L^{+},M)$")
        axs[cur_Row,cur_Col].bar(j+1,acc_align, color="g", label="$acc_{align}(L^{+},L^{-},M)$")
        axs[cur_Row,cur_Col].bar(j+2,acc_trace, color="b", label="$acc_{trace}(L^{+},L^{-},M)$")
        
        if ubs_align != None and ub_trace != None:
          # ubs trace
          # Add a horizontal dotted line above the specific bar
          axs[cur_Row,cur_Col].hlines(ub_trace, xmin=j+1.5, xmax=j+2.5, colors='k', linestyles='dotted', linewidth=2)
          
          for enum, ub_align in ubs_align.items():
            # ubs align
            # Add a horizontal dotted line above the specific bar
            axs[cur_Row,cur_Col].hlines(ub_align, xmin=j+0.5, xmax=j+1.5, colors='k', linestyles='dotted', linewidth=2)
            
        if ub_trace != None and acc_trace > ub_trace:
          print("Trace: " + str(acc_trace) + " > " + str(ub_trace))
          print("Data: " + str(logP_name) + " " + str(logM_name) + " " + str(miner) + " " + str(use_gnn))
        
        # xTickLabel.append(miner + "\nGNN: " + str(use_gnn))
        miner_text = ""
        if miner == "IMbi_freq":
          miner_text = "Cost-Func"
        elif miner == "IMbi_rel":
          miner_text = "Reward-Func"
        elif miner == "IMbi_aprox":
          miner_text = "Aprox-Func"
          
        if miner == best_miner:
          miner_text = "$\\bf{" + miner_text + "}$"
        xTickLabel.append(miner_text)
          
        idx.append(j + 1.5)
        j += 4
        
      
      axs[cur_Row,cur_Col].set_yticks(setupYTickList(minValue, 0.25))
      axs[cur_Row,cur_Col].set_xticks(idx)
      axs[cur_Row,cur_Col].set_xticklabels(xTickLabel, rotation=90)
      
      legend_elements = [
          Line2D([0], [0], color='r', lw=2, label="$prec(L^{+},M)$"),
          Line2D([0], [0], color='g', lw=2, label="$acc_{align}(L^{+},L^{-},M)$"),
          Line2D([0], [0], color='b', lw=2, label="$acc_{trace}(L^{+},L^{-},M)$")
      ]

      # Add the legend to the subplot
      axs[cur_Row, cur_Col].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.2, 0.5))
      cur_Col += 1
      
      if cur_Col == numberOfPlotPerRow:
        cur_Row += 1
        cur_Col = 0
    
  if saveFig:
    fig.savefig(os.path.join(file_path, output_file_name +".pdf"))
  else:
    plt.show()


def create_df():
  columns = ["miner", "logP_Name", "logM_Name","imf_noise_thr","hm_depen_thr","im_bi_sup","im_bi_ratio", "use_gnn", "fit_tok", "fit_alig", "prec_tok", "prec_alig", "f1_tok", "f1_alig", "net", "im", "fm"]
  df = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=None)
  df['use_gnn'] = df['use_gnn'].astype(bool)
  return df

def save_df(df, name):
  df.to_csv(name)
  
def filter_df_for_best_models(df):
  def custom_agg(group):
    group = group.reset_index(drop=True)
    # Define your custom logic here
    best_index = 0
    for i in range(0,len(group.index)):
      cur_score = group.iloc[i]['acc_align'] + group.iloc[i]['acc_trace']
      best_score = group.iloc[best_index]['acc_align'] + group.iloc[best_index]['acc_trace']
      if group.iloc[i]['precP'] >= 0.6 and group.iloc[best_index]['precP'] >= 0.6:
        if cur_score > best_score:
          best_index = i
        continue
      if group.iloc[i]['precP'] >= 0.6 and group.iloc[best_index]['precP'] < 0.6:
        best_index = i
        continue
      cur_score += group.iloc[i]['precP']
      best_score += group.iloc[best_index]['precP']

      if cur_score > best_score:
        best_index = i

    return group.loc[best_index]

  df = df.reset_index(drop=True)
  filtered_df = df.groupby(['miner', 'logP_Name', 'logM_Name'], group_keys=False).apply(custom_agg).reset_index(drop=True)
  # Displaying the resulting DataFrame
  return filtered_df
  
def run_comparison(csv_filename, result_path, parallel = True):
  df = create_df()
  lpPaths, lmPaths = get_data_paths()

  if not os.path.exists(result_path):
    os.mkdir(result_path)
  


  with open(os.path.join(result_path,"output.txt"), 'w') as txt_file:
    
    mar_better_runs = 0
    ali_better_runs = 0
    aprox_better_runs = 0
    
    ratios = [1, 0.8, 0.5]
    # ratios = [1]
    sups = [0, 0.2, 0.3, 0.4]
    # sups = [0, 0.2]
    
    dataList = []
    for ratio in ratios:
      for sup in sups:
        for i in range(0,len(lpPaths)):
          df_temp = create_df()
          dataList.append((df_temp, result_path, lpPaths[i][1], lmPaths[i][1], lpPaths[i][0],lmPaths[i][0], 0.2, 0.99, sup, ratio))

    start_time = time.time()
    
    parallel = True
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
          input_data.append(batch_data)
          offset += batch_size
 

      warnings.filterwarnings("ignore")
      pool_res = None
      with multiprocessing.Pool(num_processors) as pool:
          pool_res = tqdm(pool.imap(applyMinerToLog_on_list, input_data),total=len(input_data))
          
          for result, res_tuples in pool_res:
              # Process individual evaluation result
              df = pd.concat([df, result])
              ali_better_runs += res_tuples[0]
              mar_better_runs += res_tuples[1]
              aprox_better_runs += res_tuples[2]
            
            
    else:
      warnings.filterwarnings("ignore")
      df = create_df()
      for input in dataList:
        txt_file.write("Running: " + str(input) + "\n")
        df_temp, _ = applyMinerToLog(*input)
        df = pd.concat([df, df_temp])
    txt_file.write("Time: " + str(time.time() - start_time) + "\n")
    save_df(df, os.path.join(result_path,csv_filename))
    return df
  
def save_petri_nets(df, result_path):
  folder = "petri_nets"
  folder_path = os.path.join(result_path,folder)
  if not os.path.exists(folder_path):
    os.mkdir(folder_path)
  else:
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    
  warnings.filterwarnings("ignore")
  
  for i in range(0,len(df.index)):
    net = df.iloc[i]['net']
    im = df.iloc[i]['im']
    fm = df.iloc[i]['fm']
    logPName = df.iloc[i]['logP_Name']
    logMName = df.iloc[i]['logM_Name']
    dataFolderName = logPName[:logPName.rfind(".")]
    if not os.path.exists(os.path.join(folder_path, dataFolderName)):
      os.mkdir(os.path.join(folder_path, dataFolderName))
    
    im_bi_sup = df.iloc[i]['im_bi_sup']
    im_bi_ratio = df.iloc[i]['im_bi_ratio']
    df_Variant = df.iloc[i]['miner']
    if df_Variant == "IMbi_freq":
      cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE
    elif df_Variant == "IMbi_rel":
      cost_Variant = custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE
    elif df_Variant == "IMbi_aprox":
      cost_Variant = custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE
    
    full_name_logP = get_original_log_paths(logPName)
    full_name_logM = get_original_log_paths(logMName)
    parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
    logP = xes_importer.apply(full_name_logP, parameters = parameter)
    logM = xes_importer.apply(full_name_logM, parameters = parameter)
    
    net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=im_bi_sup, ratio=im_bi_ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant)
    save_vis_petri_net(net,im,fm,os.path.join(folder_path, dataFolderName, "pr_" + df_Variant + "_"  + logPName + "_" + logMName + "_sup_"+ str(im_bi_sup) + "_ratio_"+ str(im_bi_ratio) + ".png"))
  
def get_comparison_df(csv_filename, result_path):
  if not os.path.exists(os.path.join(result_path,csv_filename)):
    df = run_comparison(csv_filename, result_path)
  else:
    df = pd.read_csv(os.path.join(result_path,csv_filename),index_col=0)
    
  displayDoubleLogSplit(df, saveFig=True, file_path=result_path)
  df = filter_df_for_best_models(df)
  save_petri_nets(df, result_path)
  displayDoubleLogSplitSingleBest(df, saveFig=True, file_path=result_path)
  
  return df
   
def filter_and_sort_dataframe(df, number_rows, get_mar_improved = True):
  def custom_agg(group):
    precision_diff = group.loc[group['miner'] == "IMbi_freq", 'precP'].mean() - \
                    group.loc[group['miner'] == "IMbi_rel", 'precP'].mean()

    fit_f1_diff = group.loc[group['miner'] == "IMbi_freq", 'f1_fit_logs'].mean() - \
                group.loc[group['miner'] == "IMbi_rel", 'f1_fit_logs'].mean()

    return pd.Series({'precision': precision_diff, 'fit_f1': fit_f1_diff})

  column_features = ['logP_Name', 'logM_Name', 'im_bi_sup', 'im_bi_ratio', 'use_gnn']

  # Apply the custom aggregation function and reset the index
  df_measurement = df.groupby(column_features).apply(custom_agg).reset_index()
  if get_mar_improved:
    df_measurement = df_measurement[df_measurement['precision'] < 0]
    df_measurement = df_measurement[df_measurement['fit_f1'] < 0]
  else:
    df_measurement = df_measurement[df_measurement['precision'] > 0]
    df_measurement = df_measurement[df_measurement['fit_f1'] > 0]
  df_measurement["precision"] = df_measurement["precision"].abs()
  df_measurement["fit_f1"] = df_measurement["fit_f1"].abs()
  
  df_measurement["sum_scores"] = df_measurement["fit_f1"] + df_measurement["precision"]
  # Sort the filtered DataFrame by "f1" column in descending order
  sorted_df = df_measurement.sort_values(by='sum_scores', ascending=False)
  
  # Return the top n rows
  result_df = sorted_df.head(number_rows)
    
  return result_df

def create_comparison_petri_nets(result_path, logP_path, logM_path, sup, ratio):
  rootPath = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  logP = xes_importer.apply(rootPath + logP_path, parameters=parameter)
  logM = xes_importer.apply(rootPath + logM_path, parameters=parameter)
  
  cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=sup, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant)

  cost_Variant = custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE
  net_mar, im_mar, fm_mar = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=sup, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant)
  
  save_vis_petri_net(net,im,fm,os.path.join(result_path, "petri1.png"))
  save_vis_petri_net(net_mar,im_mar,fm_mar,os.path.join(result_path, "petri2.png"))

  import matplotlib.image as mpimg
  # Load the first PNG image
  image1 = mpimg.imread(os.path.join(result_path, "petri1.png"))

  # Load the second PNG image
  image2 = mpimg.imread(os.path.join(result_path, "petri2.png"))

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
  ax1.set_title("PetriNet cost: ali", fontsize=12)  # Add a title above the first image

  # Plot the second image on the second subplot
  ax2.imshow(np.pad(image2, ((0, 0), (width_diff2, width_diff2), (0, 0)), mode='constant'))
  ax2.axis('off')  # Turn off axes
  ax2.set_title("PetriNet cost: Mar", fontsize=12)  # Add a title above the second image

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.0)  # You can adjust the vertical spacing as needed

  # Save the combined image
  plt.savefig(os.path.join(result_path, "file.png"), bbox_inches='tight')  # bbox_inches='tight' to remove extra 

def create_petri_net_model(file_name, result_path, logP_path, logM_path, sup, ratio, cost_Variant):
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  logP = xes_importer.apply(logP_path, parameters=parameter)
  logM = xes_importer.apply(logM_path, parameters=parameter)
  
  net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=sup, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant)
  
  from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
  log_fitness = replay_fitness.evaluate(logP, variant=replay_fitness.Variants.ALIGNMENT_BASED)

  print(log_fitness)
  
  write_pnml(net, im, fm, os.path.join(result_path,file_name + ".pnml"))
  
def generate_manual_models(result_path):
  rootPath = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  lpNames = ["2012_O_lp.xes", "2017_O_lp.xes"]
  lMNames = ["2012_O_lm.xes", "2017_O_lm.xes"]
  
  create_petri_net_model("petriNet1",result_path, rootPath + lpNames[0], rootPath + lMNames[0], 0.2, 0.5, custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE)
   
if __name__ == '__main__':
  data_path = os.path.join(root_path, "analysing_cost_functions")
  result_path = os.path.join(data_path, "results")
  
  # generate_manual_models(result_path)
  df = get_comparison_df("df.csv", result_path)


  # df_measurement = filter_and_sort_dataframe(df, 10, get_mar_improved = True)
  # create_comparison_petri_nets(result_path, "lp_2017_f.xes","lm_2017_f.xes",0.4,0.8)

  


















































