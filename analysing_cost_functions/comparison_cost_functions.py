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
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
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

def get_original_log_paths(subString, file_path):
  pathP, pathM = get_data_paths(file_path)
  for p in pathP:
    if subString in p[1]:
      return p[1]
  for m in pathM:
    if subString in m[1]:
      return m[1]
  return None
  
def get_data_paths(file_path):
  # rootPath = "C:/Users/Marko/Desktop/IMbi_Data/analysing/"
  # rootPath = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  # lpNames = ["2012_O_lp.xes", "2017_O_lp.xes"]
  # lMNames = ["2012_O_lm.xes", "2017_O_lm.xes"]
  data_folder = "comparison-data"
  logs_path_root = os.path.join(file_path,data_folder)
  
  # rootPath = os.path.join(root_path,"analysing_cost_functions","new-data")
  lpNames = ["RTFM-LP.xes", "BPIC12-A-LP.xes", "BPIC17-A-LP.xes"]
  lMNames = ["RTFM-LM.xes", "BPIC12-A-LM.xes", "BPIC17-A-LM.xes"]
  
  
  lpPaths = []
  lmPaths = []

  for lp in lpNames:
    lpPaths.append((lp,os.path.join(logs_path_root,lp)))
  for lm in lMNames:
    lmPaths.append((lm,os.path.join(logs_path_root,lm)))
    
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
    "fitP-Align": mes['fitP'],
    "fitM-Align": mes['fitM'],
    "fitP-Trace": mes['fitPTrace'],
    "fitM-Trace": mes['fitMTrace'],
    "acc_align": mes['acc'],
    "acc_trace": mes['acc_perf'],
    "f1_align" : mes['F1'],
    "f1_trace" : mes['F1_perf'],
    "precP" : mes['precision'],
    "net": net,
    "im" : im,
    "fm" : fm
  }])])
  return df

def runSingleLogEvaluation(df,log,logM, name, net, im, fm, logPName = "",logMName = "", imf_noiseThreshold = 0.0, hm_dependency_threshold = 0.0,im_bi_sup = 0.0, im_bi_ratio = 0.0, use_gnn = False):
  
  from pm4py.algo.evaluation.precision.variants.align_etconformance import Parameters as align_etconformance_parameters
  from pm4py.util import constants
  
  constants.ENABLE_MULTIPROCESSING_DEFAULT = True
  parameter = {alignments.Parameters.SHOW_PROGRESS_BAR: True}
  
  parametersAlign = {align_etconformance_parameters.SHOW_PROGRESS_BAR: True, align_etconformance_parameters.MULTIPROCESSING: True}
  
  print("Running precision")
  try:
    prec = precision_evaluator.apply(log, net, im, fm,parameters=parametersAlign,
                                          variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
  except:
    prec = 0
    
  print("Running alignment")
  try:
    alp = alignments.apply_log(log, net, im, fm,parameters=parameter)
    fp_inf = replay_fitness.evaluate(alp,parameters=parameter, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    align_fit = fp_inf['averageFitness']
    trace_fit = fp_inf['percentage_of_fitting_traces']/100
  except:
    align_fit = 0
    trace_fit = 0
    
  df = pd.concat([df, pd.DataFrame.from_records([{
    "miner" : name,
    "logP_Name": logPName,
    "logM_Name": "",
    "imf_noise_thr" : imf_noiseThreshold,
    "hm_depen_thr" : hm_dependency_threshold,
    "im_bi_sup" : im_bi_sup,
    "im_bi_ratio" : im_bi_ratio,
    "use_gnn" : use_gnn,
    "fitP-Align": align_fit,
    "fitM-Align": 0,
    "fitP-Trace": 0,
    "fitM-Trace": 0,
    "acc_align": 0,
    "acc_trace": 0,
    "f1_align" : 0,
    "f1_trace" : 0,
    "precP" : prec,
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
  print("Running IMbi_cost: ", logPName)
  
  imbi_cuts_path = os.path.join(root_path,"imbi_cuts")
  df = run_imbi_miner(custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, "IMbi_freq",logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn,imbi_cuts_path)

  # imbi_mar  
  print("Running IMbi_rel: ", logPName)
  df = run_imbi_miner(custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE, "IMbi_rel",logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn, imbi_cuts_path)
  
  # imbi_mar  
  print("Running IMbi_aprox: ", logPName)
  df = run_imbi_miner(custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE, "IMbi_aprox",logP, logM, logPName, logMName, support, ratio, df, result_path, use_gnn, imbi_cuts_path)
         
  result = 0
  return df, result

def applySingleMinerToLog(df, miner, result_path, logPathP, logPathM, logPName, logMName = "", noiseThreshold = 0.0, dependency_threshold=0.0, support = 0, ratio = 0, use_gnn = False):
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  logP = xes_importer.apply(logPathP, parameters=parameter)
  logM = xes_importer.apply(logPathM, parameters=parameter)
  
  print("Run: " + miner + " on: " + logPName + " and " + logMName + " with noise: " + str(noiseThreshold) + " and dependency: " + str(dependency_threshold) + " and support: " + str(support) + " and ratio: " + str(ratio) + " and use_gnn: " + str(use_gnn))
  
  if miner == "IMF":
    parameters = {pm4py_imf.IMFParameters.NOISE_THRESHOLD : noiseThreshold}
    pt = pm4py_algorithm.apply(logP,variant=ind_Variants.IMf, parameters=parameters)
    net, im, fm = convert_to_petri_net(pt)
    df = add_Model_To_Database(df=df,log=logP, logM=logM,net=net,im=im,fm=fm,name=miner,logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  elif miner == "IMbi_freq":
    net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE,use_gnn=use_gnn)
    df = add_Model_To_Database(df=df,log=logP, logM=logM,net=net,im=im,fm=fm,name=miner,logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  elif miner == "IMbi_rel":
    net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE,use_gnn=use_gnn)
    df = add_Model_To_Database(df=df,log=logP, logM=logM,net=net,im=im,fm=fm,name=miner,logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
  elif miner == "IMbi_aprox":
    net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=support, ratio=ratio, size_par=len(logP)/len(logM), cost_Variant=custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE,use_gnn=use_gnn)
    df = add_Model_To_Database(df=df,log=logP, logM=logM,net=net,im=im,fm=fm,name=miner,logPName=logPName, logMName=logMName,im_bi_sup=support,im_bi_ratio=ratio, use_gnn = use_gnn)
    
    
  return df


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
    axs[cur_Row,cur_Col].set_xticklabels(xTickLabel, fontsize=20)
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
    
    logP_name_org = get_original_log_paths(logP_name, file_path)
    logM_name_org = get_original_log_paths(logM_name, file_path)
    ubs_align = None
    ub_trace = None
    
    if logP_name_org != None and logM_name_org != None:
      ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
      ub_trace = ubc.run_upper_bound_traces_simple(logP_name_org, logM_name_org)
    

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
  
  numberOfPlotPerRow = 1
  rows = math.ceil(float(df_group.ngroups)/numberOfPlotPerRow)
  cols = min(df_group.ngroups,numberOfPlotPerRow)
  # fig, axs = plt.subplots(rows, cols, figsize=(17 * (cols / numberOfPlotPerRow), 10 * rows), squeeze=False)
  fig, axs = plt.subplots(rows, cols, figsize=(40 , 44), squeeze=False)
  # fig.tight_layout(pad=18.0)
  cur_Row = 0
  cur_Col = 0
  
  import matplotlib.font_manager as fm
  custom_font = fm.FontProperties(family='Arial', size=36)
  
  output_file_name = "plot_Summary_logs"
  
  if os.path.exists(os.path.join(file_path, output_file_name +".txt")):
    os.remove(os.path.join(file_path, output_file_name +".txt"))
    
  with open(os.path.join(file_path, output_file_name +".txt"), "a") as util_para_file:
    for group_keys, group_df in df_group:
      logP_name = group_df['logP_Name'].iloc[0]
      logM_name = group_df['logM_Name'].iloc[0]

      logP_name_org = get_original_log_paths(logP_name, file_path)
      logM_name_org = get_original_log_paths(logM_name, file_path)
      ubs_align = None
      ub_trace = None
      
      if use_upper_bound and logP_name_org != None and logM_name_org != None:
        ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
        ub_trace = ubc.run_upper_bound_traces_on_logs(logP_name_org, logM_name_org)
        
      axs[cur_Row,cur_Col].set_title("LogP: " + logP_name + "\nLogM: " + logM_name, fontproperties=custom_font)
      axs[cur_Row,cur_Col].set_xlabel("Miners", fontproperties=custom_font)
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
      axs[cur_Row,cur_Col].set_xticklabels(xTickLabel, rotation=0, fontsize=30)
      axs[cur_Row,cur_Col].tick_params(axis='x', labelsize=30)
      axs[cur_Row,cur_Col].tick_params(axis='y', labelsize=30)
      
      legend_elements = [
          Line2D([0], [0], color='r', lw=2, label="$prec(L^{+},M)$"),
          Line2D([0], [0], color='g', lw=2, label="$acc_{align}(L^{+},L^{-},M)$"),
          Line2D([0], [0], color='b', lw=2, label="$acc_{trace}(L^{+},L^{-},M)$")
      ]

      # Add the legend to the subplot
      axs[cur_Row, cur_Col].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.00, 0.5),prop={'size': 30})
      cur_Col += 1
      
      if cur_Col == numberOfPlotPerRow:
        cur_Row += 1
        cur_Col = 0
    
  if saveFig:
    fig.savefig(os.path.join(file_path, output_file_name +".pdf"))
  else:
    plt.show()

def displayDoubleLogSplitSingleBest_TrueSplit(df, saveFig = False, file_path = ""):
  df.reset_index(drop=True, inplace=True)
  df_group = df.groupby(by=["logP_Name",	"logM_Name"], group_keys=True)
  use_upper_bound = True
  
  numberOfPlotPerRow = 1
  rows = math.ceil(float(df_group.ngroups)/numberOfPlotPerRow)
  cols = min(df_group.ngroups,numberOfPlotPerRow)
  
  import matplotlib.font_manager as fm
  custom_font = fm.FontProperties(family='Arial', size=24)
  
  output_file_name = "plot_Summary_logs"
  
  if os.path.exists(os.path.join(file_path, output_file_name +".txt")):
    os.remove(os.path.join(file_path, output_file_name +".txt"))
    
  with open(os.path.join(file_path, output_file_name +".txt"), "a") as util_para_file:
    for group_keys, group_df in df_group:
      logP_name = group_df['logP_Name'].iloc[0]
      logM_name = group_df['logM_Name'].iloc[0]

      logP_name_org = get_original_log_paths(logP_name, file_path)
      logM_name_org = get_original_log_paths(logM_name, file_path)
      ubs_align = None
      ub_trace = None
      
      if use_upper_bound and logP_name_org != None and logM_name_org != None:
        ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
        ub_trace = ubc.run_upper_bound_traces_on_logs(logP_name_org, logM_name_org)
        
      fig, axs = plt.subplots(figsize=(14 , 12))
      # fig.tight_layout(pad=18.0)
        
      axs.set_title("LogP: " + logP_name + "\nLogM: " + logM_name, fontproperties=custom_font)
      axs.set_xlabel("Miners", fontproperties=custom_font)
      j = 0
      xTickLabel = []
      idx = []
      minValue = 0
      best_miner = ""

      pointA_X = []
      pointA_Y = []
      pointB_X = []
      pointB_Y = []
      pointC_X = []
      pointC_Y = []
      pointD_X = []
      pointD_Y = []

      for miner, use_gnn, precP, acc_align, acc_trace, f1_align, f1_trace , sup, ratio in zip(group_df.miner, group_df.use_gnn, group_df.precP, group_df.acc_align, group_df.acc_trace, group_df.f1_align, group_df.f1_trace, group_df.im_bi_sup, group_df.im_bi_ratio):
        
        util_para_file.write("Miner: " + miner + " | LogP: " + logP_name + " | LogM: " + logM_name + " | Sup: " + str(sup) + " | Ratio: " + str(ratio) + " | GNN:" + str(use_gnn) + "\n")
        
        
        bar_width = 0.8
        minValue = min([minValue, precP, acc_align, acc_trace])
        axs.bar(j,precP, color="r", label="$prec(L^{+},M)$", width=bar_width)
        axs.bar(j+1,acc_align, color="g", label="$acc_{align}(L^{+},L^{-},M)$", width=bar_width)
        axs.bar(j+2,acc_trace, color="b", label="$acc_{trace}(L^{+},L^{-},M)$", width=bar_width)
        axs.bar(j+3,f1_align, color="m", label="$acc_{align}(L^{+},L^{-},M)$", width=bar_width)
        axs.bar(j+4,f1_trace, color="tab:orange", label="$acc_{trace}(L^{+},L^{-},M)$", width=bar_width)
        
        if ubs_align != None and ub_trace != None:
          # ubs trace
          # Add a horizontal dotted line above the specific bar
          axs.hlines(ub_trace, xmin=j+1.5, xmax=j+2.5, colors='k', linestyles='dotted', linewidth=4)
          text_x = j + 2  # Adjust x-coordinate as needed
          text_y = ub_trace + 0.02  # Adjust y-coordinate as needed
          pointD_X.append(text_x)
          pointD_Y.append(text_y)
          # axs.text(text_x, text_y, text, ha='center', va='bottom', fontsize=26, color='black')

          
          for enum, ub_align in ubs_align.items():
            # ubs align
            # Add a horizontal dotted line above the specific bar
            axs.hlines(ub_align, xmin=j+0.5, xmax=j+1.5, colors='k', linestyles='dotted', linewidth=4)
            text_x = j + 1  # Adjust x-coordinate as needed
            text_y = ub_align + 0.02  # Adjust y-coordinate as needed
            if enum == ubc.ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE:
              pointA_X.append(text_x)
              pointA_Y.append(text_y)
            if enum == ubc.ShorestModelPathEstimation.ALLOW_LONGEST_SEQUENCE_PART:
              pointB_X.append(text_x)
              pointB_Y.append(text_y)
            if enum == ubc.ShorestModelPathEstimation.ALLOW_MIN_TRACE_LENGTH:
              pointC_X.append(text_x)
              pointC_Y.append(text_y)

            # axs.text(text_x, text_y, text, ha='center', va='bottom', fontsize=26, color='black')
            
            
            
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
          miner_text = "Approx-Func"
          
        if miner == best_miner:
          miner_text = "$\\bf{" + miner_text + "}$"
        xTickLabel.append(miner_text)
          
        idx.append(j + 2.5)
        j += 6
        
      
      axs.set_yticks(setupYTickList(minValue, 0.25))
      axs.set_xticks(idx)
      axs.set_xticklabels(xTickLabel, rotation=0, fontsize=24)
      axs.tick_params(axis='x', labelsize=24)
      axs.tick_params(axis='y', labelsize=24)
      
      
      legend_elements = [
        Line2D([0], [0], color='r', lw=4, label=r"$\operatorname{prec}$"),
        Line2D([0], [0], color='g', lw=4, label=r"$\operatorname{align{-}acc}$"),
        Line2D([0], [0], color='b', lw=4, label=r"$\operatorname{trace{-}acc}$"),
        Line2D([0], [0], color='m', lw=4, label=r"$\operatorname{align{-}F1{-}score}$"),
        Line2D([0], [0], color='c', lw=4, label=r"$\operatorname{trace{-}F1{-}score}$"),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'A $\operatorname{est{-}ub{-}acc_{\operatorname{trace}}}$', markerfacecolor='white'),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'B $\operatorname{ub{-}acc_{\operatorname{align}}}(\beta_1)$', markerfacecolor='white'),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'C $\operatorname{ub{-}acc_{\operatorname{align}}}(\beta_2)$', markerfacecolor='white'),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'D $\operatorname{ub{-}acc_{\operatorname{align}}}(\beta_3)$', markerfacecolor='white'),
      ]
      scatter1 = axs.scatter(pointA_X, pointA_Y, marker=r'$\operatorname{A}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{align{-}acc}}(\beta_1)$')
      scatter2 = axs.scatter(pointB_X, pointB_Y, marker=r'$\operatorname{B}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{align{-}acc}}(\beta_2)$')
      scatter3 = axs.scatter(pointC_X, pointC_Y, marker=r'$\operatorname{C}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{align{-}acc}}(\beta_3)$')
      scatter4 = axs.scatter(pointD_X, pointD_Y, marker=r'$\operatorname{D}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{trace{-}acc}}$')


      # Add the legend to the subplot
      lgd = axs.legend(handles=legend_elements + [scatter1, scatter2, scatter3, scatter4], loc='center left', bbox_to_anchor=(1.00, 0.5),prop={'size': 24})

      
      if saveFig:
        fig.savefig(os.path.join(file_path, output_file_name + "-" + logP_name +".pdf"), bbox_extra_artists=(lgd,), bbox_inches='tight')
      else:
        plt.show()


def calculate_difference_without_outliers(data):

    # Calculate quartiles
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    return IQR

def displayDoubleLogSplitBoxplot_TrueSplit(df, saveFig = False, file_path = ""):
  df.reset_index(drop=True, inplace=True)
  df_group = df.groupby(by=["logP_Name",	"logM_Name"], group_keys=True)
  use_upper_bound = True
  
  numberOfPlotPerRow = 1
  rows = math.ceil(float(df_group.ngroups)/numberOfPlotPerRow)
  cols = min(df_group.ngroups,numberOfPlotPerRow)
  
  import matplotlib.font_manager as fm
  custom_font = fm.FontProperties(family='Arial', size=24)
  
  output_file_name = "plot_Summary_logs"
  
  if os.path.exists(os.path.join(file_path, output_file_name +".txt")):
    os.remove(os.path.join(file_path, output_file_name +".txt"))
    
  with open(os.path.join(file_path, output_file_name +".txt"), "a") as util_para_file:
    for group_keys, group_df in df_group:
      logP_name = group_df['logP_Name'].iloc[0]
      logM_name = group_df['logM_Name'].iloc[0]

      logP_name_org = get_original_log_paths(logP_name, file_path)
      logM_name_org = get_original_log_paths(logM_name, file_path)
      ubs_align = None
      ub_trace = None
      
      if use_upper_bound and logP_name_org != None and logM_name_org != None:
        ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
        ub_trace = ubc.run_upper_bound_traces_on_logs(logP_name_org, logM_name_org)
        
      fig, axs = plt.subplots(figsize=(14 , 12))
      # fig.tight_layout(pad=18.0)
        
      axs.set_title("LogP: " + logP_name + "\nLogM: " + logM_name, fontproperties=custom_font)
      axs.set_xlabel("Miners", fontproperties=custom_font)
      j = 0
      xTickLabel = []
      idx = []
      minValue = 0
      best_miner = ""

      pointA_X = []
      pointA_Y = []
      pointB_X = []
      pointB_Y = []
      pointC_X = []
      pointC_Y = []
      pointD_X = []
      pointD_Y = []
      
      boxplot_width = 0.3
      boxplot_distance = 0.4
      
      difference_to_switch_color = 0.02
      
      df_group_on_miners = group_df.groupby(by=["miner"], group_keys=True)
      for group_keys, group_df_miner in df_group_on_miners:
        
        util_para_file.write("LogP: " + logP_name + " | Miner: " + group_keys + " | Dataitems: " + str(len(group_df_miner)) + "\n")
        difference = calculate_difference_without_outliers(group_df_miner.precP)
        axs.boxplot(group_df_miner.precP,positions=[j],widths=boxplot_width, patch_artist=True,    # Fill boxes with color
                        showfliers=False,     # Hide outliers
                        boxprops={'facecolor': '#FFCCCC', 'linewidth': 2,'edgecolor': 'r'},
                        capprops={'linewidth': 2, "color" : 'r'},
                        whiskerprops={'linewidth': 2,'color': 'r'},
                        medianprops={'color': 'r' if difference <= difference_to_switch_color else 'black', 'linewidth': 2},
                        notch=False)
        
        difference = calculate_difference_without_outliers(group_df_miner.acc_align)
        axs.boxplot(group_df_miner.acc_align,positions=[j+boxplot_distance],widths=boxplot_width, patch_artist=True,    # Fill boxes with color
                        showfliers=False,     # Hide outliers
                        boxprops={'facecolor': 'lightgreen', 'linewidth': 2,'edgecolor': 'g'},
                        capprops={'linewidth': 2,'color': 'g'},
                        whiskerprops={'linewidth': 2,'color': 'g'},
                        medianprops={'color': 'g' if difference <= difference_to_switch_color else 'black', 'linewidth': 2},
                        notch=False)
        
        difference = calculate_difference_without_outliers(group_df_miner.acc_trace)
        axs.boxplot(group_df_miner.acc_trace,positions=[j+2*boxplot_distance],widths=boxplot_width, patch_artist=True,    # Fill boxes with color
                        showfliers=False,     # Hide outliers
                        boxprops={'facecolor': 'lightblue', 'linewidth': 2,'edgecolor': 'b'},
                        capprops={'linewidth': 2,'color': 'b'},
                        whiskerprops={'linewidth': 2,'color': 'b'},
                        medianprops={'color': 'b' if difference <= difference_to_switch_color else 'black', 'linewidth': 2},
                        notch=False)
        
        difference = calculate_difference_without_outliers(group_df_miner.f1_align)
        axs.boxplot(group_df_miner.f1_align,positions=[j+3*boxplot_distance],widths=boxplot_width, patch_artist=True,    # Fill boxes with color
                        showfliers=False,     # Hide outliers
                        boxprops={'facecolor': 'lavenderblush', 'linewidth': 2,'edgecolor': 'm'},
                        capprops={'linewidth': 2,'color': 'm'},
                        whiskerprops={'linewidth': 2,'color': 'm'},
                        medianprops={'color': 'm' if difference <= difference_to_switch_color else 'black', 'linewidth': 2},
                        notch=False)
        
        difference = calculate_difference_without_outliers(group_df_miner.f1_trace)
        axs.boxplot(group_df_miner.f1_trace,positions=[j+4*boxplot_distance],widths=boxplot_width, patch_artist=True,    # Fill boxes with color
                        showfliers=False,     # Hide outliers
                        boxprops={'facecolor': 'lightcyan', 'linewidth': 2,'edgecolor': 'c'},
                        capprops={'linewidth': 2,'color': 'c'},
                        whiskerprops={'linewidth': 2,'color': 'c'},
                        medianprops={'color': 'c' if difference <= difference_to_switch_color else 'black', 'linewidth': 2},
                        notch=False)
        
        if ubs_align != None and ub_trace != None:
          # ubs trace
          # Add a horizontal dotted line above the specific bar
          axs.hlines(ub_trace, xmin=j+2*boxplot_distance - boxplot_width/2, xmax=j+2*boxplot_distance + boxplot_width/2, colors='k', linestyles='dotted', linewidth=4)
          text_x = j + 2*boxplot_distance   # Adjust x-coordinate as needed
          text_y = ub_trace + 0.02  # Adjust y-coordinate as needed
          pointD_X.append(text_x)
          pointD_Y.append(text_y)
          # axs.text(text_x, text_y, text, ha='center', va='bottom', fontsize=26, color='black')

          
          for enum, ub_align in ubs_align.items():
            # ubs align
            # Add a horizontal dotted line above the specific bar
            axs.hlines(ub_align, xmin=j+ boxplot_distance - boxplot_width/2, xmax=j+boxplot_distance + boxplot_width/2, colors='k', linestyles='dotted', linewidth=4)
            text_x = j + boxplot_distance  # Adjust x-coordinate as needed
            text_y = ub_align + 0.02  # Adjust y-coordinate as needed
            if enum == ubc.ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE:
              pointA_X.append(text_x)
              pointA_Y.append(text_y)
            if enum == ubc.ShorestModelPathEstimation.ALLOW_LONGEST_SEQUENCE_PART:
              pointB_X.append(text_x)
              pointB_Y.append(text_y)
            if enum == ubc.ShorestModelPathEstimation.ALLOW_MIN_TRACE_LENGTH:
              pointC_X.append(text_x)
              pointC_Y.append(text_y)

            # axs.text(text_x, text_y, text, ha='center', va='bottom', fontsize=26, color='black')
            
        # xTickLabel.append(miner + "\nGNN: " + str(use_gnn))
        miner_text = ""
        if group_keys == "IMbi_freq":
          miner_text = "Cost-Func"
        elif group_keys == "IMbi_rel":
          miner_text = "Reward-Func"
        elif group_keys == "IMbi_aprox":
          miner_text = "Approx-Func"
          
        if group_keys == best_miner:
          miner_text = "$\\bf{" + miner_text + "}$"
        xTickLabel.append(miner_text)
          
        idx.append(j + 2.5*boxplot_distance)
        j += 6*boxplot_distance
        
      
      axs.set_yticks(setupYTickList(minValue, 0.25))
      axs.set_xticks(idx)
      axs.set_xticklabels(xTickLabel, rotation=0, fontsize=24)
      axs.tick_params(axis='x', labelsize=24)
      axs.tick_params(axis='y', labelsize=24)
      
      
      legend_elements = [
        Line2D([0], [0], color='r', lw=4, label=r"$\operatorname{prec}$"),
        Line2D([0], [0], color='g', lw=4, label=r"$\operatorname{align{-}acc}$"),
        Line2D([0], [0], color='b', lw=4, label=r"$\operatorname{trace{-}acc}$"),
        Line2D([0], [0], color='m', lw=4, label=r"$\operatorname{align{-}F1{-}score}$"),
        Line2D([0], [0], color='c', lw=4, label=r"$\operatorname{trace{-}F1{-}score}$"),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'A $\operatorname{est{-}ub{-}acc_{\operatorname{trace}}}$', markerfacecolor='white'),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'B $\operatorname{ub{-}acc_{\operatorname{align}}}(\beta_1)$', markerfacecolor='white'),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'C $\operatorname{ub{-}acc_{\operatorname{align}}}(\beta_2)$', markerfacecolor='white'),
        # Line2D([0], [0], color='white', marker='', linestyle='', markersize=0, label=r'D $\operatorname{ub{-}acc_{\operatorname{align}}}(\beta_3)$', markerfacecolor='white'),
      ]
      scatter1 = axs.scatter(pointA_X, pointA_Y, marker=r'$\operatorname{A}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{align{-}acc}}(\beta_1)$')
      scatter2 = axs.scatter(pointB_X, pointB_Y, marker=r'$\operatorname{B}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{align{-}acc}}(\beta_2)$')
      scatter3 = axs.scatter(pointC_X, pointC_Y, marker=r'$\operatorname{C}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{align{-}acc}}(\beta_3)$')
      scatter4 = axs.scatter(pointD_X, pointD_Y, marker=r'$\operatorname{D}$', color='white', s=200, edgecolors='black', label=r'$\overline{\operatorname{trace{-}acc}}$')


      # Add the legend to the subplot
      lgd = axs.legend(handles=legend_elements + [scatter1, scatter2, scatter3, scatter4], loc='center left', bbox_to_anchor=(1.00, 0.5),prop={'size': 24})

      
      if saveFig:
        fig.savefig(os.path.join(file_path, output_file_name + "-" + logP_name +".pdf"), bbox_extra_artists=(lgd,), bbox_inches='tight')
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
  
def run_comparison(ratio_list, sup_list, result_path, parallel = True):
  df = create_df()
  lpPaths, lmPaths = get_data_paths(result_path)

  if not os.path.exists(result_path):
    os.mkdir(result_path)
  


  with open(os.path.join(result_path,"output.txt"), 'w') as txt_file:
    
    mar_better_runs = 0
    ali_better_runs = 0
    aprox_better_runs = 0
    
    ratios = ratio_list
    sups = sup_list
    
    dataList = []
    for ratio in ratios:
      for sup in sups:
        for i in range(0,len(lpPaths)):
          df_temp = create_df()
          dataList.append((df_temp, result_path, lpPaths[i][1], lmPaths[i][1], lpPaths[i][0],lmPaths[i][0], 0.2, 0.99, sup, ratio))

    start_time = time.time()

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
      print("Running in single process mode")
      warnings.filterwarnings("ignore")
      df = create_df()
      for index, input in enumerate(dataList):
        print("Running log: " + input[2] + " " + input[3])
        print("Run " + str(index) + " of " + str(len(dataList)))
        
        df_temp, _ = applyMinerToLog(*input)
        df = pd.concat([df, df_temp])
    txt_file.write("Time: " + str(time.time() - start_time) + "\n")
    return df
  
def run_comparison_fixed_supports(ratio_list, sup_dict, result_path):
  df = create_df()
  lpPaths, lmPaths = get_data_paths(result_path)
  
  print("Running in single process mode")
  warnings.filterwarnings("ignore")
  
  runs = len(lpPaths) * len(sup_dict) * len(ratio_list) * len(sup_dict)
  iterator = 1
  for data_it in range(0,len(lpPaths)):
    lp_path = lpPaths[data_it][1]
    lm_path = lmPaths[data_it][1]
    for miner, dict in sup_dict.items():
      for ratio in ratio_list:
        logName = lpPaths[data_it][0].replace(".xes","")
        sup = dict[logName]
        print("Run " + str(iterator) + " of " + str(runs))
        if miner == "IMF":
          df = applySingleMinerToLog(df, miner, result_path, lp_path, lm_path, lpPaths[data_it][0], lmPaths[data_it][0], sup, 0.0, sup, ratio)
        else:
          df = applySingleMinerToLog(df, miner, result_path, lp_path, lm_path, lpPaths[data_it][0], lmPaths[data_it][0], 0.2, 0.0, sup, ratio)
        iterator += 1
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
    
    full_name_logP = get_original_log_paths(logPName, file_path)
    full_name_logM = get_original_log_paths(logMName, file_path)
    parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
    logP = xes_importer.apply(full_name_logP, parameters = parameter)
    logM = xes_importer.apply(full_name_logM, parameters = parameter)
    
    net, im, fm = inductive_miner.apply_bi(logP,logM, variant=inductive_miner.Variants.IMbi, sup=im_bi_sup, ratio=im_bi_ratio, size_par=len(logP)/len(logM), cost_Variant=cost_Variant)
    save_vis_petri_net(net,im,fm,os.path.join(folder_path, dataFolderName, "pr_" + df_Variant + "_"  + logPName + "_" + logMName + "_sup_"+ str(im_bi_sup) + "_ratio_"+ str(im_bi_ratio) + ".png"))
  
def getBaseLineInductiveMinerDfStarArg(arg):
  return getBaseLineInductiveMinerDfStar(*arg)
  
def getBaseLineInductiveMinerDfStar(logs_name, logpath):
  df = create_df()
  warnings.filterwarnings("ignore")
  print(str(os.getpid()) + " Loading log: " + logs_name)
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  
  log = xes_importer.apply(logpath, parameters=parameter)

  noiseThreshold = 0.2
  #imf 
  print(str(os.getpid()) + " Running IMF")
  parameters = {pm4py_imf.IMFParameters.NOISE_THRESHOLD : noiseThreshold}
  pt = pm4py_algorithm.apply(log,variant=ind_Variants.IMf, parameters=parameters)
  net, im, fm = convert_to_petri_net(pt)
  print(str(os.getpid()) + " Eval IMF")
  
  df = runSingleLogEvaluation(df, log, None, "IMF", net, im, fm, logs_name, logs_name, noiseThreshold, 0, 0, 0, False) 
  print(str(os.getpid()) + " Done")
  return df
  
def getBaseLineInductiveMinerDf(df, logs_name, file_path, noise_list):
  # logs_path_root =  "C:/Users/Marko/Desktop/IMbi_Data/new-data-03-24/"
  data_folder = "comparison-data"
  logs_path_root = os.path.join(file_path,data_folder)
  
  parallel = False
  if parallel:
    num_processors_available = multiprocessing.cpu_count()
    print("Number of available processors:", num_processors_available)
    if num_processors_available > 20:
      num_processors = max(1,round(num_processors_available))
    else:
      num_processors = max(1,round(num_processors_available/2))
    print("Number of used processors:", num_processors)

    input_data = []
    for i in range(0,len(logs_name)):
      input_data.append((logs_name[i], os.path.join(logs_path_root,logs_name[i])))

    warnings.filterwarnings("ignore")
    pool_res = None
    with multiprocessing.Pool(num_processors) as pool:
        pool_res = tqdm(pool.imap(getBaseLineInductiveMinerDfStarArg, input_data),total=len(input_data))
        
        for result in pool_res:
            # Process individual evaluation result
            df = pd.concat([df, result])
            
    return df
    
  else:
    for i in range(0,len(logs_name)):
      for noise in noise_list:
        logpath = os.path.join(logs_path_root,logs_name[i])
        print("Running log: " + logs_name[i])
        parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
        
        log = xes_importer.apply(logpath, parameters=parameter)

        noiseThreshold = noise
        #imf 
        print("Running IMF")
        parameters = {pm4py_imf.IMFParameters.NOISE_THRESHOLD : noiseThreshold}
        pt = pm4py_algorithm.apply(log,variant=ind_Variants.IMf, parameters=parameters)
        net, im, fm = convert_to_petri_net(pt)
        print("Eval IMF")
        
        df = runSingleLogEvaluation(df, log, None, "IMF", net, im, fm, logs_name[i], logs_name[i], noiseThreshold, 0, 0, 0, False) 
      
  return df

def plot_line_chart(df, highlight_support, saveFig, file_path):
  import matplotlib.font_manager as fm
  custom_font = fm.FontProperties(family='Arial', size=24)
  
  df.reset_index(drop=True, inplace=True)
  
  if highlight_support == True:
    df = df.drop(["logM_Name"], axis=1)
    df_group = df.groupby(by=["logP_Name"], group_keys=True)
  else:
    df_group = df.groupby(by=["logP_Name",	"logM_Name"], group_keys=True)
  
  
  use_upper_bound = False
  output_file_name = "plot_linechart"

  for group_keys, group_df in df_group:
    logP_name = group_df['logP_Name'].iloc[0]
    logP_name = logP_name.replace(".xes", "")
    logP_name_org = get_original_log_paths(logP_name, file_path)
    logM_name_org = None
    if highlight_support == False:
      logM_name = group_df['logM_Name'].iloc[0]
      logM_name = logM_name.replace(".xes", "")
      logM_name_org = get_original_log_paths(logM_name, file_path)
    ubs_align = None
    ub_trace = None
      
    if use_upper_bound and logP_name_org != None and logM_name_org != None:
      ubs_align = ubc.run_upper_bound_align_on_logs_upper_bound_trace_distance(logP_name_org, logM_name_org)
      ub_trace = ubc.run_upper_bound_traces_on_logs(logP_name_org, logM_name_org)
      
    df_group_on_miners = group_df.groupby(by=["miner"], group_keys=True)
    for group_keys, group_df_miner in df_group_on_miners:
      miner = ""
      if group_keys == "IMbi_freq":
        miner = "Cost-Func"
      elif group_keys == "IMbi_rel":
        miner = "Reward-Func"
      elif group_keys == "IMbi_aprox":
        miner = "Approx-Func"
      else:
        miner = "IMF"
      
      
      fig, ax = plt.subplots(figsize=(14 , 12))
      # fig.tight_layout(pad=18.0)
      # print(group_df_miner)
      if highlight_support == False:
        ax.set_title(r"$L^+$: " + logP_name + r"\n$L^-$: " + logM_name, fontproperties=custom_font)
        ax.set_xlabel('Ratio Values', fontproperties=custom_font)
      else:
        if miner == "IMF":
          ax.set_title(r"$L^+$: " + logP_name + "\nMiner : " + miner, fontproperties=custom_font)
        else:
          ax.set_title(r"$L^+$: " + logP_name + "\nCut-Evaluation function: " + miner, fontproperties=custom_font)
        ax.set_xlabel('Support Values', fontproperties=custom_font)
        
      boxplot_width = 0.3
      boxplot_distance = 0.4

      if miner == "IMF":
        support_values = group_df_miner['imf_noise_thr'].tolist()
      else:
        support_values = group_df_miner['im_bi_sup'].tolist()
      precision_values = group_df_miner['precP'].tolist()
      fitness_values = group_df_miner['fitP-Align'].tolist()

      from scipy.stats import hmean
      harmonic_means = [hmean([precision_values[i], fitness_values[i]]) for i in range(len(precision_values))]
      max_index = harmonic_means.index(max(harmonic_means))
      print("Miner: " + miner + " | LogP: " + logP_name + " BEST Support: " + str(support_values[max_index]))
      
      ax.plot(support_values, precision_values, 'r-', label='Precision')
      ax.plot(support_values, precision_values, 'rx')
      ax.plot(support_values, fitness_values, 'b-', label='Fitness')
      ax.plot(support_values, fitness_values, 'bx')
      ax.plot(support_values, harmonic_means, 'g-', label='Harmonic Means')
      ax.plot(support_values, harmonic_means, 'gx')
      ax.grid(True)
      
      # Adding legend
      legend_elements = [
        Line2D([0], [0], color='r', lw=4, label=r"$\operatorname{prec(L^+, M)}$"),
        Line2D([0], [0], color='b', lw=4, label=r"$\operatorname{align{-}fit(L^+, M)}$"),
        Line2D([0], [0], color='g', lw=4, label=r"$\operatorname{harmonic{-}mean}$")
      ]
      
      ax.legend(handles=legend_elements, prop={'size': 20})
      ax.set_ylabel('Values', fontproperties=custom_font)
      
      ax.set_ylim(0.0, 1.05)
      ax.set_xlim(0.0, 1.05)
      # Set y-axis ticks with increments of 0.1
      ax.set_yticks([i/10 for i in range(11)])
      ax.set_yticklabels([i/10 for i in range(11)], fontsize=20)
      
      ax.set_xticks(support_values)
      ax.set_xticklabels(support_values,fontsize=20)
   
      if ubs_align != None and ub_trace != None:
        # ubs trace
        # Add a horizontal dotted line above the specific bar
        axs.hlines(ub_trace, xmin=j+2*boxplot_distance - boxplot_width/2, xmax=j+2*boxplot_distance + boxplot_width/2, colors='k', linestyles='dotted', linewidth=4)
        text_x = j + 2*boxplot_distance   # Adjust x-coordinate as needed
        text_y = ub_trace + 0.02  # Adjust y-coordinate as needed
        pointD_X.append(text_x)
        pointD_Y.append(text_y)
        # axs.text(text_x, text_y, text, ha='center', va='bottom', fontsize=26, color='black')

        
        for enum, ub_align in ubs_align.items():
          # ubs align
          # Add a horizontal dotted line above the specific bar
          axs.hlines(ub_align, xmin=j+ boxplot_distance - boxplot_width/2, xmax=j+boxplot_distance + boxplot_width/2, colors='k', linestyles='dotted', linewidth=4)
          text_x = j + boxplot_distance  # Adjust x-coordinate as needed
          text_y = ub_align + 0.02  # Adjust y-coordinate as needed
          if enum == ubc.ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE:
            pointA_X.append(text_x)
            pointA_Y.append(text_y)
          if enum == ubc.ShorestModelPathEstimation.ALLOW_LONGEST_SEQUENCE_PART:
            pointB_X.append(text_x)
            pointB_Y.append(text_y)
          if enum == ubc.ShorestModelPathEstimation.ALLOW_MIN_TRACE_LENGTH:
            pointC_X.append(text_x)
            pointC_Y.append(text_y)

          # axs.text(text_x, text_y, text, ha='center', va='bottom', fontsize=26, color='black')
          
          
      file_path_folder = os.path.join(file_path, output_file_name)
      if not os.path.exists(file_path_folder):
        os.mkdir(file_path_folder)
      
      if not os.path.exists(os.path.join(file_path_folder, logP_name)):
        os.mkdir(os.path.join(file_path_folder, logP_name))
        
          
      if saveFig:
        fig.savefig(os.path.join(file_path_folder, logP_name,  miner +".pdf"))
      else:
        plt.show()




  
def get_comparison_df(result_path):
  get_sup_parameter = False
  
  if get_sup_parameter:
    csv_filename = "comparison.csv"
    ratio_list = [0]
    sup_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  
    if not os.path.exists(os.path.join(result_path,csv_filename)):
      df = run_comparison(ratio_list, sup_list, result_path, parallel=False)
      save_df(df, os.path.join(result_path,csv_filename))
    else:
      df = pd.read_csv(os.path.join(result_path,csv_filename),index_col=0)

    csv_filename2 = "comparison2.csv"
    if not os.path.exists(os.path.join(result_path,csv_filename2)):
      df2 = create_df()
      logs_name = ["BPIC12-A-LP.xes", "BPIC17-A-LP.xes", "RTFM-LP.xes"]
      noise_list = sup_list
      df2 = getBaseLineInductiveMinerDf(df2, logs_name, result_path, noise_list)
      save_df(df2, os.path.join(result_path,csv_filename2))
    else:
      df2 = pd.read_csv(os.path.join(result_path,csv_filename2),index_col=0)

    df_combined = pd.concat([df, df2])
    
    plot_line_chart(df_combined, get_sup_parameter, saveFig=True, file_path=result_path)
  else:
    csv_filename = "comparison_ratio.csv"
    ratio_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    miner_sup_dic = {}
    miner_sup_dic["IMbi_freq"] = {"BPIC12-A-LP": 0.1, "BPIC17-A-LP": 0.4, "RTFM-LP": 0.2}
    miner_sup_dic["IMbi_rel"] = {"BPIC12-A-LP": 0.6, "BPIC17-A-LP": 0.1, "RTFM-LP": 0.2}
    miner_sup_dic["IMbi_aprox"] = {"BPIC12-A-LP": 0.1, "BPIC17-A-LP": 0.1, "RTFM-LP": 0.9}
    miner_sup_dic["IMF"] = {"BPIC12-A-LP": 0.1, "BPIC17-A-LP": 0.1, "RTFM-LP": 0.2}
  
    if not os.path.exists(os.path.join(result_path,csv_filename)):
      df = run_comparison_fixed_supports(ratio_list, miner_sup_dic, result_path)
      save_df(df, os.path.join(result_path,csv_filename))
    else:
      df = pd.read_csv(os.path.join(result_path,csv_filename),index_col=0)

    # plot_line_chart(df_combined, get_sup_parameter, saveFig=True, file_path=result_path)
  
  
  if False:
    # displayDoubleLogSplit(df, saveFig=True, file_path=result_path)
    use_single_best = False
    # save_petri_nets(df, result_path)
    if use_single_best:
      df = filter_df_for_best_models(df)
      save_petri_nets(df, result_path)
      displayDoubleLogSplitSingleBest_TrueSplit(df, saveFig=True, file_path=result_path)
    else:
      displayDoubleLogSplitBoxplot_TrueSplit(df, saveFig=True, file_path=result_path)
    

    
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
  rootPath = "C:/Users/Marko/Desktop/IMbi_Data/new-data-03-24/"
  lpNames = ["RTFM-SAMP-LP.xes", "2017_O_lp.xes"]
  lMNames = ["2012_O_lm.xes", "2017_O_lm.xes"]
  
  create_petri_net_model("petriNet1",result_path, rootPath + lpNames[0], rootPath + lMNames[0], 0.2, 0.5, custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE)
   
if __name__ == '__main__':
  data_path = os.path.join(root_path, "analysing_cost_functions")
  result_path = os.path.join(data_path, "results")
  
  # generate_manual_models(result_path)
  time_cur = time.time()
  df = get_comparison_df(result_path)
  print("Time: " + str(time.time() - time_cur))


  # df_measurement = filter_and_sort_dataframe(df, 10, get_mar_improved = True)
  # create_comparison_petri_nets(result_path, "lp_2017_f.xes","lm_2017_f.xes",0.4,0.8)

  
