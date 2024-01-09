import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

from pm4py.objects.log.importer.xes import importer as xes_importer
from local_pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.util import xes_constants
from pm4py.objects.log import obj as log_instance
from pm4py.objects.log.obj import EventLog
import warnings
import numpy as np
import ot 
from enum import Enum
from Levenshtein import distance as levenshtein_distance
import networkx as nx

class ShorestModelPathEstimation(Enum):
    Worst_CASE_ALLOW_EMPTY_TRACE = 0
    ALLOW_MIN_TRACE_LENGTH = 1
    ALLOW_AVERAGE_TRACE_LENGTH = 2
    ALLOW_LONGEST_SEQUENCE_PART = 3

def levenshtein_distance_no_substitution(s1, s2):
  len_s1, len_s2 = len(s1), len(s2)

  # Create a 2D array to store the edit distances
  dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

  # Initialize the first row and column
  for i in range(len_s1 + 1):
      dp[i][0] = i
  for j in range(len_s2 + 1):
      dp[0][j] = j

  # Fill in the dynamic programming table
  for i in range(1, len_s1 + 1):
      for j in range(1, len_s2 + 1):
          if s1[i - 1] != s2[j - 1]:
              # Exclude substitution, only consider insertion and deletion
              dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
          else:
              dp[i][j] = dp[i - 1][j - 1]

  # The Levenshtein distance is the value in the bottom-right cell of the matrix
  return dp[len_s1][len_s2]

def levenshtein_distance_no_substitution_import(s1, s2):
  distance = levenshtein_distance(s1, s2,weights=(1,1,len(s1) + len(s2) + 1))
  return distance

def get_distance_matrix_edit_distance(trace_variants):
    array = np.zeros((len(trace_variants), len(trace_variants)))
    for i in range(len(trace_variants)):
        for j in range(len(trace_variants)):
            array[i][j] = levenshtein_distance_no_substitution_import(trace_variants[i], trace_variants[j])
    return array

def get_distance_matrix_upper_bound(trace_variants,set_traces_P, set_traces_M, shortest_model_estimation):
  array = {}
  for key in shortest_model_estimation.keys():
    array[key] = np.zeros((len(trace_variants), len(trace_variants)))

  for i in range(len(trace_variants)):
    for j in range(len(trace_variants)):
      res_upper_bound = {}
      if trace_variants[i] in set_traces_P and trace_variants[j] in set_traces_M:
        res_upper_bound = get_align_uppper_bound_for_trace(trace_variants[i], trace_variants[j], shortest_model_estimation)
      elif trace_variants[i] in set_traces_M and trace_variants[j] in set_traces_P:
        res_upper_bound = get_align_uppper_bound_for_trace(trace_variants[j], trace_variants[i], shortest_model_estimation)
      else:
        for key in shortest_model_estimation.keys():
          res_upper_bound[key] = 0
        
      for key, value in res_upper_bound.items():
        array[key][i][j] = value
  return array
  
def emd_distance_pyemd(trace_frequency_1,trace_frequency_2, distance_matrix):
    trace_frequency_1 = np.array(trace_frequency_1)
    trace_frequency_2 = np.array(trace_frequency_2)
  
    transport_plan  = ot.emd(trace_frequency_1, trace_frequency_2, distance_matrix, numThreads=4, numItermax=1000000)
    total_transport_cost = np.sum(transport_plan * distance_matrix)
    #return cost_lp, distamce_df_emd
    return total_transport_cost, transport_plan

def artificial_start_end(log):
    st = 'start'
    en = 'end'
    activity_key = xes_constants.DEFAULT_NAME_KEY
    start_event = log_instance.Event()
    start_event[activity_key] = st
    
    end_event = log_instance.Event()
    end_event[activity_key] = en

    for trace in log:
        trace.insert(0, start_event)
        trace.append(end_event)
    return log

def load_log(file_path):
  parameter = {xes_importer.iterparse_20.Parameters.SHOW_PROGRESS_BAR: False}
  log = xes_importer.apply(file_path, parameters=parameter)
  return log

def get_dfg_from_log(log):
  dfg = {}
  for trace in log:
    found_edges = set()
    last_activity = None
    for activity in trace:
      if last_activity is not None:
        edge = (last_activity[xes_constants.DEFAULT_NAME_KEY], activity[xes_constants.DEFAULT_NAME_KEY])
        if edge in found_edges:
          continue
        if edge in dfg:
          dfg[edge] += 1
        else:
          dfg[edge] = 1
        found_edges.add(edge) 
      last_activity = activity
  return dfg

def get_relative_traces_from_log(log):
  trace_dict = {}
  for trace in log:
    trace_tuple = tuple([event[xes_constants.DEFAULT_NAME_KEY] for event in trace])
    if trace_tuple in trace_dict:
      trace_dict[trace_tuple] += 1
    else:
      trace_dict[trace_tuple] = 1
      
  sum_values = sum(trace_dict.values())
  
  for key, value in trace_dict.items():
    trace_dict[key] = value / sum_values
  return trace_dict

def get_min_trace_length_from_log(log):
  min = sys.maxsize
  for trace in log:
    trace_tuple = tuple([event[xes_constants.DEFAULT_NAME_KEY] for event in trace])
    min = len(trace_tuple) if len(trace_tuple) < min else min
  return min

def generate_nx_graph_from_dfg(dfg):
    dfg_acts = set()
    for x in dfg:
        dfg_acts.add(x[0])
        dfg_acts.add(x[1])
    G = nx.DiGraph()
    for act in dfg_acts:
        G.add_node(act)
    for edge in dfg:
        G.add_edge(edge[0], edge[1])
    return G

def get_longest_Sequence_part(log):
  log_art = artificial_start_end(log.__deepcopy__())
  dfg_art = dfg_discovery.apply(log_art, variant=dfg_discovery.Variants.FREQUENCY)
  nx_graph_art = generate_nx_graph_from_dfg(dfg_art)
  shortest_path = nx.shortest_path(nx_graph_art, source="start", target="end")
  # 2 because of artificial start and end
  result = len(shortest_path) - 2
  return result

def get_small_subset_of_dict(dic, size):
  res_dic = {}
  for key, value in dic.items():
    res_dic[key] = value
    if len(res_dic) == size:
      break
  return res_dic

def remove_traces_with_edge_from_log(log, edge):
  log_filtered = log.__deepcopy__()
  log_res = EventLog()
  for trace in log_filtered:
    can_include = True
    for i in range(len(trace)-1):
      
      from_concept_name = trace[i][xes_constants.DEFAULT_NAME_KEY]
      to_concept_name = trace[i+1][xes_constants.DEFAULT_NAME_KEY]
      
      if from_concept_name == edge[0] and to_concept_name == edge[1]:
        can_include = False
        break
    
    if can_include:
      log_res.append(trace)
      
      
  return log_res

def transform_dfg_to_relative(dfg, sum_values):
  dfg_relative = {}
  for edge, value in dfg.items():
    dfg_relative[edge] = value / sum_values
  return dfg_relative

def get_union_trace_variants(variants_P, variants_M):
  union = set()
  for variant in variants_P:
    union.add(variant)
  for variant in variants_M:
    union.add(variant)
  return list(union)

def get_relative_trace_frequency_from_union(union, variants):
  relative_trace_frequency = []
  for variant in union:
    if variant in variants:
      relative_trace_frequency.append(variants[variant])
    else:
      relative_trace_frequency.append(0)
  return relative_trace_frequency

def get_maximum_gain_edge(logP, logM, sum_valueP, sum_valueM):
  
  dfgP = transform_dfg_to_relative(get_dfg_from_log(logP), sum_valueP)
  dfgM = transform_dfg_to_relative(get_dfg_from_log(logM), sum_valueM)
  max_gain_edge = None
  max_gain_value = 0
  for edge, value in dfgM.items():
    if edge in dfgP:
      if dfgP[edge] < value:
        gain = value - dfgP[edge]
        if gain > max_gain_value:
          max_gain_value = gain
          max_gain_edge = edge
    else:
      gain = value
      if gain > max_gain_value:
        max_gain_value = gain
        max_gain_edge = edge
  return max_gain_edge

def run_upper_bound_traces_on_logs(log_P_path, log_m_path):
  print("Loading logs")
  log_P = load_log(log_P_path)
  log_M = load_log(log_m_path)
  log_P = artificial_start_end(log_P.__deepcopy__())
  log_M = artificial_start_end(log_M.__deepcopy__())
  
  original_p_size = len(log_P)
  original_m_size = len(log_M)

  print("Running trace upper bound calculation")
  while True:
    max_gain_edge = get_maximum_gain_edge(log_P, log_M, original_p_size, original_m_size)
    if max_gain_edge is None:
      break
    log_P = remove_traces_with_edge_from_log(log_P, max_gain_edge)
    log_M = remove_traces_with_edge_from_log(log_M, max_gain_edge)
    # print("log_P: ", len(log_P), " log_M: ", len(log_M))
    
  fit_p = len(log_P) / original_p_size
  fit_m = len(log_M) / original_m_size
  upper_bround = fit_p - fit_m
  return upper_bround

def parse_result_emd(trace_variants,distamce_df_emd, relative_trace_dic_P, relative_trace_dic_M):
  matrix_length = len(distamce_df_emd)
  result = []
  for i in range(matrix_length):
    if trace_variants[i] in relative_trace_dic_P:
      for j in range(matrix_length):
        if trace_variants[j] in relative_trace_dic_M:
          if distamce_df_emd[i][j] > 0:
            result.append(((trace_variants[i], trace_variants[j]),distamce_df_emd[i][j]))
        else:
          if distamce_df_emd[i][j] > 0:
            raise Exception("Wrong flow")
    else:
      for j in range(matrix_length):
        if distamce_df_emd[i][j] > 0:
          raise Exception("Wrong flow")

  return result

def get_align_uppper_bound_for_trace(trace1, trace2, shortest_model_estimation = {}):
  result = {}
  distance = levenshtein_distance_no_substitution_import(trace1, trace2)
  if len(shortest_model_estimation) == 0:
    shortest_model_estimation.update({ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE:0})
  
  s = len(trace2) - len(trace1)
  for estimate, value in shortest_model_estimation.items():
    gamma = 0
    if estimate == ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE:
      gamma = 0
    elif estimate == ShorestModelPathEstimation.ALLOW_MIN_TRACE_LENGTH:
      gamma = value
    elif estimate == ShorestModelPathEstimation.ALLOW_AVERAGE_TRACE_LENGTH:
      gamma = value
    elif estimate == ShorestModelPathEstimation.ALLOW_LONGEST_SEQUENCE_PART:
      gamma = value
    else:
      raise Exception("Unknown shortest model estimation")
    
    nominator = 0.0
    denominator = 0.0
    if s < 0:
      nominator = -s * distance
      # -2 because of artificial start and end
      denominator = (len(trace2) - 2 + gamma)
    else:
      nominator = distance
      # -2 because of artificial start and end
      denominator = len(trace2) - 2 + gamma
      
    nominator = min(nominator, denominator)
      
    result[estimate] = nominator / denominator

  return result
  

def print_result(result):
  for i in range(len(result)):
    print(result[i][0][0], " with ", result[i][1], " -> ", result[i][0][1])
  print()

def run_upper_bound_align_on_logs_edit_distance(log_P_path, log_m_path, shortest_model_estimation = {}):
  log_P = load_log(log_P_path)
  log_M = load_log(log_m_path)
  log_P_art = artificial_start_end(log_P.__deepcopy__())
  log_M_art = artificial_start_end(log_M.__deepcopy__())
  
  
  relative_trace_dic_P = get_relative_traces_from_log(log_P_art)
  relative_trace_dic_M = get_relative_traces_from_log(log_M_art)
  
  trace_variants = get_union_trace_variants(relative_trace_dic_P.keys(), relative_trace_dic_M.keys())
  
  trace_frequency_1 = get_relative_trace_frequency_from_union(trace_variants, relative_trace_dic_P)
  trace_frequency_2 = get_relative_trace_frequency_from_union(trace_variants, relative_trace_dic_M)
  
  distance_matrix = get_distance_matrix_edit_distance(trace_variants)
  
  _, distamce_df_emd = emd_distance_pyemd(trace_frequency_1,trace_frequency_2,distance_matrix)
  
  result_emd = parse_result_emd(trace_variants, distamce_df_emd, relative_trace_dic_P, relative_trace_dic_M)

  if len(shortest_model_estimation) == 0:
    shortest_model_estimation[ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE] = 0
    min_trace_length = get_min_trace_length_from_log(log_P)
    shortest_model_estimation[ShorestModelPathEstimation.ALLOW_MIN_TRACE_LENGTH] = min_trace_length
    longest_Sequence =  get_longest_Sequence_part(log_P)
    shortest_model_estimation[ShorestModelPathEstimation.ALLOW_LONGEST_SEQUENCE_PART] = longest_Sequence
    
  result = {}
  for (trace1, trace2), distance in result_emd:
    res_estimates_t1_t2 = get_align_uppper_bound_for_trace(trace1, trace2, shortest_model_estimation)
    for estimate, value in res_estimates_t1_t2.items():
      if estimate in result:
        result[estimate] += distance * value
      else:
        result[estimate] = distance * value
        
  return result
    
# Current state of the art upper bound calculation
def run_upper_bound_align_on_logs_upper_bound_trace_distance(log_P_path, log_m_path, shortest_model_estimation = {}):
  # Data preparation
  print("Loading logs")
  log_P = load_log(log_P_path)
  log_M = load_log(log_m_path)
  log_P_art = artificial_start_end(log_P.__deepcopy__())
  log_M_art = artificial_start_end(log_M.__deepcopy__())
  
  
  relative_trace_dic_P = get_relative_traces_from_log(log_P_art)
  relative_trace_dic_M = get_relative_traces_from_log(log_M_art)
  
  trace_variants = get_union_trace_variants(relative_trace_dic_P.keys(), relative_trace_dic_M.keys())
  
  trace_frequency_1 = get_relative_trace_frequency_from_union(trace_variants, relative_trace_dic_P)
  trace_frequency_2 = get_relative_trace_frequency_from_union(trace_variants, relative_trace_dic_M)
  
  print("Running align upper bound calculation")
  # Create distance matrix
  if len(shortest_model_estimation) == 0:
    shortest_model_estimation[ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE] = 0
    min_trace_length = get_min_trace_length_from_log(log_P)
    shortest_model_estimation[ShorestModelPathEstimation.ALLOW_MIN_TRACE_LENGTH] = min_trace_length
    longest_Sequence =  get_longest_Sequence_part(log_P)
    shortest_model_estimation[ShorestModelPathEstimation.ALLOW_LONGEST_SEQUENCE_PART] = longest_Sequence
    
  distances_matrix_upper_bound = get_distance_matrix_upper_bound(trace_variants,set(relative_trace_dic_P.keys()),set(relative_trace_dic_M.keys()), shortest_model_estimation)
  
  upper_bound_result_dict = {}
  for estimation, distance_matrix in distances_matrix_upper_bound.items():
    upper_bound, _ = emd_distance_pyemd(trace_frequency_1, trace_frequency_2, distance_matrix)
    upper_bound_result_dict[estimation] = upper_bound

  return upper_bound_result_dict
   
  
  
def show_EMD_correctness():
  print("Running correctness test")
  print()
  relative_trace_dic_P = {("Start","a","b","End"):0.2, ("Start","a","c","End"):0.5, ("Start","f","End"):0.3}
  relative_trace_dic_M = {("Start","b","End"):0.3, ("Start","a","End"):0.3, ("Start","c","End"):0.1, ("Start","f","End"):0.3}
  
  # relative_trace_dic_P = {("Start","a","b","c","End"):1.0}
  # relative_trace_dic_M = {("Start","a","b","c","d","End"):1.0}
  
  print("Edit distance result: ")
  trace1 = ("Start","a","b","f","End")
  trace2 = ("Start","b", "c","End")
  distance = levenshtein_distance_no_substitution(trace1, trace2)
  print(trace1, " with distance", distance, " -> ", trace2)
  distance = levenshtein_distance_no_substitution_import(trace1, trace2)
  print(trace1, " with distance_import", distance, " -> ", trace2)
  print()
  
  
  trace_variants = get_union_trace_variants(relative_trace_dic_P.keys(), relative_trace_dic_M.keys())
  trace_frequency_1 = get_relative_trace_frequency_from_union(trace_variants, relative_trace_dic_P)
  trace_frequency_2 = get_relative_trace_frequency_from_union(trace_variants, relative_trace_dic_M)
  
  distance_matrix = get_distance_matrix_edit_distance(trace_variants)
  _, distamce_df_emd = emd_distance_pyemd(trace_frequency_1,trace_frequency_2,distance_matrix)
  result_emd = parse_result_emd(trace_variants, distamce_df_emd, relative_trace_dic_P, relative_trace_dic_M)
  print("EMD distribution result: ")
  print("Log p: ", relative_trace_dic_P)
  print("Log m: ", relative_trace_dic_M)
  print()
  print_result(result_emd)

  shortest_model_estimation = {ShorestModelPathEstimation.Worst_CASE_ALLOW_EMPTY_TRACE:0}
  print()
  print("Running align upper bound calculation")
  result = {}
  for (trace1, trace2), distance in result_emd:
    res_estimates_t1_t2 = get_align_uppper_bound_for_trace(trace1, trace2, shortest_model_estimation)
    for estimate, value in res_estimates_t1_t2.items():
      if estimate in result:
        result[estimate] += distance * value
      else:
        result[estimate] = distance * value
        
  print("Align upper bound for result", result)
  
  print("Running align upper bound calculation directly")
  distances_matrix_upper_bound = get_distance_matrix_upper_bound(trace_variants, shortest_model_estimation)
  upper_bound_result_dict = {}
  for estimation, distance_matrix in distances_matrix_upper_bound.items():
    upper_bound, _ = emd_distance_pyemd(trace_frequency_1, trace_frequency_2, distance_matrix)
    upper_bound_result_dict[estimation] = upper_bound
  print("Align upper bound for result", upper_bound_result_dict)
  


def run_upper_bound_traces():
  rootPath = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  lpNames = ["2012_O_lp.xes", "2017_O_lp.xes"]
  lMNames = ["2012_O_lm.xes", "2017_O_lm.xes"]
  
  for i in range(len(lpNames)):
    log_P_path = os.path.join(rootPath, lpNames[i])
    log_M_path = os.path.join(rootPath, lMNames[i])
    print("Running trace upper bound on data ", lpNames[i], " and ", lMNames[i])
    upper_bound = run_upper_bound_traces_on_logs(log_P_path, log_M_path)
    print("Trace upper bound for ", lpNames[i], " and ", lMNames[i], " is ", upper_bound)
    print()
  
     
def run_upper_bound_alignment():
  rootPath = "C:/Users/Marko/Desktop/IMbi_Data/FilteredLowActivity/"
  lpNames = ["2012_O_lp.xes", "2017_O_lp.xes"]
  lMNames = ["2012_O_lm.xes", "2017_O_lm.xes"]
  
  for i in range(len(lpNames)):
    log_P_path = os.path.join(rootPath, lpNames[i])
    log_M_path = os.path.join(rootPath, lMNames[i])
    print("Running align upper bound on data ", lpNames[i], " and ", lMNames[i])
    upper_bound = run_upper_bound_align_on_logs_upper_bound_trace_distance(log_P_path, log_M_path)
    print("Align upper bound for ", lpNames[i], " and ", lMNames[i], " is:")
    for key, value in upper_bound.items():
      print(value, " for ", key)
    print()
    
    # upper_bound = run_upper_bound_align_on_logs_edit_distance(log_P_path, log_M_path)
    # print("Align upper bound edit distance for ", lpNames[i], " and ", lMNames[i], " is:")
    # for key, value in upper_bound.items():
    #   print(value, " for ", key)
    # print()


if __name__ == '__main__':
  folder_path = os.path.join(root_path, "upperBoundCalculation")
  result_path = os.path.join(folder_path, "results")
  
  warnings.filterwarnings("ignore")
  
  # show_EMD_correctness()
  # run_upper_bound_alignment()
  run_upper_bound_traces()
  
  
  
  
  
