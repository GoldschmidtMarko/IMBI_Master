import collections
from copy import copy
import time
import sys
import os

root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
sys.path.append(root_path)

from pm4py.algo.discovery.dfg.utils.dfg_utils import infer_start_activities, infer_end_activities
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import util as pmutil
from local_pm4py.algo.discovery.inductive.variants.im_bi.util import splitting as split
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.util import exec_utils
from pm4py.util import constants
from enum import Enum
from pm4py.objects.log import obj as log_instance
from pm4py.util import xes_constants
from local_pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import networkx as nx
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.discovery.dfg.utils.dfg_utils import get_activities_from_dfg
from GNN_partitioning.GNN_Model_Generation.uni import gnn_models as gnn_models_uni
from GNN_partitioning.GNN_Model_Generation.bi import gnn_models as gnn_models_bi

import logging 

from local_pm4py.algo.analysis import dfg_functions
from local_pm4py.algo.analysis import custom_enum
import copy
from collections import Counter

# TODO delete debuging code
from pm4py import save_vis_dfg
from pm4py import view_dfg
from tqdm import tqdm

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
def generate_nx_graph_from_dfg(dfg):
    dfg_acts = set()
    for x in dfg:
        dfg_acts.add(x[0])
        dfg_acts.add(x[1])
    G = nx.DiGraph()
    for act in dfg_acts:
        G.add_node(act)
    for edge in dfg:
        if dfg[edge] > 0:
            G.add_edge(edge[0], edge[1], weight=dfg[edge])
    return G

def get_loop_data(dfg,A,B):
    start_A = set([x[1] for x in dfg if ((x[0] == 'start') and (x[1] in A))])
    end_A = set([x[0] for x in dfg if (x[0] in A and (x[1] == 'end'))])
    start_B = set([x[1] for x in dfg if (x[1] in B and (x[0] == 'start'))])
    input_B = set([x[1] for x in dfg if ((x[0] not in B) and (x[1] in B))])
    output_B = set([x[0] for x in dfg if ((x[0] in B) and (x[1] not in B))])

    return {'start_A':start_A,'end_A':end_A,'start_B':start_B,'input_B':input_B,'output_B':output_B}


def get_loop_tau_data(dfg, dfgM, A, B,log,logM, log_art, logM_art, activity_key, activities, activitiesM,netP,netM):
    cut_function_util_dataP = dict()
    cut_function_util_dataM = dict()
    
    dic_indirect_follow_logP = get_indirect_follow_dic(log_art, activity_key, list(activities.keys()))
    dic_indirect_follow_logM = get_indirect_follow_dic(logM_art, activity_key, list(activitiesM))
    cut_function_util_dataP.update({'dic_indirect_follow_log': dic_indirect_follow_logP})
    cut_function_util_dataM.update({'dic_indirect_follow_log': dic_indirect_follow_logM})
    
    bias_lP = get_bias_l(log, activity_key)
    bias_lM = get_bias_l(logM, activity_key)

    cut_function_util_dataP.update({'bias_l': bias_lP})
    cut_function_util_dataM.update({'bias_l': bias_lM})
    
    cut_function_util_dataP.update(get_loop_data(dfg,A,B))
    cut_function_util_dataM.update(get_loop_data(dfgM,A,B))
    
    cut_function_util_dataP.update({"A" : A})
    cut_function_util_dataM.update({"A" : A})
    cut_function_util_dataP.update({"B" : B})
    cut_function_util_dataM.update({"B" : B})
    cut_function_util_dataP.update({"net": netP})
    cut_function_util_dataM.update({"net": netM})
    
    
    return cut_function_util_dataP, cut_function_util_dataM
    


def get_indirect_follow_dic(log, activity_key : str, activities : "list[str]") -> "dict[str, dict[str, int]]":
    """
    Calculates the strictly indirectly follow count of activites in a log

    Parameters
    ----------
    log
        Log
    activity_key
        Activity key for the activity e.g. concept:name
    activities
        List of all activities

    Returns
    ----------
    dic
        Dictionary of dictionary, (activity_1 : ( activity_2 : int ) ), cardinality where activity_2 strictly indirectly follows activity_1
    """
    
    dic = {}
    # debug_activities = []
    
    if "start" not in activities:
        activities.append("start")
    if "end" not in activities:
        activities.append("end")
    
    # initialize dic with 0
    for x in activities:
        for y in activities:
            if not x in dic:
                dic[x] = {y : 0}
            else:
                dic[x][y] = 0

    for trace in log:
        explored_activities = []
        last_checked_activity = None
        checking_activity = None
        # debug_activity = []
        
        for trace_dic in trace:
            if activity_key in trace_dic:
                last_checked_activity = checking_activity
                checking_activity = trace_dic[activity_key]
                # debug_activity.append(checking_activity)
                
                for activity in explored_activities:
                    dic[activity][checking_activity] +=1
                
                if last_checked_activity != None:
                    if last_checked_activity not in explored_activities:
                        explored_activities.append(last_checked_activity)
                        continue
        # debug_activities.append(debug_activity)
    # print(debug_activities)
    # print(dic)
    return dic
   
def get_dic_activities(log, activity_key : str) -> "dict[str, int]":
    """
    Returns a dic with each activity and their occurrence 

    Parameters
    ----------
    log
        Log
    activity_key
        Activity key for the activity e.g. concept:name

    Returns
    ----------
    dic
        Dictinary str, int
    """
    
    dic = {}
    for trace in log:
        for trace_dic in trace:
            if activity_key in trace_dic:
                activity = trace_dic[activity_key]
                if activity not in dic:
                    dic[activity] = 1
                else:
                    dic[activity] += 1
    return dic

def average(lst : "list[float]") -> float:
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def get_bias_l(log, activity_key : str) -> float:
    dic_activities = get_dic_activities(log, activity_key)
    trace_lengths = []
    for trace in log:
        trace_lengths.append(len(trace))
    res = max(average(trace_lengths) - len(dic_activities.keys()),0)
    if len(dic_activities.keys()) == 0:
        return 1
    res = min(1,res/len(dic_activities.keys()))
    return res
   

def repetition_Factor(log, activity_key) -> float:
    dic_activities = get_dic_activities(log, activity_key)
    
    numberTraces = len(log)
    totalNumberEvents = sum(dic_activities.values())
    numberActivities = len(dic_activities.keys())
    if numberActivities == 0 or totalNumberEvents == 0:
        return 1
    
    return (numberTraces / (totalNumberEvents / numberActivities))

def show_dfg(log):
    parameters = {}
    start_act_cur_dfg = start_activities_get.get_start_activities(log, parameters=parameters)
    end_act_cur_dfg = end_activities_get.get_end_activities(log, parameters=parameters)
    cur_dfg = dfg_inst.apply(log, parameters=parameters)
    view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)
    
def save_dfg(log, filename):
    parameters = {}
    start_act_cur_dfg = start_activities_get.get_start_activities(log, parameters=parameters)
    end_act_cur_dfg = end_activities_get.get_end_activities(log, parameters=parameters)
    cur_dfg = dfg_inst.apply(log, parameters=parameters)
    save_vis_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg, filename)

def combine_score_values(scoreP, scoreM, cost_Variant, ratio, size_par):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return scoreP - ratio * size_par * scoreM
    else:
        # res = scoreP - ratio * scoreM
        res = scoreP - ratio * scoreM
        return res

def get_score_for_cut_type(log, logM, A, B, cut_Type, sup, ratio, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):
    log_art = artificial_start_end(log.__deepcopy__())
    logM_art = artificial_start_end(logM.__deepcopy__())
    
    dfg2 = dfg_discovery.apply(log_art, variant=dfg_discovery.Variants.FREQUENCY)
    netP = generate_nx_graph_from_dfg(dfg2)
    dfg2M = dfg_discovery.apply(logM_art, variant=dfg_discovery.Variants.FREQUENCY)
    netM = generate_nx_graph_from_dfg(dfg2M)
    
    fP = dfg_functions.max_flow_graph(netP)
    fM = dfg_functions.max_flow_graph(netM)
    
    dfgP = dfg_discovery.apply(log_art, variant=dfg_discovery.Variants.FREQUENCY)
    dfgM = dfg_discovery.apply(logM_art, variant=dfg_discovery.Variants.FREQUENCY)
    
    start_activities = start_activities_filter.get_start_activities(log)
    start_activitiesM = start_activities_filter.get_start_activities(logM)
    end_activities = end_activities_filter.get_end_activities(log)
    end_activitiesM = end_activities_filter.get_end_activities(logM)
    
    loop_data_P = get_loop_data(dfgP,A,B)
    loop_data_M = get_loop_data(dfgM,A,B)
    
    feat_scores_togg = collections.defaultdict(lambda: 1, {})
    feat_scores = collections.defaultdict(lambda: 1, {})
    for x in dfg2.keys():
        feat_scores[x] = 1
        feat_scores_togg[x] = 1
    for y in dfg2M.keys():
        feat_scores[y] = 1
        feat_scores_togg[y] = 1
        
    logP_var = Counter([tuple([x['concept:name'] for x in t]) for t in log])
    logM_var = Counter([tuple([x['concept:name'] for x in t]) for t in logM])
    activitiesM = set(a for x in logM_var.keys() for a in x)
    
    start_acts_P = set([x[1] for x in dfgP if (x[0] == 'start')])-{'end'}
    end_acts_P = set([x[0] for x in dfgP if (x[1] == 'end')])-{'start'}
    
    size_par = len(log) / len(logM)
    
    parameters = {}
    activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters,pmutil.xes_constants.DEFAULT_NAME_KEY)
    
    dfg = [(k, v) for k, v in dfg_inst.apply(log, parameters=parameters).items() if v > 0]
    activities = get_activities_from_dfg(dfg)

    cut_function_util_dataP = dict()
    cut_function_util_dataM = dict()
    
    cut_function_util_dataP.update({'flow': fP})
    cut_function_util_dataM.update({'flow': fM})

    for cut_function_util_datas in [cut_function_util_dataP, cut_function_util_dataM]:
        cut_function_util_datas.update({'cost_Variant': cost_Variant})
        cut_function_util_datas.update({'sup': sup})
        cut_function_util_dataP.update({'feat_scores': feat_scores})
        cut_function_util_dataM.update({'feat_scores_togg': feat_scores_togg})
        

    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE or cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
        dic_indirect_follow_logP = get_indirect_follow_dic(log_art, activity_key, list(activities.keys()))
        dic_indirect_follow_logM = get_indirect_follow_dic(logM_art, activity_key, list(activitiesM))
        count_activitiesP = attributes_get.get_attribute_values(log_art, activity_key)
        count_activitiesM = attributes_get.get_attribute_values(logM_art, activity_key)
        calc_repetition_FactorP = repetition_Factor(log_art, activity_key)
        calc_repetition_FactorM = repetition_Factor(logM_art, activity_key)
        
        cut_function_util_dataP.update({'dic_indirect_follow_log': dic_indirect_follow_logP})
        cut_function_util_dataM.update({'dic_indirect_follow_log': dic_indirect_follow_logM})
        cut_function_util_dataP.update({'count_activities': count_activitiesP})
        cut_function_util_dataM.update({'count_activities': count_activitiesM})
        cut_function_util_dataP.update({'calc_repetition_Factor': calc_repetition_FactorP})
        cut_function_util_dataM.update({'calc_repetition_Factor': calc_repetition_FactorM})
    
    data_loop_tau_P, data_loop_tau_M = get_loop_tau_data(dfgP, dfgM, activities, {},log,logM, log_art, logM_art, activity_key, activities, activitiesM,netP,netM)
    
    if len(set(activitiesM).intersection(A))==0 or len(set(activitiesM).intersection(B))==0:
        ratio = 0
    
    if cut_Type == "seq":
        cost_seq_P = dfg_functions.cost_seq(netP, A, B, loop_data_P["start_B"], loop_data_P["end_A"],feat_scores, cut_function_util_dataP)
        cost_seq_M = dfg_functions.cost_seq(netM, A.intersection(activitiesM), B.intersection(activitiesM), loop_data_M["start_B"].intersection(activitiesM), loop_data_M["end_A"].intersection(activitiesM),feat_scores_togg, cut_function_util_dataM)
        
        return combine_score_values(cost_seq_P,cost_seq_M,cost_Variant,ratio,size_par)
    elif cut_Type == "exc":
        cost_exc_P = dfg_functions.cost_exc(netP, A, B, cut_function_util_dataP)
        cost_exc_M = dfg_functions.cost_exc(netM, A.intersection(activitiesM), B.intersection(activitiesM), cut_function_util_dataM)
        
        return combine_score_values(cost_exc_P,cost_exc_M,cost_Variant,ratio,size_par)
    elif cut_Type == "exc_tau":
        if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE or cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
            cost_exc_tau_P = dfg_functions.cost_exc_tau(netP,log,sup,cost_Variant)
            cost_exc_tau_M = dfg_functions.cost_exc_tau(netM,logM,sup,cost_Variant)
            return combine_score_values(cost_exc_tau_P,cost_exc_tau_M,cost_Variant,ratio,size_par)
    elif cut_Type == "par":
        cost_par_P = dfg_functions.cost_par(netP, A.intersection(activitiesM), B.intersection(activitiesM), cut_function_util_dataP)
        cost_par_M = dfg_functions.cost_par(netM, A.intersection(activitiesM), B.intersection(activitiesM), cut_function_util_dataM)
        return combine_score_values(cost_par_P,cost_par_M,cost_Variant,ratio,size_par)
    elif cut_Type == "loop":
        cost_loop_P = dfg_functions.cost_loop(netP, A, B, loop_data_P["start_A"], loop_data_P["end_A"], loop_data_P["input_B"], loop_data_P["output_B"], cut_function_util_dataP)
        cost_loop_M = dfg_functions.cost_loop(netM, A.intersection(activitiesM), B.intersection(activitiesM), loop_data_M["start_A"], loop_data_M["end_A"], loop_data_M["input_B"], loop_data_M["output_B"], cut_function_util_dataM)
        if cost_loop_P is not False:
            return combine_score_values(cost_loop_P,cost_loop_M,cost_Variant,ratio,size_par)
    elif cut_Type == "loop_tau":
        cost_loop_P, c_recP = dfg_functions.cost_loop_tau(start_acts_P,end_acts_P,log,sup,dfgP,start_activities,end_activities,cost_Variant,data_loop_tau_P)
        cost_loop_M, c_recM = dfg_functions.cost_loop_tau(start_acts_P.intersection(start_activitiesM.keys()),end_acts_P.intersection(end_activitiesM.keys()), logM, sup, dfgM, start_activitiesM,end_activitiesM,cost_Variant,data_loop_tau_M)
        if c_recP > 0:
            return combine_score_values(cost_loop_P,cost_loop_M,cost_Variant,ratio,size_par)

    return 0

# outside call function
def get_best_cut_with_cut_type(log, logM, cut_type = "", sup= None, ratio = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):
    
        log_art = artificial_start_end(log.__deepcopy__())
        logM_art = artificial_start_end(logM.__deepcopy__())
        parameters = {}
        
        start_activities = start_activities_filter.get_start_activities(log)
        start_activitiesM = start_activities_filter.get_start_activities(logM)
        end_activities = end_activities_filter.get_end_activities(log)
        end_activitiesM = end_activities_filter.get_end_activities(logM)
        
        dfg = [(k, v) for k, v in dfg_inst.apply(log, parameters=parameters).items() if v > 0]
        activities = get_activities_from_dfg(dfg)

        
        size_par=len(log)/len(logM)
        
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters, 
                                                  pmutil.xes_constants.DEFAULT_NAME_KEY)
        
        isbase, cut, sorted_cuts, detected_cut, new_log_P, new_log_M, _ = get_cuts(log,logM, log_art, logM_art,start_activities,end_activities,start_activitiesM,end_activitiesM,activities,activity_key,sup,ratio,size_par,cost_Variant,"None",parameters)
        
        for cur_cut in sorted_cuts[:10]:
            if cur_cut[1] == cut_type:
                return True, cur_cut
                
        return False, cut

# outside call function
def get_best_cut(log, logM, sup= None, ratio = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):
    
        log_art = artificial_start_end(log.__deepcopy__())
        logM_art = artificial_start_end(logM.__deepcopy__())
        parameters = {}
        
        start_activities = start_activities_filter.get_start_activities(log)
        start_activitiesM = start_activities_filter.get_start_activities(logM)
        end_activities = end_activities_filter.get_end_activities(log)
        end_activitiesM = end_activities_filter.get_end_activities(logM)
        
        dfg = [(k, v) for k, v in dfg_inst.apply(log, parameters=parameters).items() if v > 0]
        activities = get_activities_from_dfg(dfg)

        
        size_par=len(log)/len(logM)
        
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters, 
                                                  pmutil.xes_constants.DEFAULT_NAME_KEY)
        
        isbase, cut, sorted_cuts, detected_cut, new_log_P, new_log_M, _, _ = get_cuts(log,logM, log_art, logM_art,start_activities,end_activities,start_activitiesM,end_activitiesM,activities,activity_key,sup,ratio,size_par,cost_Variant,"None",parameters)
        return cut


def get_cuts(log, logM,log_art, logM_art, self_start_activities, self_end_activities, self_start_activitiesM, self_end_activitiesM, self_activities,activity_key, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, detected_cut = None, parameters=None, useGNN = False, self_loaded_gnn_models = None):
        logP_var = Counter([tuple([x['concept:name'] for x in t]) for t in log])
        logM_var = Counter([tuple([x['concept:name'] for x in t]) for t in logM])
        
        if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE and sup > 1:
            msg = "Error, unsupported sup value."
            logging.error(msg)
            raise Exception(msg)
        
        
        input_ratio = ratio
        isbase = False
        isRelationBase = False

        # check base cases:
        isbase, cut, detected_cut = dfg_functions.check_base_case(detected_cut, logP_var, logM_var, sup, ratio, size_par)
        
        netP = None
        netM = None
        new_log_P = None
        new_log_M = None
        
        start_time_measurement_partitioning = None
        possible_partitions = None
                
        if isbase == False:
            dfg2 = dfg_discovery.apply(log_art, variant=dfg_discovery.Variants.FREQUENCY)
            # dfg2 = dfg_functions.remove_infrequent_edges(dfg2,self_end_activities, 0.3)

            netP = generate_nx_graph_from_dfg(dfg2)
            if ('start', 'end') in dfg2:
                del dfg2[('start', 'end')]

            dfg2M = dfg_discovery.apply(logM_art, variant=dfg_discovery.Variants.FREQUENCY)

            netM = generate_nx_graph_from_dfg(dfg2M)
            if ('start', 'end') in dfg2M:
                del dfg2M[('start', 'end')]
            
            dfgP = dfg_discovery.apply(log_art, variant=dfg_discovery.Variants.FREQUENCY)
            # dfgP = dfg_functions.remove_infrequent_edges(dfgP,self_end_activities, 0.3)
            
            dfgM = dfg_discovery.apply(logM_art, variant=dfg_discovery.Variants.FREQUENCY)

            

            start_acts_P = set([x[1] for x in dfgP if (x[0] == 'start')])-{'end'}
            end_acts_P = set([x[0] for x in dfgP if (x[1] == 'end')])-{'start'}
            
            use_relation_base_case = False

            if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE and use_relation_base_case:
                isRelationBase, cut, new_log_P, new_log_M = dfg_functions.check_relation_base_case(netP, netM,log,logM, sup, ratio, size_par, dfgP, dfgM, activity_key, start_acts_P, end_acts_P, self_start_activities,self_end_activities)
                
                if isRelationBase == True:
                    isbase = True

            if isRelationBase == False:
                if parameters == {}:
                    feat_scores_togg = collections.defaultdict(lambda: 1, {})
                    feat_scores = collections.defaultdict(lambda: 1, {})
                    for x in dfg2.keys():
                        feat_scores[x] = 1
                        feat_scores_togg[x] = 1
                    for y in dfg2M.keys():
                        feat_scores[y] = 1
                        feat_scores_togg[y] = 1


                cut = []

                start_time_measurement_partitioning = time.time()
                if useGNN == True:
                    gnn_data_object_data = gnn_models_bi.generate_data_from_dfg(dfgP, dfgM, sup, ratio, size_par)
                
                # start_time_measurement_partitioning = time.time()


                if useGNN == True:
                    if ratio == 0 and False:
                        #  a check if needed if we only consider logP for gnn prediction
                        possible_partition_gnn = uni_get_partitions_from_gnn(root_path, gnn_path, log, logM, sup, ratio, size_par, 0.1)
                    else:
                        possible_partition_gnn = gnn_models_bi.get_partitions_from_gnn_models_and_data_object(self_loaded_gnn_models, gnn_data_object_data, 0.1)
                    if possible_partition_gnn == None:
                        possible_partitions = dfg_functions.find_possible_partitions(netP)
                    else:
                        possible_partitions = possible_partition_gnn
                else:
                    possible_partitions = dfg_functions.find_possible_partitions(netP)
                    
                
                if len(possible_partitions) >= 1000:
                    print("Warning: Length of possible partitions is " + str(len(possible_partitions)))
                
                # recalculate and assign since net pruning in find_possible_partitions change them
                start_acts_P = set([x[1] for x in dfgP if (x[0] == 'start')])-{'end'}
                end_acts_P = set([x[0] for x in dfgP if (x[1] == 'end')])-{'start'}

                def seperate_cuts(cuts):
                    res_partitions = []
                    for cut in cuts:
                        for cut_type in cut[2]:
                            res_partitions.append((cut[0],cut[1],{cut_type}))
                    return res_partitions
                
                possible_partitions = seperate_cuts(possible_partitions)
                    
                activitiesM = set(a for x in logM_var.keys() for a in x)


                #########################
                fP = dfg_functions.max_flow_graph(netP)
                fM = dfg_functions.max_flow_graph(netM)
                
                data_loop_tau_P, data_loop_tau_M = get_loop_tau_data(dfgP, dfgM, self_activities, {},log,logM, log_art, logM_art, activity_key, self_activities, activitiesM,netP,netM)
                
                if len(start_acts_P.intersection(end_acts_P)) == 0:
                    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE and use_relation_base_case == False:
                        cost_loop_P, c_recP = dfg_functions.cost_loop_tau(start_acts_P,end_acts_P,log,sup,dfgP,self_start_activities,self_end_activities,cost_Variant,data_loop_tau_P)
                        cost_loop_M, c_recM = dfg_functions.cost_loop_tau(start_acts_P.intersection(self_start_activitiesM.keys()),end_acts_P.intersection(self_end_activitiesM.keys()), logM, sup, dfgM, self_start_activitiesM,self_end_activitiesM,cost_Variant,data_loop_tau_M)
                        if c_recP > 0:
                            cut.append(((start_acts_P, end_acts_P),'loop_tau',cost_loop_P,cost_loop_M,combine_score_values(cost_loop_P,cost_loop_M,cost_Variant,ratio,size_par),1))
                    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE or cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
                        cost_loop_P, c_recP = dfg_functions.cost_loop_tau(start_acts_P,end_acts_P,log,sup,dfgP,self_start_activities,self_end_activities,cost_Variant,data_loop_tau_P)
                        cost_loop_M, c_recM = dfg_functions.cost_loop_tau(start_acts_P.intersection(self_start_activitiesM.keys()),end_acts_P.intersection(self_end_activitiesM.keys()), logM, sup, dfgM, self_start_activitiesM,self_end_activitiesM,cost_Variant,data_loop_tau_M)
                        if c_recP > 0:
                            cut.append(((start_acts_P, end_acts_P),'loop_tau',cost_loop_P,cost_loop_M,combine_score_values(cost_loop_P,cost_loop_M,cost_Variant,ratio,size_par),1))


                cut_function_util_dataP = dict()
                cut_function_util_dataM = dict()
                
                cut_function_util_dataP.update({'flow': fP})
                cut_function_util_dataM.update({'flow': fM})
                
                for cut_function_util_datas in [cut_function_util_dataP, cut_function_util_dataM]:
                    cut_function_util_datas.update({'cost_Variant': cost_Variant})
                    cut_function_util_datas.update({'sup': sup})
                    cut_function_util_datas.update({'feat_scores': feat_scores})
                    cut_function_util_datas.update({'feat_scores_togg': feat_scores_togg})
                    
                
                if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE or cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
                    dic_indirect_follow_logP = get_indirect_follow_dic(log_art, activity_key, list(self_activities.keys()))
                    dic_indirect_follow_logM = get_indirect_follow_dic(logM_art, activity_key, list(activitiesM))
                    count_activitiesP = attributes_get.get_attribute_values(log_art, activity_key)
                    count_activitiesM = attributes_get.get_attribute_values(logM_art, activity_key)
                    calc_repetition_FactorP = repetition_Factor(log_art, activity_key)
                    calc_repetition_FactorM = repetition_Factor(logM_art, activity_key)
                    
                    cut_function_util_dataP.update({'dic_indirect_follow_log': dic_indirect_follow_logP})
                    cut_function_util_dataM.update({'dic_indirect_follow_log': dic_indirect_follow_logM})
                    cut_function_util_dataP.update({'count_activities': count_activitiesP})
                    cut_function_util_dataM.update({'count_activities': count_activitiesM})
                    cut_function_util_dataP.update({'calc_repetition_Factor': calc_repetition_FactorP})
                    cut_function_util_dataM.update({'calc_repetition_Factor': calc_repetition_FactorM})
                    
                if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
                    bias_lP = get_bias_l(log, activity_key)
                    bias_lM = get_bias_l(logM, activity_key)
                    
                    cut_function_util_dataP.update({'bias_l': bias_lP})
                    cut_function_util_dataM.update({'bias_l': bias_lM})
                    
                  
                start_time_calculation = time.time()
                for pp in possible_partitions:
                    A = pp[0] - {'start', 'end'}
                    B = pp[1] - {'start', 'end'}
                    
                    loop_data_P = get_loop_data(dfgP,A,B)
                    loop_data_M = get_loop_data(dfgM,A,B)

                    type = pp[2]
                    if len(set(activitiesM).intersection(A))==0 or len(set(activitiesM).intersection(B))==0:
                        ratio = 0
                    else:
                        ratio = input_ratio
                        
                    #####################################################################
                    # seq check
                    if "seq" in type:
                        fit_seq = dfg_functions.fit_seq(logP_var, A, B)
                        if fit_seq > 0.0:
                            cost_seq_P = dfg_functions.cost_seq(netP, A, B, loop_data_P["start_B"], loop_data_P["end_A"],feat_scores, cut_function_util_dataP)
                            cost_seq_M = dfg_functions.cost_seq(netM, A.intersection(activitiesM), B.intersection(activitiesM), loop_data_M["start_B"].intersection(activitiesM), loop_data_M["end_A"].intersection(activitiesM),feat_scores_togg, cut_function_util_dataM)
                            
                            cut.append(((A, B), 'seq', cost_seq_P, cost_seq_M,combine_score_values(cost_seq_P,cost_seq_M,cost_Variant,ratio,size_par), fit_seq))
                    #####################################################################


                    #####################################################################
                    # xor check
                    if "exc" in type:
                        fit_exc = dfg_functions.fit_exc(logP_var, A, B)
                        if fit_exc > 0.0:
                            cost_exc_P = dfg_functions.cost_exc(netP, A, B, cut_function_util_dataP)
                            cost_exc_M = dfg_functions.cost_exc(netM, A.intersection(activitiesM), B.intersection(activitiesM), cut_function_util_dataM)
                            cut.append(((A, B), 'exc', cost_exc_P, cost_exc_M,combine_score_values(cost_exc_P,cost_exc_M,cost_Variant,ratio,size_par), fit_exc))
                    #####################################################################


                    #####################################################################
                    # xor-tau check
                    if dfg_functions.n_edges(netP,{'start'},{'end'})>0:
                        if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE and use_relation_base_case == False:
                            cost_exc_tau_P = dfg_functions.cost_exc_tau(netP,log,sup,cost_Variant)
                            cost_exc_tau_M = dfg_functions.cost_exc_tau(netM,logM,sup,cost_Variant)
                            # print(cost_exc_tau_P) 
                            cut.append(((A.union(B), set()), 'exc_tau',cost_exc_tau_P,cost_exc_tau_M,combine_score_values(cost_exc_tau_P,cost_exc_tau_M,cost_Variant,ratio,size_par),1))
                        if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE or cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
                        # if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE or cost_Variant == custom_enum.Cost_Variant.ACTIVITY_APROXIMATE_SCORE:
                            cost_exc_tau_P = dfg_functions.cost_exc_tau(netP,log,sup,cost_Variant)
                            cost_exc_tau_M = dfg_functions.cost_exc_tau(netM,logM,sup,cost_Variant)
                            # print(cost_exc_tau_P) 
                            cut.append(((A.union(B), set()), 'exc_tau',cost_exc_tau_P,cost_exc_tau_M,combine_score_values(cost_exc_tau_P,cost_exc_tau_M,cost_Variant,ratio,size_par),1))
                    #####################################################################


                    #####################################################################
                    # parallel check
                    if "par" in type:
                        cost_par_P = dfg_functions.cost_par(netP, A.intersection(activitiesM), B.intersection(activitiesM), cut_function_util_dataP)
                        cost_par_M = dfg_functions.cost_par(netM, A.intersection(activitiesM), B.intersection(activitiesM), cut_function_util_dataM)
                        cut.append(((A, B), 'par', cost_par_P, cost_par_M,combine_score_values(cost_par_P,cost_par_M,cost_Variant,ratio,size_par),1))
                    #####################################################################


                    #####################################################################
                    # loop check
                    if "loop" in type:
                        fit_loop = dfg_functions.fit_loop(logP_var, A, B, loop_data_P["end_A"], loop_data_P["start_A"])
                        if (fit_loop > 0.0):
                            cost_loop_P = dfg_functions.cost_loop(netP, A, B, loop_data_P["start_A"], loop_data_P["end_A"], loop_data_P["input_B"], loop_data_P["output_B"], cut_function_util_dataP)
                            cost_loop_M = dfg_functions.cost_loop(netM, A.intersection(activitiesM), B.intersection(activitiesM), loop_data_M["start_A"], loop_data_M["end_A"], loop_data_M["input_B"], loop_data_M["output_B"], cut_function_util_dataM)

                            if cost_loop_P is not False:
                                cut.append(((A, B), 'loop', cost_loop_P, cost_loop_M,combine_score_values(cost_loop_P,cost_loop_M,cost_Variant,ratio,size_par), fit_loop))
                    #####################################################################

        sorted_cuts = []
        if isbase == False and isRelationBase == False:
            if len(cut) != 0:
                if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
                    sorted_cuts = sorted(cut, key=lambda x: (x[4], x[2],['exc','exc_tau','seq','par','loop','loop_tau'].index(x[1]), -(len(x[0][0]) * len(x[0][1]) / (len(x[0][0]) + len(x[0][1])))))
                    cut = sorted_cuts[0]
                else:
                    sorted_cuts = list(filter(lambda x: x[2] > 0, cut))
                    if len(sorted_cuts) != 0:
                        sorted_cuts = sorted(sorted_cuts, key=lambda x: (x[4], x[2],['exc','exc_tau','seq','par','loop','loop_tau'].index(x[1]), -(len(x[0][0]) * len(x[0][1]) / (len(x[0][0]) + len(x[0][1])))))
                        sorted_cuts.reverse()
                        cut = sorted_cuts[0]
                    else:
                        cut = ('none', 'none', 'none','none','none', 'none')   
            else:
                cut = ('none', 'none', 'none','none','none', 'none')

        time_measurement_partitioning = None
        if start_time_measurement_partitioning != None:
            time_measurement_partitioning = time.time() - start_time_measurement_partitioning

        return isbase, cut, sorted_cuts, detected_cut, new_log_P, new_log_M, possible_partitions, time_measurement_partitioning

def get_gnn_cut_distance_from_best_cut(best_cut, cuts_gnn):
    cut_type = best_cut[1]
    best_gnn_cut_list = []
    for cut in cuts_gnn:
        if cut_type in cut[2]:
            best_gnn_cut_list.append(cut)
        
    if len(best_gnn_cut_list) == 0:
        return 100
    
    best_distance = len(best_cut[0][0]) + len(best_cut[0][1])
    for best_gnn_cut in best_gnn_cut_list:
        if len(best_cut[0][0]) + len(best_cut[0][1]) != len(best_gnn_cut[0]) + len(best_gnn_cut[1]):
            best_distance = 101
            break
    
        cur_distance = 0
        best_cut_parA, best_cut_parB = best_cut[0]
        best_gnn_cut_parA, best_gnn_cut_parB = best_gnn_cut[0], best_gnn_cut[1]
        
        for act in best_cut_parA:
            if act not in best_gnn_cut_parA:
                cur_distance += 1
        for act in best_cut_parB:
            if act not in best_gnn_cut_parB:
                cur_distance += 1
                
        if cur_distance < best_distance:
            best_distance = cur_distance
            
    return best_distance

import fpdf
from PIL import Image
def save_deviating_cuts(filename, log, logM, cut, cut_gnn, solution_distance, sup= None, ratio = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):
    pdf = fpdf.FPDF(format='letter') #pdf format
    pdf.add_page() #create new page
    pdf.set_font("Arial", size=8) # font and textsize

    pdf.cell(1000, 4, txt="cut | cut_gnn | solution_distance | sup | ratio | cost_Variant | dfgP | dfgM", ln=1, align="L")
    pdf.cell(1000, 4, txt="", ln=1, align="L")
    pdf.cell(1000, 4, txt="sup: " + str(sup), ln=1, align="L")
    pdf.cell(1000, 4, txt="ratio: " + str(ratio), ln=1, align="L")
    pdf.cell(1000, 4, txt="cost_Variant: " + str(cost_Variant), ln=1, align="L")
    pdf.cell(1000, 4, txt="", ln=1, align="L")
    pdf.cell(1000, 4, txt="cut: " + str(cut), ln=1, align="L")
    pdf.cell(1000, 4, txt="cut_gnn: " + str(cut_gnn), ln=1, align="L")
    pdf.cell(1000, 4, txt="solution_distance: " + str(solution_distance), ln=1, align="L")
    pdf.cell(1000, 4, txt="percentage: " + str(solution_distance/(len(cut[0][0]) + len(cut[0][1]))), ln=1, align="L")
    
    for log_name, cur_log in zip(["logP_img.png", "logM_img.png"],[log, logM]):
        save_dfg(cur_log, log_name)
        img = Image.open(log_name)
        
        width,height = img.size
        
        # print(width, height)
        pdf.image(log_name, w=min(70,width/8),h=min(70,height/8))
        img.close()
    pdf.output(filename + ".pdf")
    

    for log_name, cur_log in zip(["logP_img.png", "logM_img.png"],[log, logM]):
        if os.path.exists(log_name):
            # Delete the file
            os.remove(log_name)

class SubtreePlain(object):
    def __init__(self, logp,logm, dfg, master_dfg, initial_dfg, activities, counts, rec_depth, noise_threshold=0,
                 start_activities=None, end_activities=None, initial_start_activities=None,
                 initial_end_activities=None, parameters=None, real_init=True, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, use_gnn = False, time_comparison = False):

        if real_init:
            self.master_dfg = copy.copy(master_dfg)
            self.initial_dfg = copy.copy(initial_dfg)
            self.counts = counts
            self.rec_depth = rec_depth
            self.noise_threshold = noise_threshold
            self.start_activities = start_activities_filter.get_start_activities(logp)
            self.start_activitiesM = start_activities_filter.get_start_activities(logm)
            self.end_activities = end_activities_filter.get_end_activities(logp)
            self.end_activitiesM = end_activities_filter.get_end_activities(logm)
            self.initial_start_activities = initial_start_activities
            if self.initial_start_activities is None:
                self.initial_start_activities = infer_start_activities(master_dfg)
            self.initial_end_activities = initial_end_activities
            if self.initial_end_activities is None:
                self.initial_end_activities = infer_end_activities(master_dfg)

            self.log = logp
            self.log_art = artificial_start_end(copy.deepcopy(logp))
            self.logM = logm
            self.logM_art = artificial_start_end(copy.deepcopy(logm))
            self.inverted_dfg = None
            self.original_log = logp
            self.activities = None
            
            self.loaded_gnn_models = None
            if use_gnn == True or time_comparison == True:
                gnn_path = os.path.join("GNN_partitioning", "GNN_Model")
                root_path = os.getcwd().split("IMBI_Master")[0] + "IMBI_Master"
                self.loaded_gnn_models = gnn_models_bi.load_gnn_models(root_path, gnn_path)
            
            self.initialize_tree(dfg, logp, logm, initial_dfg, activities, parameters = parameters, sup = sup, ratio = ratio, size_par = size_par, cost_Variant = cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison)


    def initialize_tree(self, dfg, logp, logm, initial_dfg, activities, second_iteration = False, end_call = True,
                        parameters = None, sup = None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, use_gnn = False,time_comparison = False):

        if activities is None:
            self.activities = get_activities_from_dfg(dfg)
        else:
            self.activities = copy.copy(activities)
        self.detected_cut = None
        self.time_measurement_partitioning_reachability = None
        self.time_measurement_partitioning_gnn = None
        self.children = []
        self.log = logp
        self.log_art = artificial_start_end(logp.__deepcopy__())
        self.logM = logm
        self.logM_art = artificial_start_end(logm.__deepcopy__())
        self.original_log = logp
        self.parameters = parameters

        self.detect_cut(second_iteration=False, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant = cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison)
        

    def detect_cut(self,second_iteration=False, parameters=None, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, use_gnn = False, time_comparison = False):
        
        ratio = ratio
        sup = sup
        input_ratio = ratio
        
        if parameters is None:
            parameters = {}
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters,
                                                    pmutil.xes_constants.DEFAULT_NAME_KEY)
        
        
        isbase, cut, sorted_cuts, detected_cut, new_log_P, new_log_M, possible_partitions, time_measurement_partitioning = get_cuts(self.log,self.logM,self.log_art,self.logM_art,self.start_activities,self.end_activities,self.start_activitiesM,self.end_activitiesM,self.activities,activity_key,sup,ratio, size_par,cost_Variant,self.detected_cut,self.parameters, useGNN = use_gnn, self_loaded_gnn_models = self.loaded_gnn_models)
        
        if time_comparison == True:
            inverse_use_gnn = not use_gnn
            _, _, _, _, _, _, _, time_measurement_partitioning_inverse = get_cuts(self.log,self.logM,self.log_art,self.logM_art,self.start_activities,self.end_activities,self.start_activitiesM,self.end_activitiesM,self.activities,activity_key,sup,ratio, size_par,cost_Variant,self.detected_cut,self.parameters, useGNN = inverse_use_gnn, self_loaded_gnn_models = self.loaded_gnn_models)
            
            if use_gnn == True:
                self.time_measurement_partitioning_gnn = time_measurement_partitioning
                self.time_measurement_partitioning_reachability = time_measurement_partitioning_inverse
            else:
                self.time_measurement_partitioning_gnn = time_measurement_partitioning_inverse
                self.time_measurement_partitioning_reachability = time_measurement_partitioning
                

        self.detected_cut = detected_cut
        
        if False:
            show_deviation = False
            if show_deviation == False:
                if isbase:
                    if use_gnn == False:
                        _, cut_gnn, _, _, _, _, _ = get_cuts(self.log,self.logM,self.log_art,self.logM_art,self.start_activities,self.end_activities,self.start_activitiesM,self.end_activitiesM,self.activities,activity_key,sup,ratio, size_par,cost_Variant,self.detected_cut,self.parameters, useGNN = True)
                    else:
                        cut_gnn = cut
                        _, cut, _, _, _, _, possible_partitions = get_cuts(self.log,self.logM,self.log_art,self.logM_art,self.start_activities,self.end_activities,self.start_activitiesM,self.end_activitiesM,self.activities,activity_key,sup,ratio, size_par,cost_Variant,self.detected_cut,self.parameters,useGNN = False)
                    
                    solution_distance = get_gnn_cut_distance_from_best_cut(cut, possible_partitions)
                    print("solution_distance: " + str(solution_distance))
                    if cut_gnn[1] != cut[1]:
                        solution_distance = get_gnn_cut_distance_from_best_cut(cut, possible_partitions)
                        freeFileNumber = 1
                        filename = "Deviations"
                        while(os.path.isfile(filename + str(freeFileNumber) + ".pdf")):
                            freeFileNumber += 1
                        save_deviating_cuts(filename + str(freeFileNumber), self.log, self.logM, cut, cut_gnn, solution_distance, sup= sup, ratio = ratio, cost_Variant = cost_Variant)

            debugCutDetection = False
            if debugCutDetection:
                dfg_temp = dfg_discovery.apply(self.log_art, variant=dfg_discovery.Variants.FREQUENCY)
                start_act_cur_dfg = start_activities_get.get_start_activities(self.log_art, parameters=parameters)
                end_act_cur_dfg = end_activities_get.get_end_activities(self.log_art, parameters=parameters)
                
                dfgM_temp = dfg_discovery.apply(self.logM_art, variant=dfg_discovery.Variants.FREQUENCY)
                start_act_cur_dfgM = start_activities_get.get_start_activities(self.logM_art, parameters=parameters)
                end_act_cur_dfgM = end_activities_get.get_end_activities(self.logM_art, parameters=parameters)

                numberBestCutsSaved = 3
                currentIteration = 1
                
                folder_name = os.path.join(root_path,"imbi_cuts")
                folder_name = "imbi_cuts"
                # Check if the folder already exists
                if not os.path.exists(folder_name):
                    # Create the folder
                    os.makedirs(folder_name)
                    
                file_path = "imbi_cuts/depth_" + str(self.rec_depth) + "_It_" + str(currentIteration)
                while(os.path.isfile(file_path + ".png")):
                    currentIteration += 1
                    file_path = "imbi_cuts/depth_" + str(self.rec_depth) + "_It_" + str(currentIteration)
                
                save_vis_dfg(dfg_temp, start_act_cur_dfg, end_act_cur_dfg, file_path + ".png")
                save_vis_dfg(dfgM_temp, start_act_cur_dfgM, end_act_cur_dfgM, file_path + "_M.png")
                with open(file_path + ".txt", "w") as file:
                    if isbase:
                        cutList = [list(cut)]
                        for cuts in cutList:
                            outputString = ""
                            for string_cut in cuts[1:]:
                                outputString = outputString + " " + str(string_cut)
                            file.write("Basecut" + outputString + "\n")
                            if cuts[0] != "none":
                                for cut_activity_sets in cuts[0]:
                                    for cut_activity in cut_activity_sets:
                                        file.write(str(cut_activity) + " | ")
                                    file.write("\n")
                    else:
                        numberCuts = min(numberBestCutsSaved,len(sorted_cuts))
                        cutList = sorted_cuts[:numberCuts]
                        for cuts in cutList:
                            outputString = ""
                            for string_cut in cuts[1:]:
                                outputString = outputString + " " + str(string_cut)
                            file.write("cut" + outputString + "\n")
                            for cut_activity_sets in cuts[0]:
                                for cut_activity in cut_activity_sets:
                                    file.write(str(cut_activity) + " | ")
                                file.write("\n")
             
                
        if self.rec_depth >= 30:
            print("rec_depth: " + str(self.rec_depth))

        if cut[1] == 'par':
            self.detected_cut = 'parallel'
            LAP, LBP = split.split('par', [cut[0][0], cut[0][1]], self.log, activity_key)
            LAM, LBM = split.split('par', [cut[0][0], cut[0][1]], self.logM, activity_key)
            new_logs = [[LAP,LAM],[LBP,LBM]]
            for l in new_logs:
                new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(l[0], activity_key)
                start_activities = list(start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                end_activities = list(end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                self.children.append(SubtreePlain(l[0],l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold, start_activities=start_activities,
                                 end_activities=end_activities,
                                 initial_start_activities=self.initial_start_activities,
                                 initial_end_activities=self.initial_end_activities,
                                 parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))
        elif cut[1] == 'seq':
            self.detected_cut = 'sequential'
  
            LAP, LBP = split.split('seq', [cut[0][0], cut[0][1]], self.log, activity_key)
            LAM, LBM = split.split('seq', [cut[0][0], cut[0][1]], self.logM, activity_key)
            new_logs = [[LAP,LAM],[LBP,LBM]]
            for l in new_logs:
                new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(l[0], activity_key)
                start_activities = list(
                    start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                end_activities = list(
                    end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                self.children.append(
                    SubtreePlain(l[0],l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold, start_activities=start_activities,
                                 end_activities=end_activities,
                                 initial_start_activities=self.initial_start_activities,
                                 initial_end_activities=self.initial_end_activities,
                                 parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))
        elif (cut[1] == 'exc') or (cut[1] == 'exc_tau'):
            self.detected_cut = 'concurrent'
            LAP,LBP = split.split('exc', [cut[0][0], cut[0][1]], self.log, activity_key)
            LAM, LBM = split.split('exc', [cut[0][0], cut[0][1]], self.logM, activity_key)
            new_logs = [[LAP,LAM],[LBP,LBM]]
            for l in new_logs:
                new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(l[0], activity_key)
                start_activities = list(
                    start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                end_activities = list(
                    end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                
                # start_act_cur_dfg = start_activities_get.get_start_activities(l[0], parameters=parameters)
                # end_act_cur_dfg = end_activities_get.get_end_activities(l[0], parameters=parameters)
                # cur_dfg = dfg_inst.apply(l[0], parameters=parameters)
                # view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)
                
                
                self.children.append(
                    SubtreePlain(l[0],l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold, start_activities=start_activities,
                                 end_activities=end_activities,
                                 initial_start_activities=self.initial_start_activities,
                                 initial_end_activities=self.initial_end_activities,
                                 parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))

        elif cut[1] == 'loop':
            self.detected_cut = 'loopCut'
            LAP,LBP = split.split('loop', [cut[0][0], cut[0][1]], self.log, activity_key)
            LAM, LBM = split.split('loop', [cut[0][0], cut[0][1]], self.logM, activity_key)
            new_logs = [[LAP,LAM],[LBP,LBM]]
            for l in new_logs:
                new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(l[0], activity_key)
                start_activities = list(
                    start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                end_activities = list(
                    end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                self.children.append(
                    SubtreePlain(l[0],l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold, start_activities=start_activities,
                                 end_activities=end_activities,
                                 initial_start_activities=self.initial_start_activities,
                                 initial_end_activities=self.initial_end_activities,
                                 parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))

        elif cut[1] == 'loop1':
            self.detected_cut = 'loopCut'
            LAP,LBP = split.split('loop1', [cut[0][0], cut[0][1]], self.log, activity_key)
            LAM, LBM = split.split('loop1', [cut[0][0], cut[0][1]], self.logM, activity_key)
            new_logs = [[LAP,LAM],[LBP,LBM]]
            for l in new_logs:
                new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(l[0], activity_key)
                start_activities = list(
                    start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                end_activities = list(
                    end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                self.children.append(
                    SubtreePlain(l[0],l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold, start_activities=start_activities,
                                 end_activities=end_activities,
                                 initial_start_activities=self.initial_start_activities,
                                 initial_end_activities=self.initial_end_activities,
                                 parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))

        elif cut[1] == 'strict_loop_tau':
            if cost_Variant != custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
                self.detected_cut = 'flower'
            else:
                self.detected_cut = 'strict_loop_tau'
                # new_log_P, new_log_M 

                new_dfg = [(k, v) for k, v in dfg_inst.apply(new_log_P, parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(new_log_P, activity_key)
                start_activities = list(
                    start_activities_get.get_start_activities(new_log_P, parameters=parameters).keys())
                end_activities = list(
                    end_activities_get.get_end_activities(new_log_P, parameters=parameters).keys())
                self.children.append(
                    SubtreePlain(new_log_P, new_log_M, new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                self.rec_depth + 1,
                                noise_threshold=self.noise_threshold, start_activities=start_activities,
                                end_activities=end_activities,
                                initial_start_activities=self.initial_start_activities,
                                initial_end_activities=self.initial_end_activities,
                                parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))
        elif cut[1] == 'loop_tau':
            self.detected_cut = 'loopCut'
            if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
                LAP, LBP = split.split('loop_tau', [cut[0][0], cut[0][1]], self.log, activity_key)
                LAM, LBM = split.split('loop_tau', [cut[0][0], cut[0][1]], self.logM, activity_key)
                new_logs = [[LAP,LAM],[LBP,LBM]]
                
                for l in new_logs:
                    new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                    activities = attributes_get.get_attribute_values(l[0], activity_key)
                    start_activities = list(
                        start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                    end_activities = list(
                        end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                    self.children.append(
                        SubtreePlain(l[0],l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                    self.rec_depth + 1,
                                    noise_threshold=self.noise_threshold, start_activities=start_activities,
                                    end_activities=end_activities,
                                    initial_start_activities=self.initial_start_activities,
                                    initial_end_activities=self.initial_end_activities,
                                    parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))

            elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:

                new_dfg = [(k, v) for k, v in dfg_inst.apply(new_log_P, parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(new_log_P, activity_key)
                start_activities = list(
                    start_activities_get.get_start_activities(new_log_P, parameters=parameters).keys())
                end_activities = list(
                    end_activities_get.get_end_activities(new_log_P, parameters=parameters).keys())
                self.children.append(
                    SubtreePlain(new_log_P,new_log_M, new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                self.rec_depth + 1,
                                noise_threshold=self.noise_threshold, start_activities=start_activities,
                                end_activities=end_activities,
                                initial_start_activities=self.initial_start_activities,
                                initial_end_activities=self.initial_end_activities,
                                parameters=parameters, sup= sup, ratio = input_ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison))

        elif cut[1] == 'none':
            self.detected_cut = 'flower'



def make_tree(logp, logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold, start_activities,
              end_activities, initial_start_activities, initial_end_activities, parameters=None, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE, use_gnn = False, time_comparison = False):

    tree = SubtreePlain(logp,logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold,
                        start_activities,
                        end_activities, initial_start_activities, initial_end_activities, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant, use_gnn = use_gnn, time_comparison = time_comparison)

    return tree
