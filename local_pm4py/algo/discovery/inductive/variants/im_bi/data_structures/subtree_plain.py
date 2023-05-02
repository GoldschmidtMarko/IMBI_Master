import collections
from copy import copy
import time
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
import logging

from local_pm4py.algo.analysis import dfg_functions
from local_pm4py.algo.analysis import custom_enum
import copy
from collections import Counter

# TODO delete debuging code
from pm4py import save_vis_dfg
from pm4py import view_dfg
debugCutDetection = True
import os


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
        G.add_edge(edge[0], edge[1], weight=dfg[edge])
    return G


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
   

def repetition_Factor(log, activity_key) -> float:
    dic_activities = get_dic_activities(log, activity_key)
    
    numberTraces = len(log)
    totalNumberEvents = sum(dic_activities.values())
    numberActivities = len(dic_activities.keys())
    if numberActivities == 0 or totalNumberEvents == 0:
        return 1
    
    return (numberTraces / (totalNumberEvents / numberActivities))

class SubtreePlain(object):
    def __init__(self, logp,logm, dfg, master_dfg, initial_dfg, activities, counts, rec_depth, noise_threshold=0,
                 start_activities=None, end_activities=None, initial_start_activities=None,
                 initial_end_activities=None, parameters=None, real_init=True, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):

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

            self.initialize_tree(dfg, logp, logm, initial_dfg, activities, parameters = parameters, sup = sup, ratio = ratio, size_par = size_par, cost_Variant = cost_Variant)


    def initialize_tree(self, dfg, logp, logm, initial_dfg, activities, second_iteration = False, end_call = True,
                        parameters = None, sup = None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):

        if activities is None:
            self.activities = get_activities_from_dfg(dfg)
        else:
            self.activities = copy.copy(activities)
        self.detected_cut = None
        self.children = []
        self.log = logp
        self.log_art = artificial_start_end(logp.__deepcopy__())
        self.logM = logm
        self.logM_art = artificial_start_end(logm.__deepcopy__())
        self.original_log = logp
        self.parameters = parameters

        self.detect_cut(second_iteration=False, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant = cost_Variant)


    def detect_cut(self,second_iteration=False, parameters=None, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):
        ratio = ratio
        sup = sup

        logP_var = Counter([tuple([x['concept:name'] for x in t]) for t in self.log])
        logM_var = Counter([tuple([x['concept:name'] for x in t]) for t in self.logM])


        if parameters is None:
            parameters = {}
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters,
                                                  pmutil.xes_constants.DEFAULT_NAME_KEY)
        
        
        if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE and sup > 1:
            msg = "Error, unsupported sup value."
            logging.error(msg)
            raise Exception(msg)

        isbase = False
        isRelationBase = False

        # check base cases:
        isbase, cut = dfg_functions.check_base_case(self, logP_var,logM_var, sup, ratio, size_par)
        
        if isbase == False:
            dfg2 = dfg_discovery.apply(self.log_art, variant=dfg_discovery.Variants.FREQUENCY)
            netP = generate_nx_graph_from_dfg(dfg2)
            del dfg2[('start', 'end')]

            dfg2M = dfg_discovery.apply(self.logM_art, variant=dfg_discovery.Variants.FREQUENCY)
            netM = generate_nx_graph_from_dfg(dfg2M)
            del dfg2M[('start', 'end')]
            
            dfgP = dfg_discovery.apply(self.log_art, variant=dfg_discovery.Variants.FREQUENCY)
            dfgM = dfg_discovery.apply(self.logM_art, variant=dfg_discovery.Variants.FREQUENCY)
            
            # start_act_cur_dfg = start_activities_get.get_start_activities(self.log, parameters=parameters)
            # end_act_cur_dfg = end_activities_get.get_end_activities(self.log, parameters=parameters)
            # cur_dfg = dfg_inst.apply(self.log, parameters=parameters)
            # view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)
            
            if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
                isRelationBase, cut, new_log_P, new_log_M = dfg_functions.check_relation_base_case(self, netP, netM,self.log,self.logM, sup, ratio, size_par, dfgP, dfgM, activity_key)
                
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


                possible_partitions = dfg_functions.find_possible_partitions(netP)

                cut = []
                activitiesM = set(a for x in logM_var.keys() for a in x)


                start_acts_P = set([x[1] for x in dfgP if (x[0] == 'start')])-{'end'}
                end_acts_P = set([x[0] for x in dfgP if (x[1] == 'end')])-{'start'}

                #########################
                fP = dfg_functions.max_flow_graph(netP)
                fM = dfg_functions.max_flow_graph(netM)

                if True and cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
                    missing_loopP = 0
                    missing_loopM = 0
                    rej_tau_loop = False
                    c_rec = 0

                    if len(start_acts_P.intersection(end_acts_P)) !=0:
                        rej_tau_loop = True
                    for x in start_acts_P:
                        for y in end_acts_P:
                            L1P = max(0, len(self.log) * sup * (self.start_activities[x] / (sum(self.start_activities.values()))) * (self.end_activities[y] / (sum(self.end_activities.values()))) - dfgP[(y, x)])
                            missing_loopP += L1P
                            c_rec += dfgP[(y, x)]

                    for x in start_acts_P.intersection(self.start_activitiesM.keys()):
                        for y in end_acts_P.intersection(self.end_activitiesM.keys()):
                            L1M = max(0, len(self.logM) * sup * (self.start_activitiesM[x] / (sum(self.start_activitiesM.values()))) * (self.end_activitiesM[y] / (sum(self.end_activitiesM.values()))) - dfgM[(y, x)])
                            missing_loopM += L1M

                    cost_loop_P = missing_loopP
                    cost_loop_M = missing_loopM

                    if rej_tau_loop == False and c_rec >0:
                        cut.append(((start_acts_P, end_acts_P), 'loop_tau', cost_loop_P, cost_loop_M,  cost_loop_P - ratio * size_par * cost_loop_M,1))
                ratio_backup = ratio
                
                dic_indirect_follow_logP = {}
                dic_indirect_follow_logM = {}
                count_activitiesP = {}
                count_activitiesM = {}
                calc_repetition_FactorP = 0
                calc_repetition_FactorM = 0
                
                if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
                    dic_indirect_follow_logP = get_indirect_follow_dic(self.log_art, activity_key, list(self.activities.keys()))
                    dic_indirect_follow_logM = get_indirect_follow_dic(self.logM_art, activity_key, list(activitiesM))
                    count_activitiesP = attributes_get.get_attribute_values(self.log_art, activity_key)
                    count_activitiesM = attributes_get.get_attribute_values(self.logM_art, activity_key)
                    calc_repetition_FactorP = repetition_Factor(self.log_art, activity_key)
                    calc_repetition_FactorM = repetition_Factor(self.logM_art, activity_key)
                    
                for pp in possible_partitions:
                    A = pp[0] - {'start', 'end'}
                    B = pp[1] - {'start', 'end'}

                    start_A_P = set([x[1] for x in dfgP if ((x[0] == 'start') and (x[1] in A))])
                    end_A_P = set([x[0] for x in dfgP if (x[0] in A and (x[1] == 'end'))])
                    start_B_P = set([x[1] for x in dfgP if (x[1] in B and (x[0] == 'start'))])
                    input_B_P = set([x[1] for x in dfgP if ((x[0] not in B) and (x[1] in B))])
                    output_B_P = set([x[0] for x in dfgP if ((x[0] in B) and (x[1] not in B))])

                    start_A_M = set([x[1] for x in dfgM if ((x[0] == 'start') and (x[1] in A))])
                    end_A_M = set([x[0] for x in dfgM if (x[0] in A and (x[1] == 'end'))])
                    start_B_M = set([x[1] for x in dfgM if (x[1] in B and (x[0] == 'start'))])
                    input_B_M = set([x[1] for x in dfgM if ((x[0] not in B) and (x[1] in B))])
                    output_B_M = set([x[0] for x in dfgM if ((x[0] in B) and (x[1] not in B))])

                    type = pp[2]
                    if len(set(activitiesM).intersection(A))==0 or len(set(activitiesM).intersection(B))==0:
                        ratio = 0
                    else:
                        ratio = ratio_backup

                    #####################################################################
                    # seq check
                    fit_seq = dfg_functions.fit_seq(logP_var, A, B)
                    if fit_seq > 0.0:
                        cost_seq_P = dfg_functions.cost_seq(netP, A, B, start_B_P, end_A_P, sup, fP, feat_scores, dic_indirect_follow_logP, self.log,  cost_Variant)
                        cost_seq_M = dfg_functions.cost_seq(netM, A.intersection(activitiesM), B.intersection(activitiesM), start_B_M.intersection(activitiesM), end_A_M.intersection(activitiesM), sup, fM, feat_scores_togg, dic_indirect_follow_logM, self.logM,  cost_Variant)
                        cut.append(((A, B), 'seq', cost_seq_P, cost_seq_M, cost_seq_P - ratio* size_par * cost_seq_M, fit_seq))
                    #####################################################################


                    #####################################################################
                    # xor check
                    if "exc" in type:
                        fit_exc = dfg_functions.fit_exc(logP_var, A, B)
                        if fit_exc > 0.0:
                            cost_exc_P = dfg_functions.cost_exc(netP, A, B, feat_scores, fP, dic_indirect_follow_logP, count_activitiesP, cost_Variant)
                            cost_exc_M = dfg_functions.cost_exc(netM, A.intersection(activitiesM), B.intersection(activitiesM), feat_scores, fM, dic_indirect_follow_logM, count_activitiesM, cost_Variant)
                            cut.append(((A, B), 'exc', cost_exc_P, cost_exc_M, cost_exc_P - ratio* size_par * cost_exc_M, fit_exc))
                    #####################################################################


                    #####################################################################
                    # xor-tau check
                    if dfg_functions.n_edges(netP,{'start'},{'end'})>0 and cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
                        # debugging code
                        # print(dfg_functions.n_edges(netP,{'start'},{'end'}))
                        # print(netP.out_degree('start', weight='weight'))
                        # seqScore = dfg_functions.cost_seq(netP, A, B, start_B_P, end_A_P, sup, fP, feat_scores, dic_indirect_follow_logP, self.log,  cost_Variant)
                        # print("Sequence score: " + str(seqScore))
                        # start_act_cur_dfg = start_activities_get.get_start_activities(self.log, parameters=parameters)
                        # end_act_cur_dfg = end_activities_get.get_end_activities(self.log, parameters=parameters)
                        # cur_dfg = dfg_inst.apply(self.log, parameters=parameters)
                        # view_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg)
                        # debugging code
            
                        cost_exc_tau_P = dfg_functions.cost_exc_tau(netP,self.log,sup,cost_Variant)
                        cost_exc_tau_M = dfg_functions.cost_exc_tau(netM,self.logM,sup,cost_Variant)
                        # print(cost_exc_tau_P)
                        cut.append(((A.union(B), set()), 'exc2', cost_exc_tau_P, cost_exc_tau_M,cost_exc_tau_P - ratio * size_par * cost_exc_tau_M,1))
                    #####################################################################


                    #####################################################################
                    # parallel check
                    if "par" in type:
                        cost_par_P = dfg_functions.cost_par(netP, A.intersection(activitiesM), B.intersection(activitiesM), sup, feat_scores, fP, dic_indirect_follow_logP, calc_repetition_FactorP, cost_Variant)
                        cost_par_M = dfg_functions.cost_par(netM, A.intersection(activitiesM), B.intersection(activitiesM), sup, feat_scores, fM,dic_indirect_follow_logM, calc_repetition_FactorM, cost_Variant)
                        cut.append(((A, B), 'par', cost_par_P, cost_par_M, cost_par_P - ratio * size_par * cost_par_M,1))
                    #####################################################################


                    #####################################################################
                    # loop check
                    if "loop" in type:
                        fit_loop = dfg_functions.fit_loop(logP_var, A, B, end_A_P, start_A_P)
                        if (fit_loop > 0.0):
                            cost_loop_P = dfg_functions.cost_loop(netP, A, B, sup, start_A_P, end_A_P, input_B_P, output_B_P, feat_scores, fP, dic_indirect_follow_logP, calc_repetition_FactorP, cost_Variant)
                            cost_loop_M = dfg_functions.cost_loop(netM, A.intersection(activitiesM), B.intersection(activitiesM), sup, start_A_M, end_A_M, input_B_M, output_B_M, feat_scores, fM, dic_indirect_follow_logM, calc_repetition_FactorM, cost_Variant)

                            if cost_loop_P is not False:
                                cut.append(((A, B), 'loop', cost_loop_P, cost_loop_M, cost_loop_P - ratio * size_par * cost_loop_M, fit_loop))
                    #####################################################################

        if isbase == False and isRelationBase == False:
            sorted_cuts = sorted(cut, key=lambda x: (x[4], x[2],['exc','exc2','seq','par','loop','loop_tau'].index(x[1]), -(len(x[0][0]) * len(x[0][1]) / (len(x[0][0]) + len(x[0][1])))))
            if len(sorted_cuts) != 0:
                if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
                    cut = sorted_cuts[0]
                else:
                    cut = sorted_cuts[-1]
            else:
                cut = ('none', 'none', 'none','none','none', 'none')

        if debugCutDetection:
            start_act_cur_dfg = start_activities_get.get_start_activities(self.log, parameters=parameters)
            end_act_cur_dfg = end_activities_get.get_end_activities(self.log, parameters=parameters)
            cur_dfg = dfg_inst.apply(self.log, parameters=parameters)
            
            # try:
            #     os.remove("imbi_cuts/cut" + str(self.rec_depth) + ".png")
            #     os.remove("imbi_cuts/cut" + str(self.rec_depth) + ".txt")
            # except OSError:
            #     pass
            
            numberBestCutsSaved = 3
            save_vis_dfg(cur_dfg, start_act_cur_dfg, end_act_cur_dfg, "imbi_cuts/cut" + str(self.rec_depth) + ".png")
            with open("imbi_cuts/cut" + str(self.rec_depth) + ".txt", "w") as file:
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
                    cutList = sorted_cuts[-numberCuts:]
                    cutList.reverse()
                    for cuts in cutList:
                        outputString = ""
                        for string_cut in cuts[1:]:
                            outputString = outputString + " " + str(string_cut)
                        file.write("cut" + outputString + "\n")
                        for cut_activity_sets in cuts[0]:
                            for cut_activity in cut_activity_sets:
                                file.write(str(cut_activity) + " | ")
                            file.write("\n")
                    
            
                
        # print(cut)

        if cut[1] == 'par':
            self.detected_cut = 'parallel'
            LAP,LBP = split.split('par', [cut[0][0], cut[0][1]], self.log, activity_key)
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
                                 parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))
        elif cut[1] == 'seq':
            self.detected_cut = 'sequential'
            LAP,LBP = split.split('seq', [cut[0][0], cut[0][1]], self.log, activity_key)
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
                                 parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))
        elif (cut[1] == 'exc') or (cut[1] == 'exc2'):
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
                                 parameters=parameters, sup= sup, ratio = ratio, size_par = size_par,
                                 cost_Variant=cost_Variant))

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
                                 parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))

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
                                 parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))

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
                                parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))
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
                                    parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))

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
                                parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant))

        elif cut[1] == 'none':
            self.detected_cut = 'flower'


def make_tree(logp, logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold, start_activities,
              end_activities, initial_start_activities, initial_end_activities, parameters=None, sup= None, ratio = None, size_par = None, cost_Variant = custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE):

    tree = SubtreePlain(logp,logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold,
                        start_activities,
                        end_activities, initial_start_activities, initial_end_activities, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, cost_Variant=cost_Variant)

    return tree
