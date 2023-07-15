import networkx as nx
from collections import Counter
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import logging
import math
from local_pm4py.algo.discovery.inductive.variants.im_bi.util import fall_through
from local_pm4py.algo.analysis import custom_enum

from pm4py.util import exec_utils

def get_direct_edge_count(net, from_node, to_node):
    if net.get_edge_data(from_node,to_node) == None:
        return 0
    else:
        return net.get_edge_data(from_node,to_node)['weight']
    
def get_frequency(dfg, edge):
    if edge not in dfg:
        return 0
    else:
        return dfg[edge]

def n_edges(net, S, T, scaling = None):
    net_c = copy.deepcopy(net)
    if scaling == None:
        edges_reweight = list(nx.edge_boundary(net_c, S, T, data='weight', default=1))
    else:
        edges = list(nx.edge_boundary(net_c, S, T, data='weight', default=1))
        edges_reweight = []
        for ed in edges:
            edges_reweight.append((ed[0],ed[1], ed[2]*scaling[(ed[0], ed[1])]))
            # net_c[ed[0]][ed[1]]['weight'] = net_c[sc[0]][sc[1]]['weight'] * scaling[sc]
        # edges = edges_reweight
    return sum(weight for u, v, weight in edges_reweight if (u in S and v in T))

def drop_SE(s):
    return s-{'start','end'}

def add_SE(net, s):
    if s & set(net.successors('start')):
        s.add('start')
    if s & set(net.predecessors('end')):
        s.add('end')
    return s

def add_SS(s):
    s.add('start')
    return s

def add_EE(s):
    s.add('end')
    return s

def r_to_s(net):
    return (set(nx.descendants(net, 'start')) == (set(net.nodes) - {'start'}))

def r_from_e(net):
    return (set(nx.ancestors(net, 'end')) == (set(net.nodes) - {'end'}))

def dfg_extract(log):
    dfgs = map((lambda t: [(t[i - 1], t[i]) for i in range(1, len(t))]), log)
    return Counter([dfg for lista in dfgs for dfg in lista])

def lal(net,a):
    return net.out_degree(weight='weight')[a]

def lAl(net,A):
    return sum([net.out_degree(weight='weight')[a] for a in A])

def toggle(dic):
    dic_new = defaultdict(lambda: 1, {})
    for x in dic:
        # dic_new[x] = (1-dic[x])+1
        dic_new[x] = 1/dic[x]
    return dic_new

def average(lst : "list[float]") -> float:
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)
    
def standard_deviation(lst : "list[float]") -> float:
    if len(lst) == 0:
        return 0
    
    mean = average(lst)
    sum = 0
    for i in lst:
        sum += math.pow( (i - mean) , 2)
        
    # TODO consider dividing by len(lst) - 1
    return math.sqrt( (sum / ( len(lst) ) ) )
            
def cost_seq(net, A, B, start_set, end_set, sup, flow, scores, dic_indirect_follow_log, cost_Variant):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return cost_seq_frequency(net, A, B, start_set, end_set, sup, flow, scores)
    elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
        return cost_seq_relation(net, A, B, dic_indirect_follow_log)
    else:
        msg = "Error, could not call a valid cost function for cost_seq."
        logging.error(msg)
        raise Exception(msg)
    
def cost_loop_tau(start_acts, end_acts, log, sup, dfg, start_activities_o, end_activities_o,  cost_Variant):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return cost_loop_tau_frequency(start_acts, end_acts, log, sup, dfg, start_activities_o, end_activities_o)
    elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
        msg = "Error, cost_loop_tau on cost variant ACTIVITY_RELATION_SCORE."
        raise Exception(msg)
    else:
        msg = "Error, could not call a valid cost function for cost_loop_tau."
        logging.error(msg)
        raise Exception(msg)
    
def cost_loop_tau_frequency(start_acts, end_acts, log, sup, dfg, start_activities_o, end_activities_o):
    missing_loop = 0
    c_rec = 0
    for x in start_acts:
        for y in end_acts:
            L1P = max(0, len(log) * sup * (start_activities_o[x] / (sum(start_activities_o.values()))) * (end_activities_o[y] / (sum(end_activities_o.values()))) - get_frequency(dfg,(y,x)))
            missing_loop += L1P
            c_rec += get_frequency(dfg,(y,x))
    return missing_loop, c_rec

def cost_seq_frequency(net, A, B, start_set, end_set, sup, flow, scores):
    scores_toggle = toggle(scores)
    c1 = n_edges(net, B, A, scaling=scores_toggle)

    c2 = 0
    
    for x in A:
        for y in B:
            c2 += max(0, scores[(x, y)] * net.out_degree(x, weight='weight') * sup * (net.out_degree(y, weight='weight') / (
                        sum([net.out_degree(p, weight='weight') for p in B]) + sum([net.out_degree(p, weight='weight') for p in A]))) - flow[(x, y)])

    c3 = 0
    for x in end_set:
        for y in start_set:
            c3 += max(0, scores[(x, y)] * n_edges(net, {x}, B.union({'end'}), scaling=scores) * sup * (n_edges(net, A.union({'start'}), {y}, scaling=scores) /
                                                                                                       (n_edges(net, A.union({'start'}), B.union({'end'}), scaling=scores))) - n_edges(net, {x}, {y}, scaling=scores))

    return c1 + c2 + c3

def cost_seq_relation(net, A, B, dic_indirect_follow_log):
    # TODO SUP
    scores = []
    for x in A:
        for y in B:
            dividend = get_direct_edge_count(net,x,y) + dic_indirect_follow_log[x][y] - (get_direct_edge_count(net,y,x) + dic_indirect_follow_log[y][x])
            
            dividend += get_direct_edge_count(net,"start",y) + get_direct_edge_count(net,x,"end")
            
            divisor = get_direct_edge_count(net,x,y) + dic_indirect_follow_log[x][y] + get_direct_edge_count(net,y,x) + dic_indirect_follow_log[y][x] + 1
            
            divisor += get_direct_edge_count(net,"start",y) + get_direct_edge_count(net,x,"end")
            
            res = dividend/divisor
            # res = res * (n_edges(net,{'start'},{'end'}) / net.out_degree('start', weight='weight'))
            scores.append(res)
    return average(scores) - standard_deviation(scores)

def fit_seq(log_var,A,B):
    count = 0
    for tr in log_var:
        for i in range(0,len(tr)-1):
            if (tr[i] in B) and (tr[i+1] in A):
                count += log_var[tr]
                break
    fit = 1-(count/sum(log_var.values()))
    return fit

def fit_exc(log_var,A,B):
    count = 0
    for tr in log_var:
        if set(tr).issubset(A) or set(tr).issubset(B):
            count += log_var[tr]
    fit = (count/sum(log_var.values()))
    return fit

def fit_loop(log_var,A,B,A_end,A_start):
    count = 0
    for tr in log_var:
        if len(tr)==0:
            continue
        if (tr[0] in B) or (tr[-1] in B):
            count += log_var[tr]
            continue
        for i in range(0,len(tr)-1):
            if (tr[i+1] in B) and (tr[i] in A):
                if (tr[i] not in A_end):
                    count += log_var[tr]
                break
            if (tr[i] in B) and (tr[i+1] in A):
                if (tr[i+1] not in A_start):
                    count += log_var[tr]
                break
    fit = 1 - (count / sum(log_var.values()))
    return fit


def cost_exc(net, A, B, scores, flow, dic_indirect_follow_log, count_activities, cost_Variant):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return cost_exc_frequency(net, A, B, scores)
    elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
        return cost_exc_relation(net, A, B, flow, dic_indirect_follow_log, count_activities)
    else:
        msg = "Error, could not call a valid cost function for cost_exc."
        logging.error(msg)
        raise Exception(msg)

def cost_exc_frequency(net, A, B, scores):
    scores_toggle = toggle(scores)
    c1 = n_edges(net, A, B, scaling =scores_toggle)
    c1 += n_edges(net,B ,A, scaling =scores_toggle)
    return c1

def cost_exc_relation(net, A, B, flow, dic_indirect_follow_log, count_activities):
    # TODO SUP
    scores = []
    for x in A:
        for y in B:
            dividend1 = count_activities[x] - (get_direct_edge_count(net,x,y) + get_direct_edge_count(net,y,x) + dic_indirect_follow_log[x][y] + dic_indirect_follow_log[y][x])
            divisor1 = count_activities[x]
            dividend2 = count_activities[y] - (get_direct_edge_count(net,x,y) + get_direct_edge_count(net,y,x) + dic_indirect_follow_log[x][y] + dic_indirect_follow_log[y][x])
            divisor2 = count_activities[y]
            scores.append( ((dividend1/divisor1)*0.5 + (dividend2/divisor2)*0.5) )
    return average(scores) - standard_deviation(scores)

def cost_exc_tau(net, log, sup_thr, cost_Variant):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return cost_exc_tau_frequency(net, log, sup_thr)
    elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
        msg = "Error, could not call cost_exc_tau on cost variant ACTIVITY_RELATION_SCORE."
        raise Exception(msg)
    else:
        msg = "Error, could not call a valid cost function for cost_exc_tau."
        logging.error(msg)
        raise Exception(msg)

def cost_exc_tau_frequency(net, log, sup_thr):
    cost = max(0, sup_thr * len(log) - n_edges(net,{'start'},{'end'}))
    return cost

def cost_exc_tau_relation(net, log):
    return n_edges(net,{'start'},{'end'}) / net.out_degree('start', weight='weight')


def cost_par(net, A, B, sup, scores, flow, dic_indirect_follow_log, calc_repetition_Factor, cost_Variant):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return cost_par_frequency(net, A, B, sup, scores)
    elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
        return cost_par_relation(net, A, B, sup, flow, dic_indirect_follow_log, calc_repetition_Factor)
    else:
        msg = "Error, could not call a valid cost function for cost_par."
        logging.error(msg)
        raise Exception(msg)

def cost_par_frequency(net, A, B, sup, scores):
    c1 = 0
    c2 = 0
    for a in A:
        for b in B:
            c1 += max(0, scores[(a, b)] * (net.out_degree(a, weight='weight') * sup * net.out_degree(b, weight='weight')) / (
                        (sum([net.out_degree(p, weight='weight') for p in B])) + (sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {a}, {b}, scaling=scores))
            c2 += max(0, scores[(b, a)] * (net.out_degree(b, weight='weight') * sup * net.out_degree(a, weight='weight')) / (
                        (sum([net.out_degree(p, weight='weight') for p in B])) + (sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {b}, {a}, scaling=scores))

    return c1+c2

def cost_par_relation(net, A, B, sup, flow, dic_indirect_follow_log, calc_repetition_Factor):
    # TODO SUP
    scores = []
    for x in A:
        for y in B:
            dividend1 = get_direct_edge_count(net,x,y)
            divisor1 = get_direct_edge_count(net,y,x) + 1
            dividend2 = get_direct_edge_count(net,y,x)
            divisor2 = get_direct_edge_count(net,x,y) + 1
            scores.append( min((dividend1/divisor1), (dividend2/divisor2)) )
    return average(scores) * min(calc_repetition_Factor, 1)


def cost_loop(net, A, B, sup, start_A, end_A, input_B, output_B, scores, flow, dic_indirect_follow_log, calc_repetition_Factor, cost_Variant):
    if cost_Variant == custom_enum.Cost_Variant.ACTIVITY_FREQUENCY_SCORE:
        return cost_loop_frequency(net, A, B, sup, start_A, end_A, input_B, output_B, scores)
    elif cost_Variant == custom_enum.Cost_Variant.ACTIVITY_RELATION_SCORE:
        return cost_loop_relation(net, A, B, sup, flow, start_A, end_A, input_B, output_B, dic_indirect_follow_log, calc_repetition_Factor)
    else:
        msg = "Error, could not call a valid cost function for cost_loop."
        logging.error(msg)
        raise Exception(msg)

def cost_loop_frequency(net, A, B, sup, start_A, end_A, input_B, output_B, scores):
    scores_toggle = toggle(scores)

    flag_loop_valid = False

    if n_edges(net, B, start_A) != 0:
        if n_edges(net, end_A, B) != 0:
            flag_loop_valid = True
        else:
            return False
    else:
        return False

    BotoAs_P = n_edges(net, output_B, start_A)
    AetoBi_P = n_edges(net, end_A, input_B)
    M_P = max(BotoAs_P, AetoBi_P)



    c1 = n_edges(net, {'start'}, B, scaling=scores_toggle)
    c1 += n_edges(net, B, {'end'}, scaling=scores_toggle)

    c2 = n_edges(net, A - end_A, B, scaling= scores_toggle)

    c3 = n_edges(net, B, A - start_A, scaling=scores_toggle)

    c4 = 0
    if len(output_B) != 0:
        for a in start_A:
            for b in output_B:
                c4 += max(0, scores[(b, a)] * M_P * sup * (n_edges(net,{'start'},{a})/n_edges(net, {'start'}, start_A)) * (n_edges(net, {b}, start_A)/ n_edges(net, output_B, start_A))- n_edges(net, {b}, {a}, scaling=scores))

    c5 = 0
    if len(input_B) != 0:
        for a in end_A:
            for b in input_B:
               c5 +=  max(0, scores[(a,b)] * M_P * sup * (n_edges(net,{a}, {'end'})/n_edges(net, end_A, {'end'})) * (n_edges(net, end_A, {b})/ n_edges(net, end_A, input_B))- n_edges(net, {a}, {b}, scaling=scores))



    if sup*M_P==0:
        return False
    if (c4+c5)/(2*sup*M_P)>0.3:
        return False

    return c1 + c2 + c3 + c4 + c5

def cost_loop_relation(net, A, B, sup, flow, start_A, end_A, input_B, output_B, dic_indirect_follow_log, calc_repetition_Factor):
    # TODO SUP
    scores = []
    for x in A:
        for y in B:
            if (x in end_A and y in input_B) or (x in start_A and y in output_B):
                # redo s
                dividend1 = get_direct_edge_count(net,x,y)
                divisor1 = dic_indirect_follow_log[y][x] + 1
                dividend2 = dic_indirect_follow_log[y][x]
                divisor2 = get_direct_edge_count(net,x,y) + 1
                scores.append( min((dividend1/divisor1), (dividend2/divisor2)) )
            else:
                # redo i
                dividend1 = dic_indirect_follow_log[x][y]
                divisor1 = dic_indirect_follow_log[y][x] + 1
                dividend2 = dic_indirect_follow_log[y][x]
                divisor2 = dic_indirect_follow_log[x][y] + 1
                scores.append( min((dividend1/divisor1), (dividend2/divisor2)) )
    return average(scores) + ( standard_deviation(scores) * (1 - min(calc_repetition_Factor,1) ) )


def visualisecpcm(cuts, ratio, size_par):
    cp = [x[2] for x in cuts]
    cm = [x[3] for x in cuts]
    tt = [str(x[1])+", "+str(x[0]) for x in cuts]
    diff = [x[2] - ratio * size_par * x[3] for x in cuts]
    color_fit = [x[5] for x in cuts]
    min_value = min(diff)
    min_index = diff.index(min_value)
    edge = [20]*len(diff)
    edge[min_index]= 100

    fig, ax = plt.subplots()
    s = ax.scatter(cp, cm, c=color_fit, cmap='inferno',s=edge)
    ax.set_xlabel(r'cp', fontsize=15)
    ax.set_ylabel(r'cm', fontsize=15)
    fig.colorbar(s, ax=ax)



    from matplotlib.widgets import Cursor
    # Defining the cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)

    # cursor grid lines
    lnx = plt.plot([60, 60], [0, 1.5], color='black', linewidth=0.3)
    lny = plt.plot([0, 100], [1.5, 1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('--')
    lny[0].set_linestyle('None')
    # annotation
    annot = ax.annotate("", xy=(0, 0), xytext=(5, 5), textcoords="offset points")
    annot.set_visible(False)
    # xy limits
    plt.xlim(min(cp) * 0.95, max(cp) * 1.05)
    plt.ylim(min(cm) * 0.95, max(cm) * 1.05)

    def hover(event):
        # check if event was in the axis
        if event.inaxes == ax:
            cont, ind = s.contains(event)
            if cont:
                # change annotation position
                annot.xy = (event.xdata, event.ydata)
                print((event.xdata, event.ydata))
                print("{}".format(', '.join([tt[n] for n in ind["ind"]])))
                # write the name of every point contained in the event
                annot.set_text("{}".format('\n '.join([tt[n] for n in ind["ind"]])))
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                annot.set_visible(False)
        # else:
        #     lnx[0].set_visible(False)
        #     lny[0].set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def check_base_case(detected_cut, logP, logM, sup_thr, ratio, size_par):
    activitiesP = set(a for x in logP.keys() for a in x)

    if len(activitiesP) <= 1:
        base_check = True
        counter = logP[()]
        counterM = logM[()]
        len_logP = sum(logP.values())
        acc_contP = sum([len(x) * logP[x] for x in logP])
        len_logM = sum(logM.values())
        acc_contM = sum([len(x) * logM[x] for x in logM])

        # empty check
        if (counter == len_logP) or (len_logP == 0):
            detected_cut = 'empty_log'
            cut = ('none', 'empty_log', 'none', 'none')
        else:
            # xor check
            cost_single_exc = max(0, sup_thr * len_logP - counter) - ratio * size_par * max(0,sup_thr * len_logM - counterM)
            if (counter > (sup_thr / 2) * len_logP) and (cost_single_exc <= 0):
            # if (cost_single_exc <= 0):
                cut = (({activitiesP.pop()}, set()), 'exc', 'none', 'none')
            else:
                # loop check
                del logP[()]
                if acc_contP > 0:
                    p_prime_Lp = (len_logP - counter) / ((len_logP - counter) + acc_contP)
                else:
                    p_prime_Lp = 'nd'

                if acc_contM > 0:
                    p_prime_Lm = (len_logM - counterM) / ((len_logM - counterM) + acc_contM)
                else:
                    p_prime_Lm = 'nd'

                if p_prime_Lm != 'nd':
                    cost_single_loop = max(0, sup_thr/2 - abs(p_prime_Lp - 0.5)) - ratio * size_par * max(0,sup_thr/2 - abs(p_prime_Lm - 0.5))
                else:
                    cost_single_loop = max(0, sup_thr/2 - ratio * size_par * abs(p_prime_Lp - 0.5))

                if (abs(p_prime_Lp - 0.5) > sup_thr / 2) and (cost_single_loop <= 0):
                # if (cost_single_loop <= 0):
                    cut = (({activitiesP.pop()}, set()), 'loop1', 'none', 'none')
                else:
                    # single activity
                    detected_cut = 'single_activity'
                    cut = ('none', 'single_activity', 'none', 'none')
    else:
        base_check = False
        cut = "not_base"

    return base_check, cut, detected_cut

def cost_loop_tau_relation(start_acts, end_acts,dfg, log, start_activities_o, end_activities_o):
    score = 0
    for x in start_acts:
        for y in end_acts:
            activityUniqueness = (start_activities_o[x] / (sum(start_activities_o.values()))) * (end_activities_o[y] / (sum(end_activities_o.values())))
            loopRatio = min(1 , dfg[(y, x)] / len(log))
            score += activityUniqueness * loopRatio
    return score

def check_relation_base_case(netP, netM, log, logM, sup, ratio, size_par, dfgP, dfgM, activity_key, start_acts_P, end_acts_P, start_activities,end_activities):
    activitiesP = netP.nodes - {'start', 'end'}
         
    # xor tau
    cost_exc_tau_P = cost_exc_tau_relation(netP, log)
    if cost_exc_tau_P > sup:
        cost_exc_tau_M = cost_exc_tau_relation(netM, logM)
        return True, ((activitiesP, set()), 'exc_tau', cost_exc_tau_P, cost_exc_tau_M,cost_exc_tau_P - ratio * cost_exc_tau_M,1), 'none', 'none'
        
    
    # strict loop_tau
    start_acts_P = set([x[1] for x in dfgP if (x[0] == 'start')])-{'end'}
    end_acts_P = set([x[0] for x in dfgP if (x[1] == 'end')])-{'start'}
    if cost_loop_tau_relation(start_acts_P,end_acts_P,dfgP,log,start_activities,end_activities) > sup:
        strict_tau_loop, new_log_P = fall_through.strict_tau_loop(log, start_acts_P, sup, end_acts_P, activity_key)
        strict_tau_loopM, new_log_M = fall_through.strict_tau_loop(logM, start_acts_P, sup, end_acts_P, activity_key)
        return True, ((start_acts_P, end_acts_P), 'loop_tau', 'none', 'none'), new_log_P, new_log_M
    
    return False, "not_base", 'none', 'none'


def remove_infrequent_edges(dfg, end_activities, threshold, show_pruning = False):
    node_set = set()
    
    # Step 1: Determine the out-edge frequency for each node
    number_edges = len(dfg)
    edge_max_out = {}
    for (u, v), weight in dfg.items():
        node_set.add(u)
        node_set.add(v)
        
        if u not in edge_max_out:
            edge_max_out[u] = weight
        else:
            edge_max_out[u] = max(weight, edge_max_out[u])
        if u in end_activities:
            edge_max_out[u] = max(edge_max_out[u], end_activities[u])
    
    dfg_list = [(x, y) for x, y in dfg.items()]
    dfg_list_filtered = dfg_list.copy()
    
    # generate nx graph
    nx_graph = generate_nx_graph_from_dfg(dfg)
    
    for x in dfg_list:
        if x[1] < threshold * edge_max_out[x[0][0]]:
            nx_graph.remove_edge(*x[0])
            # check start end reachability
            if nx.has_path(nx_graph,"start","end"):
                dfg_list_filtered.remove(x)
            else:
                nx_graph.add_edge(*x[0])

    dfg_list_filtered = [x[0] for x in dfg_list_filtered]
    # filter the elements in the DFG
    graph = {x: y for x, y in dfg.items() if x in dfg_list_filtered}
    
    number_edges_after = len(dfg_list_filtered)
    if show_pruning == True:
        print("Pruning Edges: " + str(number_edges) + " -> " + str(number_edges_after))
      
    return graph


def find_possible_partitions(net):
    
    time_search_start = time.time()
    def adj(node_set, net):
        adj_set = set()
        for node in node_set:
            adj_set = adj_set.union(set(net.neighbors(node)))
        return adj_set

    activity_list = set(net.nodes)-{'start','end'}

    queue = []
    queue.append((set(), {'start'}))
    visited = []
    valid = []
    while len(queue) != 0:
        current = queue.pop()
        for x in current[1]:
            new_state = current[0].union({x})
            new_state = add_SE(net, new_state)

            if new_state not in visited:
                new_adj = current[1].union(adj({x},net)) - new_state
                if new_state not in visited:
                    queue.append((new_state, new_adj))
                visited.append(new_state)
                B = activity_list - new_state
                if (len(B) == 0) or (len(B) == len(activity_list)):
                    continue
                B = add_SE(net, B)
                BB = net.subgraph(B)
                if 'end' in B:
                    disc_nodes_BB = set(BB.nodes) - set(nx.ancestors(BB, 'end')) - {'end'}
                    if len(disc_nodes_BB) == 0:
                        if ('end' in new_state) and ('start' in B) and (B not in visited):
                            valid.append((new_state, B, {"seq", "exc", "par", "loop"}))
                        elif 'end' in new_state:
                            valid.append((new_state, B, {"loop", "seq"}))
                        else:
                            valid.append((new_state, B, {"seq"}))

                    else:
                        new_A = new_state.union(disc_nodes_BB)
                        net_new_A = net.subgraph(new_A)
                        new_B = B - disc_nodes_BB
                        if len(drop_SE(new_B)) != 0 and (new_A not in visited):
                            if r_to_s(net_new_A) and (new_A not in visited):
                                visited.append(new_A)
                                if ('end' in new_A) and ('start' in new_B) and (new_B not in visited):
                                    valid.append((new_A, new_B, {"seq", "exc", "par", "loop"}))
                                elif 'end' in new_A:
                                    valid.append((new_A, new_B, {"loop", "seq"}))
                                else:
                                    valid.append((new_A, new_B, {"seq"}))
                                queue.append((new_A, new_adj.union(adj(disc_nodes_BB, net)) - new_A))

                if ('end' not in BB) and ('start' not in BB):
                    if nx.is_weakly_connected(BB):
                        valid.append((new_state, B, {"loop"}))
    time_search_end = time.time()
    # print("searching time = " + str(time_search_end-time_search_start))
    return valid


def max_flow_graph(net):
    flow_graph = {}
    for x in net.nodes:
        for y in net.nodes:
            if (x != y):
                flow_graph[(x, y)] = nx.algorithms.flow.maximum_flow(net, x, y, capacity='weight')[0]
    return flow_graph



def noise_filtering(dfg0, nt):
    dfg = copy.deepcopy(dfg0)
    log_size = sum([dfg[x] for x in dfg if x[0] == 'start'])
    noisy_edges = sorted([(x,dfg[x]) for x in dfg if (dfg[x]/log_size) < nt], key=lambda z:z[1])
    net = generate_nx_graph_from_dfg(dfg0)
    for ne in noisy_edges:
        net_copy = copy.deepcopy(net)
        nodes_set = set(net_copy.nodes)
        net_copy.remove_edge(ne[0][0],ne[0][1])
        if (set(nx.ancestors(net_copy, 'end')) == nodes_set-{'end'}):
            if(set(nx.descendants(net_copy, 'start')) == nodes_set-{'start'}):
                del dfg[ne[0]]
                net = net_copy
    return dfg


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