import sys
import os
from tokenize import group
from matplotlib.pyplot import get
import numpy as np
import pandas as pd 
import argparse

from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests as mt 

sys.path.insert(0, os.path.abspath('../code'))
from utils import pickle_object, read_pickle_file

groups = ['AL', 'LM', 'V1']


def unit_to_region(unit):
    '''
    Simple function to map unit to its corresponding region/group (AL, LM)
    '''
    return unit.split('_')[-2]


def edges_statistics(data, connections, groups=groups):
    '''
    Determines the count of possible edge and ratio of actual edges to possible edges 

    :param data - dataset in pandas format
    :param connections - connectivity values - "adjacency" and "adjacency array" ordered according to groups 
    :param groups - list of groups to which units belong  
    '''

    
    counts= {}
    freq = {}
    
    recordings = data.recording.unique()
    
    ## Aggregate potential counts 
    for r in recordings: 
        ## units in the rcordings 
        units = data[data.recording == r].id.unique()
        group_list = [unit_to_region(u) for u in units] 
        
        
        klim = 3 
        
        for groupX in groups:
            for groupY in groups:
                if (groupX, groupY) not in counts.keys():
                    counts[(groupX, groupY)] = 0
                    
                count_X = min(klim, group_list.count(groupX))
                count_Y = group_list.count(groupY)
                
                
                if groupX != groupY or count_Y > count_X:
                    tmp_count = count_X * count_Y
                elif groupX == groupY and count_Y == count_X:
                    tmp_count = count_Y * (count_Y - 1)
                counts[(groupX, groupY)] += tmp_count


    ## Now look for actual counts 
    for groupX in groups: 
        for groupY in groups:
            if counts[(groupX, groupY)] == 0:
                freq[(groupX, groupY)] = {}
            else:
                freq[(groupX, groupY)] = len(connections['adjacency'][groupX, groupY]) * 1. / counts[groupX, groupY]
                

    return counts, freq





def one_run_sample(input_dic):
    #a single run 
    prop1 = input_dic['prop']
    edges = input_dic['edges']
    vals_1 = []
    vals_2 = []
    
    
    for edge in edges:
        if np.random.binomial(1,prop1):
            vals_1.append(edge)
        else:
            vals_2.append(edge)
    
    if len(vals_1)==0:
        vals_1 = [0]
    elif len(vals_2) == 0:
        vals_2 = [0]
    
    diff = np.median(vals_1) - np.median(vals_2)

    return diff
    
def get_pval_diff_med(edges1,edges2,prop_1, true_diff, n_runs=10**6):
    """
    This uses Monte Carlo sampling to approximate permutation test on all possible edges (not grouping by recording)    
    """  
    if len(edges2)+ len(edges1) == 0:
        return 1.0
    
    input_dic = {'edges':list(edges1)+list(edges2),'prop':prop_1}
#     numruns = 10**4#10**6 ## Question 2: how to configure the nrums  
    print('Using %i runs'%(n_runs))
    
    vals = []
    
    pool = Pool(processes=11)
    vals = pool.map(one_run_sample, [input_dic]*n_runs)
    
    pool.close()
    pool.join()
    
    
    #Now look at one/two sided
#    print('Two sided')
    vals = np.abs(vals)
    true_diff = np.abs(true_diff)

    tmp = [1. for val in vals if val >= true_diff]
    
    return sum(tmp)/len(vals)
    


def calculate_p_value(pre_data, post_data, pre_connections, post_connections, groups = ['AL', 'LM', 'V1'], n_runs=10**4):
    
    pre_connections = read_pickle_file(pre_connections)
    post_connections = read_pickle_file(post_connections)
    # import ipdb; ipdb.set_trace()
    connections_diff = post_connections['adjacency_array']  - pre_connections['adjacency_array'] 
    pre_counts, _ = edges_statistics(pre_data, pre_connections)
    post_counts, _ = edges_statistics(post_data, post_connections)
    pvals = {}
    
    
    available_group_pairs = pre_connections['adjacency'].keys()

    for groupX in groups:
        for groupY in groups:
            iX = groups.index(groupX)
            iY = groups.index(groupY)

            true_diff  = connections_diff[iX, iY]

            edges_pre = pre_connections['adjacency'].get((groupX, groupY), [])
            edges_post = post_connections['adjacency'].get((groupX, groupY), [])


            counts_pre = pre_counts[groupX, groupY]
            counts_post = post_counts[groupX, groupY]

            prop_pre = counts_pre * 1. / (counts_post + counts_pre) ## More on this
            pvals[groupX, groupY] = get_pval_diff_med(edges_pre, edges_post, prop_pre, true_diff, n_runs=n_runs)

        
        
    return pvals      



def correct_pvals(p_values, n_groups = 3):
    '''
    TODO: smelly code 
    '''

    n_experiments = len(p_values.keys())

    original_pvals = [ p_val for exp in p_values.keys() for p_val in p_values[exp].values()]
    significance, corrected, _, _ = mt(original_pvals, method='fdr_bh')
    corrected, significance = corrected.reshape(n_experiments, n_groups, n_groups), significance.reshape(n_experiments, n_groups, n_groups)

    return corrected, np.logical_and(corrected, significance > 0.0)




# def calculate_pvals(pre_dataset, post_dataset, pre_connections, post_connections, groups=groups):
#     # diff in connections 
#     connections_diff = post_connections['adjacency_array'] - pre_connections['adjacency_array']
    
#     # calculate edge statistics 
#     pre_counts, _ = edges_statistics(pre_dataset, pre_connections)
#     post_counts, _ = edges_statistics(post_dataset, post_connections)

#     pvals = {}
    
#     for groupX in groups:
#         for groupY in groups:
#             iX = groups.index(groupX)
#             iY = groups.index(groupY)

#             true_diff = connections_diff[iX, iY]

#             ## true edges 
#             pre_edges = pre_connections['adjacency'][groupX, groupY]
#             post_edges = post_connections['adjacency'][groupX, groupY]

#             ## Debug 
#             if pre_counts[groupX, groupY] < len(pre_edges) or post_counts[groupX, groupY] <  len(post_edges):
#                 import ipdb; ipdb.set_trace()
#             ## /Debug  

            


#             # proportion based on potential edges 
#             if (pre_counts[groupX, groupY] + post_counts[groupX, groupY]):
#                 proportion = pre_counts[groupX, groupY] * 1. / (pre_counts[groupX, groupY] + post_counts[groupX, groupY])
#                 pvals[groupX, groupY] = get_pval_diff_med(pre_edges, post_edges, proportion, true_diff)
#             else:
#                 pvals[groupX, groupY] = 1


#     print('P values', pvals)
#     return pvals



def main(args):
    # unpack args
    pre_dataset = getattr(args, 'pre_dataset')
    post_dataset = getattr(args, 'post_dataset')
    pre_connections = getattr(args, 'pre_connections')
    post_connections = getattr(args, 'post_connections')


    ## load connection files 
    pre_connections = read_pickle_file(pre_connections)
    post_connections = read_pickle_file(post_connections)

    # load datasets 
    pre_dataset = pd.read_feather(pre_dataset)
    post_dataset = pd.read_feather(post_dataset)
    
    pvals = calculate_pvals(pre_dataset, post_dataset, pre_connections, post_connections)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate pvalues')
    parser.add_argument('--pre-dataset', action='store', type=str, required=True)
    parser.add_argument('--post-dataset', action='store', type=str, required=True)
    parser.add_argument('--pre-connections', action='store', type=str, required=True)
    parser.add_argument('--post-connections', action='store', type=str, required=True)
    
    args = parser.parse_args()
    main(args)

    # get the "pre" and "post" connections 

# python permutation_test.py --pre-dataset /Users/yido/Code/neuroscience/output_n_mice_5_stimuli_15/pre_spk_times.feather 
# --post-dataset /Users/yido/Code/neuroscience/output_n_mice_5_stimuli_15/post_15_spk_times.feather 
# --pre-connections /Users/yido/Code/neuroscience/output_n_mice_5_stimuli_15/group_connectivity/pre_spk_times/15/500_1500/connectivity_pre_spk_times__useallunits__normalized_.pkl 
# --post-connections /Users/yido/Code/neuroscience/output_n_mice_5_stimuli_15/group_connectivity/post_15_spk_times/15/500_1500/connectivity_post_15_spk_times__useallunits__normalized_.pkl