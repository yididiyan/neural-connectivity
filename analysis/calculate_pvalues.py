import sys 

import os 
import logging
import numpy as np
import pandas as pd 
import pickle 
import argparse

from multiprocessing import Pool

from statsmodels.stats.multitest import multipletests as mt 



sys.path.insert(0, os.path.abspath('../code'))
from utils import pickle_object, read_pickle_file
from config import Configuration



sys.path.insert(0, os.path.abspath('../analysis'))
from permutation_test import edges_statistics, unit_to_region, one_run_sample, get_pval_diff_med, calculate_p_value





def correct_pvals(p_values, n_groups = 3, significant_cutoff=0.05):
    n_exp = len(p_values.keys())

    original_pvals = [ p_val for exp in p_values.keys() for p_val in p_values[exp].values()]
    significance, corrected, _, _ = mt(original_pvals, method='fdr_bh')
    corrected, significance = corrected.reshape(n_exp, n_groups, n_groups), significance.reshape(n_exp, n_groups, n_groups)

    return corrected, np.logical_and(corrected, significance > 0.0)



def load_datasets(data_dir):
    # to parent directory 
    parent_dir = data_dir + '/../'

    post_15_data = pd.read_feather(parent_dir + 'post_15_spk_times.feather')
    post_9_data = pd.read_feather(parent_dir + 'post_9_spk_times.feather')
    pre_data = pd.read_feather(parent_dir + 'pre_spk_times.feather')


    return pre_data, post_9_data, post_15_data



def main(conf, raw_DI=False, n_runs=10**5, model_class='mdl'):
    '''
    :param model_class used for the analysis
    '''
    p_values = {}
    stimuli = [9, 15] ## pre 
    diffs = [[9, 15,], [15, 9] ] ## post - trained on 
 

    pre_data, post_9_data, post_15_data = load_datasets(conf.data_dir)

    data_dir = {
        9: '../../output_stimuli_9' + '/tts' if model_class =='tts' else '',
        15: '../../output_stimuli_15' + '/tts' if model_class =='tts' else '',
    }

    post_datasets = {
        9: post_9_data,
        15: post_15_data
    }


    for i, s in enumerate(stimuli):
        for d in diffs[i]:
            
    
            post_dataset = post_datasets[d]
            print("Working on Stimuli: {} trained on {} ".format(s, d))
            p_values['{}_{}'.format(s, d)] = calculate_p_value(
                pre_data=pre_data[pre_data.stimn==s],
                post_data=post_dataset[post_dataset.stimn==s],
                pre_connections='{}/{}/pre_spk_times/{}/500_1500/connectivity_pre_spk_times__useallunits__normalized_.pkl'.format(data_dir[s],  'raw_group_connectivity' if raw_DI else 'group_connectivity', s ),
                post_connections='{}/{}/post_{}_spk_times/{}/500_1500/connectivity_post_{}_spk_times__useallunits__normalized_.pkl'.format(data_dir[s], 'raw_group_connectivity' if raw_DI else 'group_connectivity', d, s, d ),
                n_runs=n_runs
            )

        ## post_s - post_d 

        d = 15 if s == 9 else 9
        post_dataset_1 = post_datasets[s]
        post_dataset_2 = post_datasets[d]
        
        p_values['{}_post_diff'.format(s)] = calculate_p_value(
            pre_data=post_dataset_1[post_dataset_1.stimn==s],
            post_data=post_dataset_2[post_dataset_2.stimn==s],
            pre_connections='{}/{}/post_{}_spk_times/{}/500_1500/connectivity_post_{}_spk_times__useallunits__normalized_.pkl'.format(data_dir[s],  'raw_group_connectivity' if raw_DI else 'group_connectivity', s, s, s ),
            post_connections='{}/{}/post_{}_spk_times/{}/500_1500/connectivity_post_{}_spk_times__useallunits__normalized_.pkl'.format(data_dir[s], 'raw_group_connectivity' if raw_DI else 'group_connectivity', d, s, d ),
            n_runs=n_runs
        )


        # Write them values :)
        pickle_object(p_values, data_dir[s] + '/p_values{}_{}.pkl'.format( '_raw' if raw_DI else '', model_class ))



    
    


    
if __name__ == "__main__":
    logging.basicConfig(level= logging.INFO )
    parser = argparse.ArgumentParser(description='Calculate p values ')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml',)
    parser.add_argument('--raw-di', type=lambda x: (str(x).lower() == 'true'), default=False,)
    parser.add_argument('--n-runs', type=int, default=10**3,)
    parser.add_argument('--model', type=str, action='store', choices=['tts', 'mdl'], default='mdl',)

    
    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config, model_class=args.model)
    
    main(conf, raw_DI=args.raw_di, n_runs=args.n_runs, model_class=args.model)    
    

