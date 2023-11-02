# -*- coding: utf-8 -*-
"""
This code will pick the parent sets that have the strongest influence.


@author: cjquinn
"""

import logging
import os
import sys
import pickle
import random
import glob
import argparse
from pathlib import Path
from multiprocessing import Pool
from multiprocessing import cpu_count

import pandas as pd

sys.path.insert(0, os.path.abspath('../code'))
from utils import pickle_object, read_pickle_file
from config import Configuration




class BestParentSetSelection():
    def __init__(self, config, dataset, stimulus_id=None, raw_DI=False): 
        self.config = config 
        self.dataset = dataset 
        self.stimuls_id = stimulus_id 
        self.raw_DI = raw_DI


        ## Search options 
        self.options = [
            'USEALLUNITS', 
            # 'SINGLEPAR',
            # 'CLUSTER',
            'NORMALIZED',
        ]

        self.data_dir = dataset.data_dir 

        ## Load data 
        self.data = pd.read_feather(dataset.preprocessed_path)

        ## Set of observation window we are interested in 
        self.times = [config.observation_window_string()]
        self.part = ['all2all']


        ## The units we're looking at 
        self.units = self.data.id.unique()
        


    def __call__(self):
        for time in self.times:
            print('Working on time ', time)
            self.save_best_parent_set(time) # works on each cell separately and saves    
            self.combine_best_parent_sets(time) # read_bestpar_pkls(inpt) #then combines together




    def get_best_parent_path(self, time):
        options = ''.join([ '_' + option.lower() + '_' for option in self.options ])
        
        return self.dataset.best_parent_path(time, options, stimulus_id=self.stimuls_id, raw_DI=self.raw_DI)




    def get_aggregate_best_parent_filepath(self, time):
        options = ''.join([ '_' + option.lower() + '_' for option in self.options ])


        return self.dataset.aggregate_best_parent_filepath(time, options, stimulus_id=self.stimuls_id, raw_DI=self.raw_DI)


    def combine_best_parent_sets(self, time):
        '''
        Loads and combines individual pickles(best parents sets for units) 
        into one big pickle(best parent sets of all units) 
        '''
        ## TODO: make part also configurable, i beleive it should be globally configurable instead of having multiple "parts" running at a time
        part = self.part[0]
        
        # dir_best = dirs['data_dir'] + os.sep + dirs['bestpar']
        dir_best = self.get_best_parent_path(time)

        files = glob.glob(dir_best + '/ET*.pkl')
        best_pars = {}    

        # pool = Pool(processes= max(1,cpu_count()-2))

        results = [self.load_parents(file) for file in files]
        
        # pool.close()
        # pool.join() 


        for i in range(len(files)):
            
            cellY = files[i][len(dir_best)+1:-4]
            best_pars[cellY] = {'pars':results[i][0], 'val':results[i][1]}
        
        ## Save aggregate dictionary 
        aggregate_parent_filepath = self.get_aggregate_best_parent_filepath(time)
        print('Aggregating best parent set at ', aggregate_parent_filepath)
        pickle_object(best_pars, aggregate_parent_filepath)
        print('Done aggregating best parent set...')
        
    def load_parents(self, fname):
        '''
        Load file containing best parents 
        '''      
        best_pars = read_pickle_file(fname)
        if 'pars' not in best_pars.keys():
            print(fname)
        
        return (best_pars['pars'], best_pars['val'])


    def save_best_parent_set(self, time):
        units = self.data.id.unique()
        
        
        random.seed(os.urandom(1000))

        random.shuffle(units)
        
        
        
        
        # Serial
        for unit in units:
            print('Working on unit ', unit)
            self.get_parents(time, unit)
        
        print('Completed saving best parent sets!')
        # Parallel 
        # with Pool(processes=23) as pool:
        #     pool.map(self.get_parents, unit)
            


    def get_parents(self, time, unit):
        ## TODO: For now we focus on all2all connections; check this with Dr. Quinn
        if 'all2all' in self.part:
            return self.get_parents_all2all(time, unit)
    #    else:
    #        return get_parents_grouplayer(tpl)



    # no restrictions on which layers parents coming from
    def get_parents_all2all(self, time , unit):
        cellY = unit
        


        dir_DI = self.dataset.DI_values_path(time, self.stimuls_id)
        ## make dir_DI if not exists 
        os.makedirs(dir_DI, exist_ok=True)

        # dir_DI = get_DIvalues_directory(inpt['dataset'], inpt['data_dir'], inpt['time'])

        # dir_DI = dirs['data_dir'] + os.sep+ dirs['save_DIs']


        fnameDI = 'DI_dict_'+unit+'.pkl'

        
        fnamebest = '{}/{}.pkl'.format(self.get_best_parent_path(time), cellY)
        
        #if already did this or don't have DI file, skip
        if os.path.exists(fnamebest):
            return None
        if fnameDI not in os.listdir(dir_DI):
            print (cellY + ' DI calcs not started yet')
    #        print('ERROR:  missing '+fname)
            return
        
        
        try: 
            with open(dir_DI + os.sep +fnameDI, 'rb') as f:
                DI = pickle.load(f)   
        except Exception as e:
            print('Skip reading {}'.format(dir_DI + os.sep + fnameDI))
            print('Reason {}'.format(e))
            return 


        DI = DI[cellY]

        
        best_pars = ()
        best_val = 0.0
        norm_factor = 1.0

        if not self.check_DI_calculations(DI):
            logging.warning("DI calculations for {} are not completed ".format(cellY))
            return 


        if 'NORMALIZED' in self.options:
            
            #grab any tested parent set element - the estimate will be there
            #avoid the first few keys which are not subsets of units
            tmp_keys = [key for key in DI.keys() if len(key)==1]
            
            if len(tmp_keys)==0:
                print('no parent sets checked??')
                return None
            
            else:
                
                ## if DI is empty 
                if DI[tmp_keys[0]] is None:
                    print('DI value for {} is not calculated'.format(cellY))
                    return None
                norm_factor = DI[tmp_keys[0]]['HY']/100 #the 100 converts to percentage
            
            
        
            if norm_factor < 0.00001:
                print('low HY')
                return None
        
        
        # #check if DI calculations completed
        # if (3,'all2all') not in DI.keys() or 'completed' not in DI[(3,'all2all')].keys() or not DI[(3,'all2all')]['completed']:
        #     print (cellY + ' DI calcs not completed yet')
        #     return None
        
        #we did a greedy search but with large numbers (not just top 1) so treat like did exhaustive for multiple sizes; in that case best is simply one with largest DI_discounted_MDL
            
        for key in DI.keys():
            
            if not isinstance(key,tuple) or  isinstance(key[0],int):
                continue
            
            if 'SINGLEPAR' in self.options and len(key)>1:
                continue
            
            if not DI[key]: 
                print('Skipping {}, Reason: None found'.format(key))
                continue
            if self.raw_DI:
                val = DI[key]['DI_X_Y']/norm_factor
            else:
                val = DI[key]['DI_discounted_MDL']/norm_factor

            if val>best_val:
                best_val = val
                best_pars = key
                
        #save it
        logging.info('Best value for {} is {}'.format(cellY, best_val))
        pickle_object({'pars':best_pars, 'val':best_val}, fnamebest)
        # with open(dir_best + os.sep + fnamebest,'wb') as f:
        #     pickle.dump({'pars':best_pars, 'val':best_val},f)

        
        return
        


    def check_DI_calculations(self, DI):
        '''
        Checks if DI calculations are done for specific cell 
        '''

        if (3,'all2all') not in DI.keys() or 'completed' not in DI[(3,'all2all')].keys() or not DI[(3,'all2all')]['completed']:
            print ('DI calcs not completed yet')
            return False
        
        return True
        


def main(config, raw_DI=False):
    '''
    :param raw_DI use raw DI values instead of discounted one
    '''
    ## Grab datasets 
    datasets = config.datasets
    
    for dataset in datasets:
        data = pd.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())


        for stimulus_id in stimuli:
            print("Working on stimulus : {}".format(int(stimulus_id)))
                
            selection = BestParentSetSelection(config, dataset, stimulus_id=int(stimulus_id), raw_DI=raw_DI)
            selection() ## Run selection 



    
    


if __name__ == "__main__":
    logging.basicConfig(level= logging.INFO )
    parser = argparse.ArgumentParser(description='Select best parent sets')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml',)
    parser.add_argument('--raw-di', type=lambda x: (str(x).lower() == 'true'), default=False,)
    parser.add_argument('--model', action='store', type=str, default='mdl', choices=['mdl', 'tts']) ## mdl vs tts 
    
    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config, model_class=args.model )
    
    main(conf, raw_DI=args.raw_di)    
    
