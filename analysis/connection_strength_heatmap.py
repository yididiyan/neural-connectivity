# -*- coding: utf-8 -*-
"""
This generates heatmap values which are later plotted.

@author: cjquinn
"""

import sys
import os
import argparse

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 15})

sys.path.insert(0, os.path.abspath('../code'))
from utils import pickle_object, read_pickle_file
from config import Configuration



cmap_plot_options = {
    'vmin': 0.0, 
    # 'vmax': 2.5,
    'vmax': 5

}



class ConnectionStrengthHeatmap:
    def __init__(self, config, dataset, data, stimulus_id=None, raw_DI=False): 
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
        self.options_string = ''.join([ '_' + option.lower() + '_' for option in self.options ])

        self.data_dir = dataset.data_dir 

        self.part = ['all2all']

        ## Set of observation window we are interested in 
        self.times = [config.observation_window_string()]

        ## The actual pandas data table 
        self.data = data

        if self.stimuls_id:
            self.data = self.data[self.data.stimn == stimulus_id]

        
        ## The units we're looking at 
        self.units = self.data.id.unique()


        ## Groups 
        self.groups = sorted(self.data.region.unique().tolist()) ## Synonymous to region 


        ## load unit to region pickle 
        self.unit_to_region = None

        with open(self.dataset.unit_to_region_path, 'rb') as f:
            self.unit_to_region = pickle.load(f)

    def __call__(self):
        print('Calculation connection stregth...')
        self.calculate()


    def calculate(self):
        
        adjacency_array = {} #holds vectors of values for each relationship
        adjacency = {} #holds the median values

        for time in self.times:
            print('Calculating strength for time {}'.format(time))
            adjacency_array[time], adjacency[time] = self.calculate_raw_connection_prob(time)

            if adjacency_array[time] is None or  adjacency[time] is None: 
                print('Skipping, no adjacency array is empty empty')
                continue ## skip, no data yet         

            ## Now, save the connectivity file 
            tmp = {'adjacency': adjacency[time], 'adjacency_array': adjacency_array[time], 'options': self.options}
            print('Connection Strength: Saving result at', self.dataset.connection_strength_filepath(time, self.options_string, self.stimuls_id, raw_DI=self.raw_DI))
            pickle_object(tmp, self.dataset.connection_strength_filepath(time, self.options_string, self.stimuls_id, raw_DI=self.raw_DI))

            ## Plot histogram 
            self.plot_heatmap(time, adjacency_array[time])

            
                

    def calculate_raw_connection_prob(self, time):

        Adj_actual = self.get_Adj_actual(time)
        
        return Adj_actual    
    
    



    def get_Adj_actual(self, time):
        
        

        Adj = {}
        for groupX in self.groups:
            for groupY in self.groups:
                Adj[groupX, groupY] = []

        
       
        ## Load best parent aggregate file 
        options = ''.join([ '_' + option.lower() + '_' for option in self.options ])
        fname = self.dataset.aggregate_best_parent_filepath(time, options, stimulus_id=self.stimuls_id, raw_DI=self.raw_DI)


        best_pars = read_pickle_file(fname)

        if best_pars is None:
            print('Best parents not yet identified')
            return None, None
        
        
        for cellY in self.units:        
            
            if cellY not in best_pars.keys():
                print('Missing from best_pars: '+cellY)
                continue
            
            groupY = self.unit_to_region[cellY]
            
            pars = best_pars[cellY]['pars']
            
            for cellX in pars:
                
                groupX = self.unit_to_region[cellX]
                            
                Adj[groupX, groupY].append( best_pars[cellY]['val']/len(pars) )
                
            
        #convert Adj to a numpy array
        Adj_array = np.zeros( (len(self.groups), len(self.groups)) )
        Adj_orig = {}

        
        for groupX in self.groups: #rows (in the pictures)
            for groupY in self.groups: # columns (in the pictures)
                
                row = self.groups.index(groupX)
                col = self.groups.index(groupY)            
                Adj_orig[groupX, groupY] = Adj[groupX, groupY]
                            
                list_vals = sorted( Adj[groupX, groupY] , reverse=True)
                
                        
                list_vals = [i for i in list_vals if i>0.0] #only keep positive vals

                Adj[groupX, groupY] = list_vals 
                            
                if len(list_vals)<3: #make sure at least 3 values
                    Adj_array[row, col] = 0.0
                else:
                    Adj_array[row, col] = np.median(list_vals)

        print(Adj_array)
        return Adj_array, Adj
        


    def plot_heatmap(self, time, Adj_array):
        fig, ax = plt.subplots()
        im = ax.imshow(Adj_array, 'Reds', vmin=cmap_plot_options['vmin'], vmax=cmap_plot_options['vmax'])

        # Add ticks 
        ax.set_yticks(np.arange(len(self.groups)), labels=self.groups)
        ax.set_xticks(np.arange(len(self.groups)), labels=self.groups)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        ## Add color bar 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ## Add title 
        ax.set_title('{}, stimulus {}, {} {}'.format(
            self.dataset.name, 
            self.stimuls_id, 
            time, 
            '\n raw DI' if self.raw_DI else '' ))
        fig.tight_layout()

        ## Save figure 
        options = ''.join([ '_' + option.lower() + '_' for option in self.options ])
        plot_file = self.dataset.connection_strength_plot_filepath(time, options, stimulus_id=self.stimuls_id, raw_DI=self.raw_DI)
        print('Saving heatmap at {}'.format(plot_file))
        plt.savefig(plot_file,bbox_inches='tight')
        






def main(config, raw_DI=False):
    ## Grab datasets 
    datasets = config.datasets
    
    for dataset in datasets:
        data = pd.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())


        for stimulus_id in stimuli:
            print("Working on dataset {}, stimulus {}".format(dataset, int(stimulus_id)))
            
                
            connectionHeatmap = ConnectionStrengthHeatmap(config, dataset, data, stimulus_id=int(stimulus_id), raw_DI=raw_DI)
            connectionHeatmap() ## Run selection 





        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate adjacency')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml',)
    parser.add_argument('--raw-di', type=lambda x: (str(x).lower() == 'true'), default=False,)
    parser.add_argument('--model', action='store', type=str, default='mdl', choices=['mdl', 'tts']) ## mdl vs tts 
    

    args = parser.parse_args()
    ## Pull configuration 
    conf = Configuration(args.config, model_class=args.model)
    
    main(conf, raw_DI=args.raw_di)    
    
