## Neural Connectivity
This project is an implementation to analyse directed information flows in multi-area recording using data from a neurosciene experiment. 

## Requirements 
### Easily create Python environment using Conda 
```
conda create -n <env-name> python=3.8

conda activate <env-name>
```
Replace <env-name> with the desired environment name.


## Install Python requirements 
```
pip install -r requirements.txt
```


# Description 
Folders 

* analysis -  contains scripts tho generate the heatmaps and other analysis results
* code - has the main implementation to identify the directed infromation(DI) values between neuron groups(layers, cortical areas etc.) 




## Server sbatch steps 
sbatch submission-create-configuration.bash <dataset-file> <stimulus_id>
sbatch submission-preprocessing.bash /worktmp/LAS/cjquinn-lab/Yididiya/output_n_mice_5_stimuli_9/runner_config.yml
sbatch submission-prefiltering.bash /worktmp/LAS/cjquinn-lab/Yididiya/output_n_mice_5_stimuli_9/runner_config.yml 
sbatch submission-finding-active-times.bash /worktmp/LAS/cjquinn-lab/Yididiya/output_n_mice_5_stimuli_9/runner_config.yml 
sbatch submission-finding-parents.bash /worktmp/LAS/cjquinn-lab/Yididiya/output_n_mice_5_stimuli_9/runner_config.yml 
sbatch submission-post-processing.bash /worktmp/LAS/cjquinn-lab/Yididiya/output_n_mice_5_stimuli_9/runner_config.yml 
sbatch submission-calculate-di-tts.bash <config-yaml-file> <best-parentset-pkl-file>


## Steps 
### 1) Creating Configuration files 

The following step create the configuration files necessary for the analysis 

```
cd code 
python create_configuration.py --stimuli 9  --dataset ../../YuTang/2022-spring-Yu-v01/data/pre_spk_times.feather --output_dir ../../output --n_recordings 1
```


### 2) Preprocessing data 

This step transforms the row data to the expected format in the analysis.  

python preprocess_data.py --config ../../output_n_mice_1_stimuli_9/runner_config.yml  

### 

python prefilter_spikes.py --config /Users/yido/Code/neuroscience/output_n_mice_1_stimuli_9/runner_config.yml
python find_active_times.py --config ../../output_n_mice_1_stimuli_9/runner_config.yml
python search_parents_optimized.py --config /Users/yido/Code/neuroscience/output_n_mice_1_stimuli_9/runner_config.yml
cd ../analysis
python best_parentset_selection.py --config /Users/yido/Code/neuroscience/output_n_mice_1_stimuli_9/runner_config.yml
python connection_strength_heatmap.py --config /Users/yido/Code/neuroscience/output_n_mice_1_stimuli_9/runner_config.yml


## Local steps 
### Calculating DI for TTS method with bestparent set as those selected using MDL method 
python calculate_di_values.py --config <config-yaml-file> --model tts --best-parentset <best-parentset-pkl-file>


## Calculating p-values 
