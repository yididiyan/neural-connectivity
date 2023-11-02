## Neural Connectivity
This project is an implementation to analyse directed information flows using data from simultaneous recorded visual areas.  

## Requirements 

* Python 3.8 

We can easily create the required Python environment using Conda as follows. 
```
conda create -n <env-name> python=3.8

conda activate <env-name>
```
Replace `<env-name>` with the desired environment name.


## Install Python libraries  
```
pip install -r requirements.txt
```


# Organization 
Folders 


* `code` - contains the main implementation to identify the directed infromation(DI) values between neuron groups(layers, cortical areas etc.) 

* `analysis` (and `notebooks/lasso-method` )  -  contain scripts to generate the heatmaps and other analysis results




## Steps 
The following commands automate the process required for analysis 
### 1) Creating Configuration files 

The following step create the configuration files necessary for the analysis 

```
cd code 
python create_configuration.py --stimuli 9  --dataset <dataset-path> --output_dir ../../output 
```

The above commands create a configuration file  `../../output_stimuli_9/runner_<dataset-name>.yml`



### 2) Preprocessing data 

This step transforms the raw data to the expected format in the analysis.  


```
python preprocess_data.py --config <config-path> 
python prefilter_spikes.py --config <config-path> 
python find_active_times.py --config <config-path>
```

### 3) Calculating Directed Information 

```
python lasso_select_parentset.py --config <config-path>
```

### 4) Postprocessing 