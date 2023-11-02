import os
import sys
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap



sys.path.insert(0, os.path.abspath('../code'))
from utils import pickle_object, read_pickle_file

## plot cosmetic configuration 
plt.rcParams['figure.figsize'] = (25, 25)
plt.rcParams.update({'font.size': 50})



import pandas as pd 



def plot_heatmap( Adj_array, min_value=None, max_value=None, groups=['AL', 'LM', 'V1'], title=None, cmap=None):
    '''
    WIP: simple plot with just one heatmap 
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(Adj_array, cmap or 'Reds', vmin = min_value, vmax = max_value)


    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_yticks(np.arange(len(groups)), labels=groups)
    ax.set_xticks(np.arange(len(groups)), labels=groups)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    #     ax.set_title('Stimulus {}: Connection Strength between Groups [{}]'.format(self.stimuls_id + 1, time))
    fig.tight_layout()


    plt.show()


def plot_comparision(pre_file, post_file, post_file_2, title=None, titles=[], side_titles=[], groups=['AL', 'LM', 'V1'], window='500_1500', output='./output.pdf', pvals=None):
    '''
    Doing comparision plots 
    :param pre_file - pre connectivity pickle file 
    :param post_file - post connectivity pickle file 
    :param post_file_2 - another post connectivity pickle file 
    :param title - main title  Eg. Stimulus 9 
    :param titles - titles for individual plots  row-by-row 
    :param groups - list of groups to use in the plots 
    :param window - observation window 
    :param pvals - associated pvals 
    
    '''

    pre_result, post_result, post_result_2 = read_pickle_file(pre_file), read_pickle_file(post_file), read_pickle_file(post_file_2)
    pre_result, post_result, post_result_2 = pre_result['adjacency_array'], post_result['adjacency_array'], post_result_2['adjacency_array']
    
    if type(pre_result) == dict:
        pre_result = pre_result['500_1500']
    if type(post_result) == dict:
        post_result = post_result['500_1500']
    if type(post_result_2) == dict:
        post_result_2 = post_result_2['500_1500']
        

        
    def plot_heatmap_group(Adj00, 
                           Adj01, 
                           Adj10, 
                           Adj11, 
                           min_values=None, 
                           max_values=None, 
                           groups=groups, 
                           title='',
                           titles=[],
                           side_titles=[],
                           cmap=None,
                           pvals=None):
        assert min_values.shape == (2, 2) and max_values.shape == (2, 2)
        images = [[None, None], [None, None]]
        fig, axes = plt.subplots(2, 2)
        fig.suptitle(title)
        images[0][0] = axes[0, 0].imshow(Adj00, cmap or 'Reds', vmin=min_values[0, 0], vmax=max_values[0, 0])
        axes[0,0].title.set_text(titles[0])
        axes[0,0].set_ylabel(side_titles[0], fontsize=45, labelpad=20)
        images[0][1] = axes[0, 1].imshow(Adj01, cmap or 'Reds', vmin=min_values[0, 1], vmax=max_values[0, 1])
        axes[0,1].title.set_text(titles[1])
        images[1][0] = axes[1, 0].imshow(Adj10, cmap or 'Reds', vmin=min_values[1, 0], vmax=max_values[1, 0])
        axes[1,0].title.set_text(titles[2])
        axes[1,0].set_ylabel(side_titles[1], fontsize=45, labelpad=20)
        
        if Adj11 is not None:
            images[1][1] = axes[1, 1].imshow(Adj11, cmap or 'Reds', vmin=min_values[1, 1], vmax=max_values[1, 1])
            axes[1,1].title.set_text(titles[3])
        else:
            # delete axes 
            fig.delaxes(axes[1][1])

        
        for i in range(2):
            for j in range(2):
                if not images[i][j]:
                    continue
                ax = axes[i, j]
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(images[i][j], cax=cax, orientation='vertical')

                ax.set_yticks(np.arange(len(groups)), labels=groups)
                ax.set_xticks(np.arange(len(groups)), labels=groups)

                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        if not np.all(pvals == None): 
            for index, (i, j) in enumerate([[0, 0], [0, 1], [1, 0]]): ## first row, column 1 and 2 
                
                #add pvalues  
                ax = axes[i, j]

                for k in range(len(groups)):
                    for l in range(len(groups)):
                        # import ipdb; ipdb.set_trace()
                        pvalue = pvals[index,k,l]
                        if pvalue == 0.0: 
                            continue
                        if pvalue < 0.05 and pvalue >= 0.01:
                            ax.text(l-0.0,k+0.25,'*',fontsize='xx-large',horizontalalignment='center', verticalalignment='center')
                        
                        if pvalue < 0.01 and pvalue >= 0.001:
                            ax.text(l-0.0,k+0.25,'**',fontsize='xx-large',horizontalalignment='center', verticalalignment='center')
                        
                        if pvalue < 0.001:
                            ax.text(l-0.0,k+0.25,'***',fontsize='xx-large',horizontalalignment='center', verticalalignment='center')
                    

        
        
        fig.tight_layout()
        plt.show()
        
        
        
        return fig

    
    
    diff = post_result - pre_result
    diff_2 = post_result_2 - pre_result

    ## color map options 
    # import ipdb; ipdb.set_trace()
    vmin, vmax = 0, max(np.max(pre_result), np.max(post_result), 5)
    vmin_diff, vmax_diff  = min(np.min(diff), np.min(diff_2), -1.5), max(np.max(diff), np.max(diff_2), 1.5)

    ## get the largest magnitude and make the color maps balanced
    v_diff = max(abs(vmin_diff), abs(vmax_diff))
    vmin_diff, vmax_diff = -v_diff, v_diff

    
    min_values = np.zeros((2, 2)) + vmin
    max_values = np.zeros ((2, 2 )) + vmax
    min_values_diff = np.zeros((2, 2)) + vmin_diff
    max_values_diff = np.zeros((2, 2)) + vmax_diff
    
    fig1 = plot_heatmap_group(pre_result, 
                       pre_result, 
                       post_result, 
                       post_result_2, 
                       min_values=min_values, 
                       max_values= max_values, 
                       title=title,
                       side_titles=['pre', 'post'],
                       titles=titles[:4])
    
    fig2 = plot_heatmap_group(diff, 
                       diff_2, 
                       diff-diff_2, 
                       None, 
                       min_values=min_values_diff, 
                       max_values= max_values_diff, 
                       title='',
                       side_titles=['post - pre', 'diff in post'],
                       titles=titles[4:], cmap='coolwarm', pvals=pvals)

    if output:
        # save to PDF file 
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(output)
        for fig in [fig1, fig2]: ## will open an empty extra figure :(
            
            pdf.savefig( fig )

        pdf.close()




def simultaneous_recordings_count(dataset_path, stimuli_id=None):
    '''
    Generates a summary report about the makeup of the datasets interms of regions which are simultaneously recorded 
    :param dataset_path - path to preprocessed dataset 
    :param stimuli_id - optional argument to only consider trials under certain stimuli  
    '''


    df = pd.read_feather(dataset_path)
    if stimuli_id:
        df = df[df.stimn==stimuli_id]
    
    recordings = df.recording.unique()

    n_mice_simultaneous_recordings = {
        ('AL', 'V1'): 0, # number of mice with simultaneous recordings 
        ('AL', 'LM'): 0, 
        ('LM', 'V1'): 0,
    }

    for r in recordings: 
        regions = tuple(sorted(df[df.recording == r].region.unique()))
        if len(regions) == 2:
            n_mice_simultaneous_recordings[regions] += 1

    return n_mice_simultaneous_recordings
