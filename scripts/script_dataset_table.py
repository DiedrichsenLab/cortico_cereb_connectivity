"""
script for training models (Better to use run_model.py directly - 
these are just examples of how to use the functions in run_model.py)

@ Ladan Shahshahani, Joern Diedrichsen Jan 30 2023 12:57
"""
import os
import pandas as pd
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import numpy as np

outdir = '/Users/jdiedrichsen/Dropbox/Talks/2025/07_Gordon/Gordon_connectivity/figure_parts'

def get_dataset_table():
    DST= pd.DataFrame()    
    for i,dataset in enumerate(gl.datasets):
        D,info,dscl = fdata.get_dataset(gl.base_dir,dataset,sess=gl.sessions[i],type='CondAll',subj='group',atlas='MNISymC3')
        T= dscl.get_participants() 
        dst ={'dataset':[dataset],
              'n_subj':[len(T)],
              'n_cond':[len(info)]}
        DST = pd.concat([DST,pd.DataFrame(dst)],ignore_index=True)
    return DST 

if __name__ == "__main__":
    # get the dataset table
    DST = get_dataset_table()
    # save it to a csv file
    DST.to_csv(os.path.join(outdir,'dataset_table.csv'),index=False)