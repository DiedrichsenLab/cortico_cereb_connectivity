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

def get_group_activity():
    data = pd.DataFrame()    
    for i,dataset in enumerate(gl.datasets):
        DCereb,info,dscl = fdata.get_dataset(gl.base_dir,dataset,sess=gl.sessions[i],type='CondAll',subj='group',atlas='MNISymC3')
        DCortex,info,dscl = fdata.get_dataset(gl.base_dir,dataset,sess=gl.sessions[i],type='CondAll',subj='group',atlas='fs32k')
        info['cond'] = info.task_code + '_' + info.cond_code 

        info['cortical_act'] = np.nanmean(DCortex, axis=2).squeeze()
        info['cerebellar_act'] = np.nanmean(DCereb.mean,axis=2).squeeze()

        data = pd.concat([data,pd.DataFrame(info)],ignore_index=True)
    return data

if __name__ == "__main__":
    # get the dataset table
    DST = get_group_activity()
    # save it to a csv file
    DST.to_csv(os.path.join(outdir,'selective_recruitment.tsv'),index=False)