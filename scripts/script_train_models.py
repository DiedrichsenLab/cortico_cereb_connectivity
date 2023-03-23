"""
script for training models 
@ Ladan Shahshahani, Joern Diedrichsen Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import re
import sys
from collections import defaultdict
# sys.path.append('../cortico-cereb_connectivity')
# sys.path.append('..')
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as cm

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                 crossed = "half", 
                 type = "CondHalf",
                 train_ses = 'ses-s1',
                 dataset = "MDTB"):
      
    config = rm.get_train_config(log_alpha = logalpha_list, 
                                crossed = crossed,
                                type = type, 
                                train_dataset = dataset,
                                train_ses=train_ses)
         
         # get the list of trained connectivity models and training summary
    config, conn_list, df_tmp =rm.train_model(config)
    
def avrg_model(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
            train_data = "MDTB",
            train_ses= "ses-s1",
            parcellation = 'Icosahedron1002',
            method='L2Regression'):
    mname = f"{train_data}_{train_ses}_{parcellation}_{method}"

    for la in logalpha_list: 
        mname_ext = f"A{la}"
        rm.calc_avrg_model(train_data,mname,mname_ext)


def eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                type = "CondHalf",
                train_dataset = "MDTB", 
                eval_dataset = "Demand", 
                train_id = "all", 
                eval_id = "half"
                ):
   df_eval_list = []
   df_eval_voxel_list = []
   for a in logalpha_list:
      config = rm.get_eval_config(log_alpha = a, 
                                  train_dataset=train_dataset, 
                                  eval_dataset=eval_dataset, 
                                  cross_over=cross_over, 
                                  type = type, 
                                  train_id=train_id, 
                                  eval_id=eval_id)
      eval_tmp, eval_voxels_tmp = rm.eval_model(config, save = True, avg = True)
      df_eval_list.append(eval_tmp)
      df_eval_voxel_list.append(df_eval_voxel_list)

   df = pd.concat(df_eval_list, ignore_index=True)
   # save the dataframe
   save_path = gl.conn_dir+ f"/{config['train_dataset']}/eval"

   if not os.path.isdir(save_path):
      os.mkdir(save_path)
   else:
      pass

   filepath = save_path + f"/{config['eval_dataset']}_sub_eval_model_{config['eval_id']}.tsv"
   df.to_csv(filepath, index = False, sep='\t')
   return

if __name__ == "__main__":
    train_models()
    # avrg_model()