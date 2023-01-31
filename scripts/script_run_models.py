"""
script for getting group average weights
@ Ladan Shahshahani Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import re
import sys
from collections import defaultdict
sys.path.append('../cortico-cereb_connectivity')
sys.path.append('..')
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import prepare_data as prep
import run as connect_run

# preparing data 
def prep_tensor():
   prep.save_data_tensor(dataset = "MDTB",
                        atlas='SUIT3',
                        ses_id='ses-s1',
                        type="CondHalf")
   prep.save_data_tensor(dataset = "MDTB",
                        atlas='fs32k',
                        ses_id='ses-s1',
                        type="CondHalf")
   prep.save_data_tensor(dataset = "MDTB",
                        atlas='SUIT3',
                        ses_id='ses-s2',
                        type="CondHalf")
   prep.save_data_tensor(dataset = "MDTB",
                        atlas='fs32k',
                        ses_id='ses-s2',
                        type="CondHalf")

   return

def train_models():
      df_train_list = []
      for a in [0, 2, 4, 6, 8, 10, 12]:
         config = connect_run.get_train_config(log_alpha = a)
         _, df_tmp =connect_run.train_model(config, save_tensor = False, group = False)
         df_train_list.append(df_tmp)
      df = pd.concat(df_train_list, ignore_index=True)
      # save the dataframe
      filepath = os.path.join(prep.conn_dir, config['dataset'], 'mdtb_sub_train_model_ses-s1.tsv')
      df.to_csv(filepath, index = False, sep='\t')

      return

def eval_models():
   df_eval_list = []
   df_eval_voxel_list = []
   for a in [0, 2, 4, 6, 8, 10, 12]:
      config = connect_run.get_eval_config(log_alpha = a)
      eval_tmp, eval_voxels_tmp = connect_run.eval_model(config, save_tensor = False, save = True, group = False)
      df_eval_list.append(eval_tmp)
      df_eval_voxel_list.append(df_eval_voxel_list)

   df = pd.concat(df_eval_list, ignore_index=True)
   # save the dataframe
   filepath = os.path.join(prep.conn_dir, config['dataset'], 'mdtb_sub_eval_model_ses-s2.tsv')
   df.to_csv(filepath, index = False, sep='\t')
   return

# save best group weights and scale
def get_best_weights(log_alpha=8, method = "L2Regression"):
   
   config = connect_run.get_eval_config(log_alpha = log_alpha)
   # get dataset class 
   Data = fdata.get_dataset_class(prep.base_dir, dataset=config["dataset"])
   # get info
   info = Data.get_info(config['eval_id'],config['type'])
   T = Data.get_participants()
   subject_list = T.participant_

   weights_list = []
   scales_list = []
   
   for sub in subject_list:
      print(f"- getting weights for {log_alpha} {sub}")
      fpath = os.path.join(prep.conn_dir, config["dataset"], 'train', config["name"])
      fname = os.path.join(fpath,  f"{config['method']}_alpha{config['log_alpha']}_{sub}.h5")
      fitted_model = dd.io.load(fname)
      # load the model and get the weights for the subject
      weights_list.append(fitted_model.coef_[np.newaxis, ...])
      scales_list.append(fitted_model.scale_.reshape(-1, 1))

   # get group scale and weight
   weights_arr = np.concatenate(weights_list, axis = 0)
   scales_arr = np.concatenate(scales_list, axis = 1)
   best_weights = np.nanmean(weights_arr, axis = 0)
   scales = np.nanmean(scales_arr, axis = 1)

   # save best weights and scale
   filename = os.path.join(prep.conn_dir, config["dataset"], 'train', f'{config["dataset"]}_scale.npy')
   np.save(filename,scales)
   filename = os.path.join(prep.conn_dir, config["dataset"], 'train', f'{config["name"]}_best_weights.npy')
   np.save(filename,best_weights)
   return 

if __name__ == "__main__":
   for a in [0, 2, 4, 6, 8, 10, 12]:
      print(f"-Doing alpha = {a}")
      get_best_weights(log_alpha=a)