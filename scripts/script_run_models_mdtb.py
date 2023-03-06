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
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12]):
      df_train_list = []
      for a in logalpha_list:
         print(f"- Training model for loglapha {a}")
         config = rm.get_train_config(log_alpha = a)
         _, df_tmp =rm.train_model(config, group = False)
         df_train_list.append(df_tmp)
      df = pd.concat(df_train_list, ignore_index=True)
      # save the dataframe
      filepath = os.path.join(gl.conn_dir, config['dataset'], 'mdtb_sub_train_model_ses-s1.tsv')
      df.to_csv(filepath, index = False, sep='\t')

      return df

def eval_models():
   df_eval_list = []
   df_eval_voxel_list = []
   for a in [0, 2, 4, 6, 8, 10, 12]:
      config = rm.get_eval_config(log_alpha = a)
      eval_tmp, eval_voxels_tmp = rm.eval_model(config, save_tensor = False, save = True, group = False)
      df_eval_list.append(eval_tmp)
      df_eval_voxel_list.append(df_eval_voxel_list)

   df = pd.concat(df_eval_list, ignore_index=True)
   # save the dataframe
   filepath = os.path.join(gl.conn_dir, config['dataset'], 'mdtb_sub_eval_model_ses-s2.tsv')
   df.to_csv(filepath, index = False, sep='\t')
   return

# save best group weights and scale
def get_best_weights(log_alpha=8, method = "L2Regression"):
   
   config = rm.get_eval_config(log_alpha = log_alpha)
   # get dataset class 
   Data = fdata.get_dataset_class(gl.base_dir, dataset=config["dataset"])
   # get info
   info = Data.get_info(config['eval_id'],config['type'])
   T = Data.get_participants()
   subject_list = T.participant_id

   weights_list = []
   scales_list = []
   
   for sub in subject_list:
      print(f"- getting weights for {log_alpha} {sub}")
      fpath = os.path.join(gl.conn_dir, config["dataset"], 'train', config["name"])
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
   filename = os.path.join(gl.conn_dir, config["dataset"], 'train', f'{config["dataset"]}_scale.npy')
   np.save(filename,scales)
   filename = os.path.join(gl.conn_dir, config["dataset"], 'train', f'{config["name"]}_best_weights.npy')
   np.save(filename,best_weights)
   return 

if __name__ == "__main__":
   train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12])
   # eval_models()
   # for a in [0, 2, 4, 6, 8, 10, 12]:
   #    print(f"-Doing alpha = {a}")
   #    get_best_weights(log_alpha=a)