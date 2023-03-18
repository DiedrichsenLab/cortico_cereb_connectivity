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
# sys.path.append('../cortico-cereb_connectivity')
# sys.path.append('..')
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as cm

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                 cross_over = "sess", 
                 type = "CondAll",
                 dataset = "MDTB"):
      
      df_train_list = []
      for a in logalpha_list:
         # get model config
         config = rm.get_train_config(log_alpha = a, 
                                      cross_over = cross_over,
                                      type = type, 
                                      train_dataset = dataset)
         
         # get the list of trained connectivity models and training summary
         config, conn_list, df_tmp =rm.train_model(config)
         df_train_list.append(df_tmp)
         
         # get the average weight for each alpha
         ## this will later be used if you want to evaluate the weights on a completely different dataset
         ## loop over subject level models (items in the conn_list) and save them alongside config
         weights_list = [model.coef_[np.newaxis, ...]for model in conn_list]
         # get group average weights
         weights_arr = np.concatenate(weights_list, axis = 0)
         weights_group = np.nanmean(weights_arr, axis = 0)

         # get scaling factors
         scales_list = [model.scale_[np.newaxis, ...] for model in conn_list]
         # get group average scaling factor
         scales_arr = np.concatenate(scales_list, axis = 0)
         scales_group = np.nanmean(scales_arr, axis = 0)

         # create a model object for the group
         group_model = cm.Model()
         group_model.coef_ = weights_group
         group_model.scale_ = scales_group
         group_model.alpha = np.exp(a)

         # save the group level weights in the directory where model data is saved
         save_path = os.path.join(gl.conn_dir,config['train_dataset'],'train', config['name'])
         fname = save_path + f"/{config['method']}_{config['train_id']}_logalpha{config['logalpha']}_avg.h5"
         # save group level weights
         dd.io.save(fname, group_model, compression=None)

      df = pd.concat(df_train_list, ignore_index=True)
      # save the dataframe
      filepath = os.path.join(gl.conn_dir, config['train_dataset'], f'{config["train_dataset"]}_sub_train_{config["cross_over"]}.tsv')
      df.to_csv(filepath, index = False, sep='\t')

      return df

def eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                cross_over = "sess", 
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
   # train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], dataset="MDTB", cross_over="sess", type = "CondAll")
   # eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
   #              cross_over = "half", 
   #              type = "CondHalf",
   #              train_dataset = "MDTB", 
   #              eval_dataset = "WMFS", 
   #              train_id="all", 
   #              eval_id='ses-01'
   #              )
   # eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
   #              cross_over = "half", 
   #              type = "CondHalf",
   #              train_dataset = "MDTB", 
   #              eval_dataset = "WMFS", 
   #              train_id="all", 
   #              eval_id='ses-02'
   #              )

   # eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
   #              cross_over = "half", 
   #              type = "CondHalf",
   #              train_dataset = "MDTB", 
   #              eval_dataset = "Nishimoto", 
   #              train_id="all", 
   #              eval_id='ses-01'
   #              )

   # eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
   #              cross_over = "half", 
   #              type = "CondHalf",
   #              train_dataset = "MDTB", 
   #              eval_dataset = "Nishimoto", 
   #              train_id="all", 
   #              eval_id='ses-02'
   #              )


   # eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
   #              cross_over = "half", 
   #              type = "CondHalf",
   #              train_dataset = "MDTB", 
   #              eval_dataset = "Somatotopic", 
   #              train_id="all", 
   #              eval_id="ses-motor"
   #              )

   # eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
   #              cross_over = "half", 
   #              type = "CondHalf",
   #              train_dataset = "MDTB", 
   #              eval_dataset = "Demand", 
   #              train_id="all", 
   #              eval_id="ses-01"
   #              )

   eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                cross_over = "half", 
                type = "CondHalf",
                train_dataset = "MDTB", 
                eval_dataset = "IBC", 
                train_id="all", 
                eval_id="ses-tom"
                )

   eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                cross_over = "half", 
                type = "CondHalf",
                train_dataset = "MDTB", 
                eval_dataset = "IBC", 
                train_id="all", 
                eval_id="ses-mathlang"
                )

   eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                cross_over = "half", 
                type = "CondHalf",
                train_dataset = "MDTB", 
                eval_dataset = "IBC", 
                train_id="all", 
                eval_id="ses-rsvplanguage"
                )
