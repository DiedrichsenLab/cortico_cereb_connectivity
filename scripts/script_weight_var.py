"""
script for calculating the variance of connectivity weights
@ Ali Shahbazi
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.evaluation as ev
import matplotlib.pyplot as plt
import scipy.stats as stats


var_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/variance')


def estimating_weight_var(ext_list=[8],
                          train_dataset="MDTB",
                          eval_dataset=["MDTB"],
                          train_ses="ses-s1",
                          method="L2regression",
                          parcellation="Icosahedron1002",
                          cerebellum='SUIT3',
                          type=["CondHalf"],
                          crossed='half',
                          add_rest=False,
                          subj_list="all",
                          model_subj_list="all",
                          model='ind',
                          ):

   for i,ed in enumerate(eval_dataset):
      config = rm.get_eval_config(eval_dataset = ed,
                                 eval_ses = train_ses,
                                 parcellation = parcellation,
                                 crossed = crossed, # "half", # or None
                                 type = type[i],
                                 cerebellum=cerebellum,
                                 splitby = None,
                                 add_rest = add_rest,
                                 subj_list = subj_list,
                                 model_subj_list = model_subj_list,
                                 model = model)

      dirname=[]
      mname=[]

      for a in ext_list:
         dirname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
         if a is None:
            mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
         if isinstance(a,int):
            mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_A{a}")
         elif isinstance(a,str):
            mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_{a}")

      # get dataset class
      dataset = fdata.get_dataset_class(gl.base_dir,
                                       dataset=config["eval_dataset"])

      T = dataset.get_participants()
      # get list of subjects
      if config["subj_list"]=='all':
         config["subj_list"] = T.participant_id
      elif isinstance(config["subj_list"],int):
         if config["subj_list"] < len(T.participant_id):
            config["subj_list"] = T[:config["subj_list"]].participant_id
         else:
            config["subj_list"] = T.participant_id
      
      # get list of subject for average model
      if config["model_subj_list"]=='all':
         config["model_subj_list"] = T.participant_id
      elif isinstance(config["model_subj_list"],int):
         if config["model_subj_list"] < len(T.participant_id):
            config["model_subj_list"] = T[:config["model_subj_list"]].participant_id
         else:
            config["model_subj_list"] = T.participant_id

      # loop over subjects
      for i, sub in enumerate(config["subj_list"]):
         print(f'- Calculating for {sub}')

         YY, info, _ = fdata.get_dataset(gl.base_dir,
                                       config["eval_dataset"],
                                       atlas=config["cerebellum"],
                                       sess=config["eval_ses"],
                                       type=config["type"],
                                       subj=str(sub))
         XX, info, _ = fdata.get_dataset(gl.base_dir,
                                       config["eval_dataset"],
                                       atlas=config["cortex"],
                                       sess=config["eval_ses"],
                                       type=config["type"],
                                       subj=str(sub))
         # Average the cortical data over parcels
         X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
         # get the vector containing tessel labels
         X_atlas.get_parcel(config['label_img'], unite_struct = False)
         # get the mean across tessels for cortical data
         XX, labels = fdata.agg_parcels(XX, X_atlas.label_vector,fcn=np.nanmean)

         # Remove Nans
         Y = np.nan_to_num(YY[0,:,:])
         X = np.nan_to_num(XX[0,:,:])

         # Add explicit rest to sessions 
         if config["add_rest"]:
            Y,_ = add_rest(Y,info)
            X,info = add_rest(X,info)

         X /= np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])

         sigma2_eps = estimate_sigma_eps(Y=Y, dataframe=info)
         sub_weight_variance = calc_weight_var(X=X, sigma2_eps=sigma2_eps, logalpha=ext_list[0])
         np.save(os.path.join(var_folder, f'weight_variance2_{str(sub)}.npy'), sub_weight_variance)
         # np.save(os.path.join(var_folder, f'sigma2_eps3_{str(sub)}.npy'), sigma2_eps)


def estimate_sigma_eps(Y: np.array, dataframe):
   # general way
   Y_list = []
   for i in np.unique(dataframe["half"]):
      Y_list.append(Y[dataframe["half"] == i, :])

   Y_mean = np.nanmean(np.stack(Y_list, axis=0), axis=0)

   sigma2_eps = np.zeros(Y_mean.shape[1])
   for i in np.unique(dataframe["half"]):
      Y_i = Y[dataframe["half"] == i, :]
      sigma2_eps += np.diag((Y_i-Y_mean).T @ (Y_i-Y_mean)) / (Y_mean.shape[0]-1)
   
   # subtraction way
   # Y_1 = Y[dataframe["half"] == 1, :]
   # Y_2 = Y[dataframe["half"] == 2, :]
   # sigma2_eps = np.var(Y_1 - Y_2, axis=0)/2

   # reliability way
   # _, R_vox, _, _ = ev.calculate_reliability(Y, dataframe)
   # var_Y_1 = np.var(Y[dataframe["half"] == 1, :], axis=0)
   # var_Y_2 = np.var(Y[dataframe["half"] == 2, :], axis=0)
   # var_Y = np.nanmean([var_Y_1, var_Y_2], axis=0)
   # var_Y_star = var_Y * R_vox
   # var_Y_star[var_Y_star <= 0] = np.nan
   # sigma2_eps = var_Y - var_Y_star

   sigma2_eps[sigma2_eps == 0] = np.nan
   return sigma2_eps


def calc_weight_var(X: np.array,                # 2N x Q matrix
                    sigma2_eps: np.array,       # 1 x P vector 
                    logalpha):
   
   Q: int = X.shape[1]
   P: int = len(sigma2_eps)
   X_transpose = X.T
   pseudoinverse = np.linalg.inv(X_transpose @ X + np.exp(logalpha) * np.identity(Q)) @ X_transpose

   # sub_weight_variance = sigma2_eps * np.nansum(pseudoinverse**2)
   sub_weight_variance = sigma2_eps * np.trace(pseudoinverse.T @ X_transpose @ X @ pseudoinverse)
   print(f'trace(A@A.T): {np.trace(pseudoinverse.T @ X_transpose @ X @ pseudoinverse)}')

   # sub_weight_variance = np.zeros((1, P))
   # for v in range(P):
      # var_y = np.block([[sigma2_eps[v]*I_N, rho[v]*sigma2_eps[v]*I_N], [rho[v]*sigma2_eps[v]*I_N, sigma2_eps[v]*I_N]])
      # sub_weight_variance[:, v] = np.einsum('ij,jk,ik->i', pseudoinverse, var_y, pseudoinverse) # same as `np.diag(pseudoinverse @ var_y @ pseudoinverse.T)` but more efficient
      # print(f'A A.T: {np.sum(pseudoinverse * pseudoinverse, axis=1)}')
      # sys.exit("Stopping the script...")
   return sub_weight_variance


if __name__ == "__main__":
   estimating_weight_var(ext_list=[8],
                          train_dataset="MDTB",
                          eval_dataset=["MDTB"],
                          train_ses="ses-s1",
                          method="L2regression",
                          parcellation="Icosahedron1002",
                          cerebellum='SUIT3',
                          type=["CondHalf"],
                          crossed='half',
                          add_rest=False,
                          subj_list="all",
                          model_subj_list="all",
                          model='ind',
                          )
   
