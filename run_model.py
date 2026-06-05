"""Main module for training and evaluating connectivity models.
   Designed to work together with Functional_Fusion package.
   Dataset, session, and parcellation names are as in Functional_Fusion.
   The main work is being done by train_model and eval_model functions.
   @authors: Ladan Shahshahani, Maedbh King, Ali Shahbazi, Jörn Diedrichsen
"""

# from audioop import cross
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nb
from collections import defaultdict
from sklearn.model_selection import cross_val_score
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.reliability as frel # from functional fusion module

import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.evaluation as ev
from copy import copy, deepcopy
import warnings
import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore")

def get_train_config(train_dataset = "MDTB",
                     train_ses = "all",
                     run = 'all',
                     cond_num = 'all',
                     task_code = 'all',
                     subj_list = 'all',
                     method = "L2reg",
                     log_alpha = 8,
                     cerebellum = "MNICymC3",
                     cortex = "fs32k",
                     parcellation = "Icosahedron1002",
                     type = "CondHalf",
                     crossed = "half", # or None
                     add_rest = True,
                     append = False,
                     cortical_cerebellar_act = 'ind',
                     std_cortex = 'parcel',
                     std_cerebellum = 'global'
                     ):
   """get_train_config
   Function to create a config dictionary containing the info for the training

   Args:
      train_dataset (str): training_dataset. Defaults to "MDTB".
      train_ses (str): Training session. Defaults to "all".
      run (list, str): Training run (e.g. [1,2]). Defaults to "all".
      cond_num (list, str): Training conditions (e.g. [1,2,3,4,5]). Defaults to "all".
      task_code (list, str): Training task codes (e.g. [1,2,3]). Defaults to "all".
      subj_list (list, str): Training subject list. Defaults to "all".
      method (str): Model class. Defaults to "L2reg".
      log_alpha (int): log of regularization. Defaults to 8.
      cerebellum (str): Atlas for cerebellum. Defaults to "MNICymC3".
      cortex (str): Atlas for neocortex. Defaults to "fs32k".
      parcellation (str): Parcellation for cortex. Defaults to "Icosahedron-1002_Sym.32k".
      type (str): Training type, could be "CondHalf", "CondAll", "CondRun". see Functional_Fusion. Defaults to "CondHalf".
      crossed (str): Double crossvalidation cortex-cerebellum. ("half" (default) or None)
      add_rest (bool): Add rest condition to each session and half. Defaults to True.
      append (bool): Append the current training info to the previous one. Defaults to False.
      cortical_cerebellar_act (str): 'ind': individual X and Y, 'avg': average X and Y accross subjects. Defaults to 'ind'.
      std_cortex (str): z-Standardize the cortical data. (Defaults to parcel normalization)
      std_cerebellum (str): z-Standardize the cerebellar data. (Defaults to global normalization)

   Returns:
      dict: Dictionary containing the default training configuration
   """
   train_config = {}
   train_config['train_dataset'] = train_dataset # name of the dataset to be used in
   train_config['train_ses'] = train_ses
   train_config['run'] = run
   train_config['cond_num'] = cond_num
   train_config['task_code'] = task_code
   train_config['subj_list'] = subj_list
   train_config['method'] = method   # method used in modelling (see model.py)
   train_config['logalpha'] = log_alpha # alpha will be np.exp(log_alpha)
   train_config['cerebellum'] = cerebellum
   train_config['cortex'] = cortex
   train_config['parcellation'] = parcellation
   train_config['crossed'] = crossed
   train_config["type"] = type
   train_config['add_rest'] = add_rest
   train_config['cortical_cerebellar_act'] = cortical_cerebellar_act
   train_config['std_cortex'] = std_cortex
   train_config['std_cerebellum'] = std_cerebellum
   train_config['append'] = append

   # get label images for left and right hemisphere
   train_config['label_img'] = []
   for hemi in ['L', 'R']:
      train_config['label_img'].append(gl.atlas_dir + f'/tpl-{train_config["cortex"]}' + f'/{train_config["parcellation"]}.{hemi}.label.gii')

   return train_config


def get_model_config(dataset = "MDTB",
                     subj_list = 'all',
                     model = 'avg',
                     cerebellum = "MNISymC3",
                     mix_param = None):
   """
   create a config dictionary containing the info for the model
   
   Args:
      dataset (str): training_dataset. Defaults to "MDTB".
      subj_list (str or list): List of subjects to train on. Defaults to 'all'.
      model (str or list): Model type to use. Defaults to 'avg'.
      cerebellum (str): Atlas for cerebellum. Defaults to "MNISymC3".
      mix_param (float): Mixing parameter for 'mix' model. Defaults to None.

   Returns:
      dict: Dictionary containing the default model configuration
   """
   model_config = {}
   model_config['dataset'] = dataset
   model_config['subj_list'] = subj_list
   model_config['model'] = model
   model_config['cerebellum'] = cerebellum
   model_config['mix_param'] = mix_param

   return model_config


def get_eval_config(eval_dataset = 'MDTB',
            eval_ses = 'all',
            subj_list = 'all',
            run = 'all',
            cond_num = 'all',
            task_code = 'all',
            cerebellum = 'MNICymC3',
            cortex = "fs32k",
            parcellation = "Icosahedron1002",
            crossed = "half", # or None
            type = "CondHalf",
            splitby = None,
            add_rest = True,
            std_cortex = 'parcel',
            std_cerebellum = 'global',
            cortical_act = 'avg'):
   """
   create a config dictionary for evaluation of the model

   Args:
      eval_dataset (str): evaluation dataset. Defaults to 'MDTB'.
      eval_ses (str): evaluation session. Defaults to 'all'.
      subj_list (str or list): List of subjects to evaluate. Defaults to 'all'.
      run (str or list): List of runs to evaluate. Defaults to 'all'.
      cond_num (str or list): List of conditions to evaluate. Defaults to 'all'.
      task_code (str or list): List of task codes to evaluate. Defaults to 'all'.
      cerebellum (str): Atlas for cerebellum. Defaults to 'MNICymC3'.
      cortex (str): Atlas for neocortex. Defaults to "fs32k".
      parcellation (str): Parcellation for cortex. Defaults to "Icosahedron1002".
      crossed (str): Double crossvalidation cortex-cerebellum. ("half" (default) or None)
      type (str): Type of evaluation. Defaults to "CondHalf".
      splitby (str): Split evaluation by 'sess', 'run', or None. Defaults to None.
      add_rest (bool): Add rest condition to each session and half. Defaults to True.
      std_cortex (str): Standardization method for cortex. Defaults to 'parcel'.
      std_cerebellum (str): Standardization method for cerebellum. Defaults to 'global'.
      cortical_act (str): Type of cortical activity to use. ['ind', 'avg', 'loo'].

   Returns:
      dict: Dictionary containing the evaluation configuration
   """
   eval_config = {}
   eval_config['eval_dataset'] = eval_dataset
   eval_config['eval_ses'] = eval_ses
   eval_config['run'] = run
   eval_config['cond_num'] = cond_num
   eval_config['task_code'] = task_code
   eval_config['cerebellum'] = cerebellum
   eval_config['cortex'] = cortex
   eval_config['parcellation'] = parcellation
   eval_config['crossed'] = crossed
   eval_config['add_rest'] = add_rest
   eval_config['std_cortex'] = std_cortex
   eval_config['std_cerebellum'] = std_cerebellum
   eval_config["splitby"] = splitby
   eval_config["type"] = type
   eval_config['subj_list'] = subj_list
   eval_config['cortical_act'] = cortical_act
   
   # get label images for left and right hemisphere
   eval_config['label_img'] = []
   for hemi in ['L', 'R']:
      eval_config['label_img'].append(gl.atlas_dir + f'/tpl-{eval_config["cortex"]}' + f'/{eval_config["parcellation"]}.{hemi}.label.gii')

   return eval_config


def train_metrics(model, X, Y):
   """computes training metrics (rmse and R) on X and Y

   Args:
      model (class instance): must be fitted model
      X (nd-array): Input features
      Y (nd-array): Target variables

   Returns:
      rmse_train (scalar), R_train (scalar)
   """
   Y_pred = model.predict(X)

   # get train rmse and R
   R_train, _ = ev.calculate_R(Y, Y_pred)
   R2_train,_ = ev.calculate_R2(Y, Y_pred)

   return R_train, R2_train


def eval_metrics(Y, Y_pred, info):
   """Compute evaluation, returning summary and voxel data.

   Args:
      Y (np array): The observed data
      Y_pred (np array): The predicted data
      Y_info (pd dataframe): The information dataframe for Y

   Returns:
      dict containing evaluations (R, R2, noise).
   """
   # initialise dictionary
   data = {}

   # R between predicted and observed
   data["R_eval"], data["R_vox"] = ev.calculate_R(Y=Y, Y_pred=Y_pred)

   # R2 between predicted and observed
   data["R2_eval"], data["R2_vox"] = ev.calculate_R2(Y=Y, Y_pred=Y_pred)

   # # Noise ceiling for observed cerebellum
   # (
   #    data["noise_Y_R"],
   #    data["noise_Y_R_vox"],
   #    data["noise_Y_R2"],
   #    data["noise_Y_R2_vox"],
   # ) = ev.calculate_reliability(Y=Y, dataframe = info)

   # # Noise ceiling for predicted cerebellum (squared)
   # (
   #    data["noise_X_R"],
   #    data["noise_X_R_vox"],
   #    data["noise_X_R2"],
   #    data["noise_X_R2_vox"],
   # ) = ev.calculate_reliability(Y=Y_pred, dataframe = info)

   # calculate noise ceiling
   # with warnings.catch_warnings():
   #    warnings.simplefilter("ignore", category=RuntimeWarning)

   #    data["noiseceiling_Y_R_vox"] = np.sqrt(data["noise_Y_R_vox"])
   #    data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"]) * np.sqrt(data["noise_X_R_vox"])
   return data


def cross_data(Y,info,mode):
   """Cross data across halves. This part helps reducing overfitting by providing an extra cross-validation.

   Args:
      Y (ndarray): Data matrix (n_cond,n_vox)
      info (pd.DataFrame): Information dataframe with columns: sess, half, run; n_cond rows
      mode (str): 'half' or 'run' to specify the cross-validation mode

   Returns:
      Ys (ndarray): Crossed data
   """
   if mode=='half':
      Y_list = []
      for s in np.unique(info.sess):
         Y_list.append(Y[(info.sess==s) & (info.half==2),:])
         Y_list.append(Y[(info.sess==s) & (info.half==1),:])
      Ys = np.concatenate(Y_list,axis=0)
   elif mode=='run':
      unique_runs = sorted(info.run.unique())
      first_runs = unique_runs[:len(unique_runs)//2]
      second_runs = unique_runs[len(unique_runs)//2:]
      Y_list = []
      for s in np.unique(info.sess):
         Y_list.append(Y[(info.sess==s) & (info.run.isin(second_runs)),:])
         Y_list.append(Y[(info.sess==s) & (info.run.isin(first_runs)),:])
      Ys = np.concatenate(Y_list,axis=0)
   return Ys


def subset_cond(data, info, cond_num):
   """
   Subset the data and info based on the condition number.

   Args:
       data (ndarray): Data matrix (n_cond,n_vox) or (n_subj,n_cond,n_vox)
       info (pd.DataFrame): Information dataframe with columns: sess, half, run; n_cond rows
       cond_num (str or list): Condition number(s) to subset

   Returns:
       data (ndarray): Subsetted data
       info (pd.DataFrame): Subsetted information
   """
   if isinstance(cond_num, list):
      cond_mask = info['cond_num'].isin(cond_num)
      data = data[..., cond_mask.values, :]
      info = info[cond_mask]
   else:
      codes = np.unique(info.cond_num)
      if cond_num == 'train':
         codes_mask = info.cond_num.isin(codes[:len(codes)//3])
         data = data[..., codes_mask, :]
         info = info[codes_mask]
      elif cond_num == 'eval':
         codes_mask = info.cond_num.isin(codes[len(codes)//3:])
         data = data[..., codes_mask, :]
         info = info[codes_mask]
      elif cond_num == 'rnd_train':
         rng = np.random.default_rng(seed=42)
         shuffled = rng.permutation(codes)
         codes_mask = info.cond_num.isin(shuffled[:len(shuffled)//3])
         data = data[..., codes_mask, :]
         info = info[codes_mask]
      elif cond_num == 'rnd_eval':
         rng = np.random.default_rng(seed=42)
         shuffled = rng.permutation(codes)
         codes_mask = info.cond_num.isin(shuffled[len(shuffled)//3:])
         data = data[..., codes_mask, :]
         info = info[codes_mask]

   return data, info


def add_rest(Y,info):
   """Add rest to each session and half
   Subtract the mean across all conditions

   Args:
       Y (ndarray): Data matrix (n_cond,n_vox) or (n_subj,n_cond,n_vox)
       info (pd.DataFrame): Information dataframe with columns: sess, half, task_code; n_cond rows

   Returns:
       Y (ndarray): Data with rest condition added, mean per session and half removed
       info (pd.DataFrame): Information dataframe with rest condition added
   """
   Y_list = []
   info_list = []
   for s in np.unique(info.sess):
      for h in np.unique(info.half):
         indx = (info.sess==s) & (info.half==h)
         if any([i.startswith('rest') for i in info[indx].task_code]):
            Y_list.append(Y[...,indx,:]-Y[...,indx,:].mean(axis=-2,keepdims=True))
            info_list.append(info[indx])
         else:
            Yshape = np.array(Y.shape)
            Yshape[-2]=indx.sum()+1
            Yp = np.zeros(Yshape)
            Yp[...,0:-1,:] = Y[...,indx,:]
            Yp = Yp - Yp.mean(axis=-2,keepdims=True) # subtract mean across all conditions
            Y_list.append(Yp)
            inf = info[indx]
            newD = {'task_code':['rest'],
                    'sess':[inf.sess.iloc[0]],
                    'half':[inf.half.iloc[0]]}
            inf = pd.concat([inf,pd.DataFrame(newD)],ignore_index=True)
            info_list.append(inf)
   Ys = np.concatenate(Y_list,axis=-2)
   infos = pd.concat(info_list,ignore_index=True)
   return Ys, infos


def std_data(Y,mode):
   """ Standarize the data to unit norm.

   Args:
       Y (ndarray): Data matrix (n_cond,n_vox) or (n_subj,n_cond,n_vox)
       mode (str): 'parcel' or 'global' to specify the standardization mode

   Returns:
       Y (ndarray): Standardized data
   """
   if mode is None:
      return Y
   elif mode=='parcel':
      sc=np.sqrt(np.nansum(Y ** 2, 0))# / Y.shape[0])
      return  np.nan_to_num(Y/sc)
   elif mode=='global':
      sc=np.sqrt(np.nansum(Y ** 2))# / Y.size)
      return np.nan_to_num(Y/sc)
   else:
      raise ValueError('std_mode must be None, "voxel" or "global"')
   

def prepare_data(data, info, config):
   """ Prepare the data and info for modeling. Including removing NaNs, adding rest conditions, and subsetting.

   Args:
       data (ndarray): Input features
       info (pd.DataFrame): Information dataframe
       config (dict): Configuration dictionary (train_config or eval_config)

   Returns:
       data (ndarray): Prepared input features
       info (pd.DataFrame): Prepared information dataframe
   """
   # Remove Nans
   data = np.nan_to_num(data)

   # Add rest condition?
   if config["add_rest"]:
      data,info = add_rest(data,info)

   # Indlude only some runs?
   if config["run"]!='all':
      if isinstance(config["run"], list):
         run_mask = info['run'].isin(config["run"])
         data = data[..., run_mask.values, :]
         info = info[run_mask]

   # Include only some tasks?
   if config["task_code"]!='all':
      if isinstance(config["task_code"], list):
         run_mask = info['task_code'].isin(config["task_code"])
         data = data[..., run_mask.values, :]
         info = info[run_mask]
   
   # Include only some conds?
   if config['cond_num']!='all':
      data,info = subset_cond(data, info, config['cond_num'])

   # Definitely subtract intercept across all conditions
   data = (data - data.mean(axis=-2,keepdims=True))

   return data, info


def exclude_network(XX, config):
   """ Exclude specific networks from the cortical data based on the Yeo parcellation.

   Args:
       XX (ndarray): Cortical data.
       config (dict): Configuration dictionary.

   Returns:
       XX (ndarray): Cortical data with excluded networks set to zero.
   """
   yeo_img = nb.load(gl.conn_dir + f"/maps/yeo17_{config['parcellation']}.plabel.nii")
   yeo_data = yeo_img.get_fdata().squeeze()
   XX[..., :, yeo_data==config['exclude_network']] = 0.0

   return XX


def get_cortical_data(dataset, sessions, subj, config):
   """ Get cortical data according to the training or evaluation config file. Uses Functional_Fusion.dataset.

   Args:
      dataset (str): Name of the dataset to load.
      sessions (str or list): Session(s) to load data from.
      subj (str or list): Subject ID to load data for.
      config (dict): Configuration dictionary (train_config or eval_config).

   Returns:
      XX (ndarray): Cortical data.
      info (pd.DataFrame): Information dataframe.
   """
   XX, info, _ = fdata.get_dataset(gl.base_dir,
                                   dataset,
                                   sess=sessions,
                                   subj=subj,
                                   atlas=config["cortex"],
                                   type=config["type"])
   # Average the cortical data over pacels
   X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(config['label_img'], unite_struct = False)
   # get the mean across tessels for cortical data
   XX, labels = fdata.agg_parcels(XX, X_atlas.label_vector,fcn=np.nanmean)

   # Prepare the data and info
   XX, info = prepare_data(XX, info, config)

   # Standardize the data if specified
   for i in range(XX.shape[0]):
      if 'std_cortex' in config.keys():
         XX[i,:,:] = std_data(XX[i,:,:],config['std_cortex'])

   # Exclude specific networks if specified
   if 'exclude_network' in config.keys():
      XX = exclude_network(XX, config)

   return XX, info


def get_cerebellar_data(dataset, sessions, subj, config):
   """Get cerebellar data for training or evaluation.

   Args:
      dataset (str): Name of the dataset to load.
      sessions (str or list): Session(s) to load data from.
      config (dict): Configuration dictionary containing:
         add_res, run, cond_num, cerebellum, std_cerebellum, type, crossed, 

   Returns:
      YY (ndarray): Cerebellar data.
      info (pd.DataFrame): Information dataframe.
   """
   # Load the cerebellar data
   YY, info, _ = fdata.get_dataset(gl.base_dir,
                                   dataset,
                                   sess=sessions,
                                   subj=subj,
                                   atlas=config["cerebellum"],
                                   type=config["type"])

   # Prepare the data and info
   YY, info = prepare_data(YY, info, config)

   # Standardize the data if specified
   for i in range(YY.shape[0]):
      if 'std_cerebellum' in config.keys():
         YY[i,:,:] = std_data(YY[i,:,:],config['std_cerebellum'])

      # cross the halves within each session
      if config["crossed"] is not None:
         YY[i,:,:] = cross_data(YY[i,:,:],info,config["crossed"])

   return YY, info 


def save_XY_data(save_name, XX, YY, config, info, dataset=None):
   """
   Save the preprocessed cortical and cerebellar data as CIFTI files.

   Args:
      save_name (str): Name to save the CIFTI files.
      XX (ndarray): Preprocessed cortical data.
      YY (ndarray): Preprocessed cerebellar data.
      config (dict): Configuration dictionary.
      info (pd.DataFrame): Information dataframe.
      dataset (str): Name of the dataset.
   """

   if dataset is None:
      dataset = config['train_dataset']

   Yatlas,_ = at.get_atlas(config['cerebellum'])
   row_axis = dataset + '_' + info.names 
   Ycifti = Yatlas.data_to_cifti(YY, row_axis=row_axis)
   nb.save(Ycifti,f'{gl.conn_dir}/maps/{save_name}_cerebellum.dscalar.nii')

   Xatlas,_ = at.get_atlas(config['cortex'])
   Xatlas.get_parcel(config['label_img'], unite_struct = False)      
   Xparcelaxis  = Xatlas.get_parcel_axis()
   Xrowaxis = nb.cifti2.ScalarAxis(row_axis)
   header = nb.Cifti2Header.from_axes((Xrowaxis, Xparcelaxis))
   Xcifti = nb.Cifti2Image(XX, header=header)
   nb.save(Xcifti,f'{gl.conn_dir}/maps/{save_name}_cortex.pscalar.nii')

   return


def train_model(config, save_path=None, mname=None, save_name=None):
   """training a specific model based on the config file created
   model will be trained on cerebellar voxels and average within cortical tessels.

   Args:
      config (dict): dictionary with configuration parameters
      save_path (str): path to save the trained model
      mname (str): name of the model
      save_name (str): name to save the model as a dscalar or pscalar

   Returns:
      conn_model_list (list): list of trained models on the list of subjects / log-alphas
      config (dict): dictionary containing info for training. Can be saved as json
      train_df (pd.DataFrame): dataframe containing training information
   """

   # get list of subjects
   subj = get_subj_list(config['subj_list'], config["train_dataset"])

   # initialize training dict
   conn_model_list = []

   # Generate model name and create directory
   if mname is None:
      mname = f"{config['train_dataset']}_{config['train_ses']}_{config['parcellation']}_{config['method']}"
   if save_path is None:
      save_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',mname)
   # check if the directory exists
   try:
      os.makedirs(save_path)
   except OSError:
      pass

   # Check if training file already exists:
   train_info_name = save_path + "/" + mname + ".tsv"
   if os.path.isfile(train_info_name) and config["append"]:
      train_info = pd.read_csv(train_info_name, sep="\t")
   else:
      train_info = pd.DataFrame()

   # Get cerebellar and cortical data
   YY, info = get_cerebellar_data(config["train_dataset"], config["train_ses"], subj, config)
   XX, info = get_cortical_data(config["train_dataset"], config["train_ses"], subj, config)

   # average cortical and cerebellar data across subjects, if needed
   if config['cortical_cerebellar_act'] == 'avg':
      if config['subj_list'] != 'all':
         # Get cerebellar and cortical data
         all_subj = get_subj_list('all', config["train_dataset"])
         YY, info = get_cerebellar_data(config["train_dataset"], config["train_ses"], all_subj, config)
         XX, info = get_cortical_data(config["train_dataset"], config["train_ses"], all_subj, config)
      XX = XX.mean(axis=0,keepdims=True) # get average cortical data
      YY = YY.mean(axis=0,keepdims=True) # get the average cerebellar data
      subj = ['group']

   elif config['cortical_cerebellar_act'] == 'loo':
      if config['subj_list'] != 'all':
         # Get cerebellar and cortical data
         all_subj = get_subj_list('all', config["train_dataset"])
         YY, info = get_cerebellar_data(config["train_dataset"], config["train_ses"], all_subj, config)
         XX, info = get_cortical_data(config["train_dataset"], config["train_ses"], all_subj, config)
      XX = (XX.sum(axis=0,keepdims=True) - XX)/(XX.shape[0]-1)
      YY = (YY.sum(axis=0,keepdims=True) - YY)/(YY.shape[0]-1)
      subj = [s+'_group_loo' for s in subj]

   if save_name is not None:
      save_XY_data(save_name, XX[0,:,:], YY[0,:,:], config, info)

   for i,sub in enumerate(subj):
      X = XX[i,:,:] # get the data for the subject
      Y = YY[i,:,:] # get the data for the subject

      for la in config["logalpha"]:
         print(f'- Train {sub}, {config["method"]}, logalpha {la}')

         if la is not None:
            # Generate new model
            alpha = np.exp(la) # get alpha
            conn_model = getattr(model, config["method"])(alpha)
            mname_spec = f"{mname}_A{la}_{sub}"
         else:
            conn_model = getattr(model, config["method"])()
            mname_spec = f"{mname}_{sub}"

         # Fit model, get train and validate metrics
         if config["method"] == 'L2reg':
            conn_model.fit(X, Y, info)
         elif config["method"] == 'L2reghalf':
            conn_model.fit(X, Y, config, info)
         else:
            conn_model.fit(X, Y)
         R_train, R2_train = train_metrics(conn_model, X, Y)
         # conn_model_list.append(conn_model) # commented to prevent memory issues

         # collect train metrics ( R)
         model_info = {"subj_id": sub,
                       "mname": mname_spec,
                       "R_train": R_train,
                       "R2_train": R2_train,
                       "num_regions": X.shape[1],
                       "logalpha": la
                       }

         # Copy over all scalars or strings from config to eval dict:
         for key, value in config.items():
            if not isinstance(value, (list, dict, pd.Series, np.ndarray)):
               model_info.update({key: value})

         # Save the individuals model and info files
         cio.save_model(conn_model, model_info, save_path + "/" + mname_spec)
         train_info = pd.concat([train_info, pd.DataFrame(model_info)], ignore_index=True)

   # Save training information
   train_info.to_csv(train_info_name, sep='\t')
   return config, conn_model_list, train_info


def train_global_model(config, save_path=None, mname=None, save_data_name=None):
   """
   train a model based on the concatination of multiple datasets from functional fusion.
   Data is group-averaged across subjects. 

   Args:
      config (dict): dictionary with configuration parameters.
      save_path (str): path to save the trained model and info files.
      mname (str): name of the model.
      save_data_name (str): name of the data files to save.

   Returns:
      conn_model_list (list): list of trained models on the list of subjects / log-alphas
      config (dict): dictionary containing info for training. Can be saved as json
      train_df (pd.DataFrame): dataframe containing training information
   """

   # get list of datasets - interpret them over globals.dscode 
   num_ds = int(len(config['train_dataset'])/2)
   datasets = []   
   sessions = []
   add_rest = [] 
   std_cortex = [] 
   for i in range(num_ds):
      code = config['train_dataset'][i*2:i*2+2]
      if code in gl.dscode:
         indx = gl.dscode.index(code)
         datasets.append(gl.datasets[indx])
         sessions.append(gl.sessions[indx])
         add_rest.append(gl.add_rest[indx])
         std_cortex.append(gl.std_cortex[indx])
      else:
         raise ValueError(f"Dataset code {code} not found in globals.dscode")   
      
   # Compile lists of activity patterns 
   XX = []
   YY = []
   info_list = [] 
   for i in range(num_ds):
      print(f'Loading data for {datasets[i]}')
      subj = get_subj_list('all', datasets[i])
      # Get cerebellar and cortical data
      config['add_rest'] = add_rest[i]
      config['std_cortex'] = std_cortex[i]
      Y, info = get_cerebellar_data(datasets[i], sessions[i], subj, config)
      X, _ = get_cortical_data(datasets[i], sessions[i], subj, config)
      info['dataset'] = datasets[i]
      XX.append(X.mean(axis=0))
      YY.append(Y.mean(axis=0))
      info_list.append(info)
   
   XX = np.concatenate(XX, axis=0)
   YY = np.concatenate(YY, axis=0)
   info = pd.concat(info_list, ignore_index=True)

   if save_data_name is not None:
      save_XY_data(save_data_name, XX, YY, config, info, dataset=info.dataset)

   conn_model_list = []

   # Generate model name and create directory
   if mname is None:
      mname = f"{config['train_dataset']}_{config['parcellation']}_{config['method']}"
   if save_path is None:
      save_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',mname)
   # check if the directory exists
   try:
      os.makedirs(save_path)
   except OSError:
      pass

   # Check if training file already exists:
   train_info_name = save_path + "/" + mname + ".tsv"
   if os.path.isfile(train_info_name) and config["append"]:
      train_info = pd.read_csv(train_info_name, sep="\t")
   else:
      train_info = pd.DataFrame()

   # Train models for each logalpha
   for la in config["logalpha"]:
      print(f'- Train, {config["method"]}, logalpha {la}')

      if la is not None:
         # Generate new model
         alpha = np.exp(la) # get alpha
         conn_model = getattr(model, config["method"])(alpha)
         mname_spec = f"{mname}_A{la}_global"
      else:
         conn_model = getattr(model, config["method"])(0)
         mname_spec = f"{mname}_global"

      # Fit model, get train and validate metrics
      if config["method"] == 'L2reg':
         conn_model.fit(XX, YY, info)
      elif config["method"] == 'L2reghalf':
         conn_model.fit(XX, YY, config, info)
      else:
         conn_model.fit(XX, YY)
      R_train, R2_train = train_metrics(conn_model, XX, YY)
      # conn_model_list.append(conn_model)

      # collect train metrics ( R)
      model_info = {"subj_id": 'group',
                    "mname": mname_spec,
                    "R_train": R_train,
                    "R2_train": R2_train,
                    "num_regions": XX.shape[1],
                    "logalpha": la
                    }

      # Copy over all scalars or strings from config to eval dict:
      for key, value in config.items():
         if not isinstance(value, (list, dict,pd.Series,np.ndarray)):
            model_info.update({key: value})

      # Save the individuals info files
      cio.save_model(conn_model,model_info,save_path + "/" + mname_spec)
      train_info = pd.concat([train_info,pd.DataFrame(model_info)],ignore_index= True)

   # Save training information
   train_info.to_csv(train_info_name,sep='\t')
   return config, conn_model_list, train_info


def get_model_names(train_dataset, train_ses, parcellation, method, ext_list):
   """ Makes a list of model dirs and model names, based on training set, etc.

   Args:
         train_dataset (str): training dataset
         train_ses (str): Session of the training dataset
         parcellation (str): Cortical parcellation
         method (str): 'L2regression', 'WTA', 'L1regression', 'NNLS', etc
         ext_list (list): List of extensions (numeric or string) to add to model name
   Returns:
         dirname (list): List of model directories 
         mname (list): List of model names 
   """   
   dirname = [] # Model directory name
   mname = [] # Model name - without the individual, average, or loo extension

   if train_ses is None:
      root_name = f"{train_dataset}"
   else: 
      root_name = f"{train_dataset}_{train_ses}"

   # Build list of to-be-evaluated models
   for a in ext_list:
      dirname.append(f"{root_name}_{parcellation}_{method}")
      if a is None:
         mname.append(f"{root_name}_{parcellation}_{method}")
      if isinstance(a,int):
         mname.append(f"{root_name}_{parcellation}_{method}_A{a}")
      elif isinstance(a,str):
         mname.append(f"{root_name}_{parcellation}_{method}_{a}")
   return dirname, mname


def get_subj_list(subj_list, dataset):
   """Get the list of subjects to evaluate or train on.

   Args:
      subj_list (str, int, list): 'all', integer number of subjects, or list of subject ids
      dataset (str): Name of the dataset to get the subject list from

   Returns:
      subj_list (list): List of subject ids to use for evaluation or training
   """
   # get dataset class
   T = fdata.get_dataset_class(gl.base_dir, dataset=dataset).get_participants()

   # get list of subjects
   if subj_list is None:
      subj_list = T.participant_id
   elif isinstance(subj_list,int):
      if subj_list < len(T.participant_id):
         subj_list = T[:subj_list].participant_id
      else:
         subj_list = T.participant_id
   elif isinstance(subj_list,(list,pd.Series,np.ndarray)):
      if isinstance(subj_list[0],str):
         pass
      else: # Numerical 
         subj_list = T.participant_id.iloc[subj_list]
   elif isinstance(subj_list, str):
      if subj_list == 'all':
         subj_list = T.participant_id
      else:
         subj_list = [subj_list]
   else:
      raise ValueError('config["subj_list"] must be a list of str, integers or "all"')
   
   if not isinstance(subj_list, list):
      subj_list = list(subj_list)

   return subj_list


def get_fitted_models(model_dirs, model_names, config):
   """Builds a list of fitted models from the saved files
   In case of individual-specific models (ind or loo), it builds a list of lists.

   if model_config['model']=='avg' it will average the models across subjects
   if model_config['model']=='ind' it will evaluate each subejct individually
   if model_config['model']=='loo' it will average all other subjects
   if model_config['model']=='mix_loo' it will do: p*subject + (1-p)*loo
   if model_config['model']=='mix' it will do: p*subject + (1-p)*given_model
   For 'ind', 'loo', and 'mix_loo' training and evaluation dataset must be the same 

   Args:
      model_dirs (list): List of dirctory names for models 
      model_names (list): List of model names (without subject extension)
      model_config (dict): Dictonary with model parameters

   Returns:
      fitted_models (list): _description_
      train_info (list): information on each trained model
   """

   # Load all the models to evaluate:
   fitted_model = []
   train_info = []

   if isinstance(config['model'],list):
      if isinstance(config['model'][0],str):
         for ind in config['model']:
            for d,m in zip(model_dirs,model_names):
               model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
               fname = model_path + f"/{m}_{ind}"
               mo,inf = cio.load_model(fname)
               fitted_model.append(mo)
               train_info.append(inf)
      elif isinstance(config['model'][0],model.Model):
         fitted_model = config['model']
         train_info = config['train_info']
      elif isinstance(config['model'][0][0],model.Model):
         fitted_model = config['model']
         train_info = config['train_info']
      else:
         raise ValueError('config["model"] must be a list of strings or a list of models')
      
   elif config['model']=='avg' or config['model']=='group':
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         fname = model_path + f"/{m}_{config['model']}"
         mo,inf = cio.load_model(fname)
         fitted_model.append(mo)
         train_info.append(inf)

   elif config['model']=='ind' or config['model']=='group_loo':
      # get list of subject for model
      model_subj = get_subj_list(config["subj_list"], config["dataset"])
      if config['model']=='group_loo':
         model_subj = [s+'_group_loo' for s in model_subj]
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         fm=[]
         ti = []
         for sub in model_subj:
            fname = model_path + f"/{m}_{sub}"
            mo,inf = cio.load_model(fname)
            fm.append(mo)
            ti.append(inf)
         fitted_model.append(fm)
         train_info.append(ti)

   elif config['model']=='loo':
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         if m.startswith(d):
            ext = m[len(d):]
         else:
            ext = ''
         # ext = '_' + m.split('_')[-1]
         # if 'L2reghalf' in d:
         #    fm,fi = calc_avrg_model(config['dataset'],d,ext,
         #                            cerebellum=config['cerebellum'],
         #                            avrg_mode='loo-half')
         # else:
         fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                 cerebellum=config['cerebellum'],
                                 avrg_mode='loo_sep')
         fitted_model.append(fm)
         train_info.append(fi)

   elif config['model']=='mix':
      # get list of subject for model
      model_subj = get_subj_list(config["subj_list"], config["dataset"])
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         ext = '_' + m.split('_')[-1]
         fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                 cerebellum=config['cerebellum'],
                                 model_subj=model_subj,
                                 avrg_mode=config['model'],
                                 mix_param=config['mix_param'],
                                 mix_model=config['mix_model'])
         fitted_model.append(fm)
         train_info.append(fi)

   elif config['model']=='mix_loo':
      # get list of subject for model
      model_subj = get_subj_list(config["subj_list"], config["dataset"])
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         ext = '_' + m.split('_')[-1]
         fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                 cerebellum=config['cerebellum'],
                                 model_subj=model_subj,
                                 avrg_mode=config['model'],
                                 mix_param=config['mix_param'])
         fitted_model.append(fm)
         train_info.append(fi)

   return fitted_model, train_info


def eval_model(model_dirs, model_names, eval_config, model_config):
   """Evaluate group model on a specific dataset and session.
   
   Args:
      model_dirs (list): list of model directories
      model_names (list): list of full model names (without .h5) to evaluate
      eval_config (dict): dictionary with evaluation parameters
      model_config (dict): dictionary with model parameters

   Returns:
      eval_df (pd.DataFrame): DataFrame with evaluation results
      eval_voxels (defaultdict): dictionary with voxel-wise evaluation results
   """

   # initialize eval dictionary
   eval_df = pd.DataFrame()
   eval_voxels = defaultdict(list)

   # get list of subjects
   eval_subj = get_subj_list(eval_config["subj_list"], eval_config["eval_dataset"])

   # Get the list of fitted models
   fitted_model, train_info = get_fitted_models(model_dirs, model_names, model_config)

   # Get cerebellar abd cortical data
   YY, info = get_cerebellar_data(eval_config["eval_dataset"], eval_config["eval_ses"], eval_subj, eval_config)
   XX, info = get_cortical_data(eval_config["eval_dataset"], eval_config["eval_ses"], eval_subj, eval_config)

   # Calculate group reliability of subjects
   group_noiseceil_lower = frel.between_subj_loo(YY)
   group_noiseceil_upper = frel.between_subj_avrg(YY)

   # Evaluate models
   for i, sub in enumerate(eval_subj):
      print(f'- Evaluate {sub}')
      # Loop over models
      if eval_config['cortical_act'] == 'ind':
         X = XX[i,:,:] # get the data for the subject
      elif eval_config['cortical_act'] == 'avg':
         X = XX.mean(axis=0) # get average cortical data
      elif eval_config['cortical_act'] == 'loo':
         n_subj = len(eval_config['subj_list'])
         subj_vec = np.arange(n_subj)
         X = XX[subj_vec!=i,:,:].mean(axis=0) # get average cortical data
      Y = YY[i,:,:] # get the data for the subject

      for j, (fm, tinfo) in enumerate(zip(fitted_model, train_info)):
         # Use subject-specific model? (indiv or loo or mix)
         if (isinstance(fm,list)):
            fitM = fm[i]
         else:
            fitM = fm

         if (isinstance(tinfo,list)):
            ti = tinfo[i]
         else:
            ti = tinfo

         # Get model predictions
         Y_pred = fitM.predict(X)

         eval_sub = {"eval_subj": sub,
                     "num_regions": X.shape[1]}

         # Copy over all scalars or strings to eval_all dataframe:
         for key, value in ti.items():
            if not isinstance(value,(list,pd.Series,np.ndarray)):
               eval_sub.update({key: value})
         for key, value in eval_config.items():
            if not isinstance(value,(list,pd.Series,np.ndarray)):
               eval_sub.update({key: value})
         for key, value in model_config.items():
            if not isinstance(value,(list,pd.Series,np.ndarray)):
               eval_sub.update({key: value})

         # add evaluation (summary)
         evals = eval_metrics(Y=Y, Y_pred=Y_pred, info = info)

         # add evaluation (voxels)
         for k, v in evals.items():
            if "vox" in k:
               eval_voxels[k].append(v)
            else:
               eval_sub[k]=v

         # Add group noise ceiling 
         eval_sub['group_noiseceil_Y_upper'] = group_noiseceil_upper[i]
         eval_sub['group_noiseceil_Y_lower'] = group_noiseceil_lower[i]

         # don't save voxel data to summary
         eval_df = pd.concat([eval_df,pd.DataFrame(eval_sub,index=[0])],ignore_index= True)

   return eval_df, eval_voxels


def comb_eval(models = ['Md_s1'],
              eval_data = ["MDTB", "WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC"],
              methods = ['L2reg'],
              cerebellum = 'MNISymC3',
              eval_t = 'eval',
              eval_type = None):
   """Combine different tsv files from different datasets into one dataframe

   Args:
      models (list): Strings of eval_ids to include. Defaults to ['Md_s1'].
      eval_data (list): Eval datasets to include. Defaults to ["MDTB", "WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC"].
      cerebellum (str, optional): Cerebellum atlas to use. Defaults to 'MNISymC3'.
      eval_t (str, optional): Evaluation type. Defaults to 'eval'.
      eval_type (str, optional): Evaluation type. Defaults to None.

   Returns:
       _type_: _description_
   """
   T = []

   for dataset in eval_data:
      for m in models:
         for meth in methods:
            if eval_type is None:
               f = gl.conn_dir + f'/{cerebellum}/{eval_t}/{dataset}_{meth}_{m}.tsv'
            else:
               f = gl.conn_dir + f'/{cerebellum}/{eval_t}/{dataset}_{eval_type}_{meth}_{m}.tsv'
            # get the dataframe
            if os.path.exists(f):
               dd = pd.read_csv(f, sep='\t')
               # add a column for the name of the dataset
               # get the noise ceilings

               # Remove negative values from dd.noise
               dd.group_noiseceil_Y_upper = dd.group_noiseceil_Y_upper.apply(lambda x: np.nan if x < 0 else x)
               dd.group_noiseceil_Y_lower = dd.group_noiseceil_Y_lower.apply(lambda x: np.nan if x < 0 else x)
               dd['group_noiseceiling'] = ((dd.group_noiseceil_Y_upper)+(dd.group_noiseceil_Y_lower)) /2
               dd['R_eval_adj'] = dd.R_eval/dd["group_noiseceiling"]
               T.append(dd)
   df = pd.concat(T, ignore_index=True)
   return df


def calc_avrg_model(train_dataset,
                    mname_base,
                    mname_ext,
                    cerebellum = 'MNISymC3',
                    parameters = ['coef_'],
                    avrg_mode = 'avrg_sep',
                    mix_param = [],
                    mix_model = None,
                    subj = 'all',
                    model_subj = 'all'):
   """Get the fitted models from all the subjects in the training data set and create group-averaged model

   Args:
      train_dataset (str): name of the training dataset
      mname_base (str): Directory name for model (e.g. MDTB_all_Icosahedron1002_L2reg)
      mname_ext (str): Extension of name - typically logalpha (_A0)
      parameters (list): List of parameters to average
      avrg_mode (str): Averaging mode
      mix_param (list): List of mixing parameters
      mix_model (str): Name of the mixing model
      subj (str): Subject to use for fitting
      model_subj (str): Subject to use for model

   Returns:
      avrg_model (list): List of fitted models for each subject
      df (pd.DataFrame): DataFrame with training information for each subject
   """

   # Get the list of subjects
   subject_list = get_subj_list(subj, train_dataset)
   model_subject_list = get_subj_list(model_subj, train_dataset)

   # get the directory where models are saved
   model_path = gl.conn_dir + f"/{cerebellum}/train/{mname_base}/"

   # Collect the parameters in lists
   param_lists={}
   for p in parameters:
      param_lists[p]=[]

   # Loop over subjects
   df = pd.DataFrame()
   for sub in subject_list:
      print(f"- getting weights for {sub}")
      # load the model and info file
      fname = model_path + f"/{mname_base}{mname_ext}_{sub}"
      fitted_model, info = cio.load_model(fname)
      df = pd.concat([df, pd.DataFrame(info, index=[0])], ignore_index=True)

      for p in parameters:
         param_lists[p].append(getattr(fitted_model, p))

   avrg_model = fitted_model
   if avrg_mode=='avrg_sep':
      for p in parameters:
         P = np.stack(param_lists[p], axis=0)
         setattr(avrg_model, p, P.mean(axis=0))

   elif avrg_mode=='loo_sep':
      avrg_model = []
      subj_ind = np.arange(len(subject_list))
      for s, sub in enumerate(model_subject_list):
         avrg_model.append(copy(fitted_model))
      for p in parameters:
         P = np.stack(param_lists[p], axis=0)
         for s, sub in enumerate(model_subject_list):
            sel_ind = list(subject_list).index(sub)
            setattr(avrg_model[s], p, P[subj_ind != sel_ind].mean(axis=0))

   elif avrg_mode=='mix_loo':
      avrg_model = []
      portion_value = float(mix_param) / 100
      print(f"portion_value = {portion_value}")
      subj_ind = np.arange(len(subject_list))
      for s,sub in enumerate(subject_list):
         avrg_model.append(copy(fitted_model))
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         for s,sub in enumerate(subject_list):
            attr_value = P[subj_ind!=s].mean(axis=0)*(1-portion_value) + P[subj_ind==s].mean(axis=0)*(portion_value)
            setattr(avrg_model[s],p,attr_value)

   elif avrg_mode=='mix':
      avrg_model = []
      portion_value = float(mix_param) / 100
      print(f"portion_value = {portion_value}")
      subj_ind = np.arange(len(subject_list))
      for s,sub in enumerate(subject_list):
         avrg_model.append(copy(fitted_model))
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         for s,sub in enumerate(subject_list):
            attr_value = getattr(mix_model,p)*(1-portion_value) + P[subj_ind==s].mean(axis=0)*(portion_value)
            setattr(avrg_model[s],p,attr_value)
         
   # Assemble the summary
   # first fill in NoneTypes with Nans. This is a specific case for WTA
   df.logalpha.fillna(value=np.nan, inplace=True)
   dict = {'train_dataset': df.train_dataset[0],
           'train_ses': df.train_ses[0],
           'train_type': df.type[0],
           'cerebellum': df.cerebellum[0],
           'cortex': df.cortex[0],
           'method': df.method[0],
           'logalpha': float(df.logalpha[0])
           }
   
   # save dict as json
   return avrg_model, dict
