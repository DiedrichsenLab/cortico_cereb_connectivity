"""Main module for training and evaluating connectivity models.
   Designed to work together with Functional_Fusion package.
   Dataset, session, and parcellation names are as in Functional_Fusion.
   The main work is being done by train_model and eval_model functions.
   @authors: Ladan Shahshahani, Maedbh King, Jörn Diedrichsen, Ali Shahbazi
"""

# from audioop import cross
import os
import sys
import numpy as np
import pandas as pd
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
                     train_ses = "ses-s1",
                     train_run = 'all',
                     train_cond_num = 'all',
                     subj_list = 'all',
                     method = "L2regression",
                     log_alpha = 8,
                     cerebellum = "SUIT3",
                     cortex = "fs32k",
                     parcellation = "Icosahedron1002",
                     type = "CondHalf",
                     crossed = "half", # or None
                     validate_model = True,
                     cv_fold = 4,
                     add_rest = True,
                     append = False,
                     cortical_act = 'ind',
                     std_cortex = None,
                     std_cerebellum = None
                     ):
   """get_train_config
   Function to create a config dictionary containing the info for the training

   Args:
      train_dataset (str): training_dataset. Defaults to "MDTB".
      train_ses (str): Training session. Defaults to "ses-s1".
      method (str): Model class. Defaults to "L2regression".
      log_alpha (int): log of regularization. Defaults to 8.
      cerebellum (str): Atlas for cerebellum. Defaults to "SUIT3".
      cortex (str): Atlas for neocortex. Defaults to "fs32k".
      parcellation (str): Parcellation for cortex. Defaults to "Icosahedron-1002_Sym.32k".
      type (str): _description_. Defaults to "CondHalf".
      crossed (str): Double crossvalidation cortex-cerebellum. ("half" (default) or None)
      validate_model (bool): Do cross-validation in training set for hyperparameter tuning? Defaults to True.
      cv_fold (int): Number of validation folds. Defaults to 4.
      add_rest (bool): Add rest condition to each session and half. Defaults to True.
      std_cortex(): z-Standardize the cortical data. (Defaults to None)
      std_cerebelum(): z-Standardize the cortical data. (Defaults to None)
   Returns:
      dict: Dictionary containing the default training configuration
   """
   train_config = {}
   train_config['train_dataset'] = train_dataset # name of the dataset to be used in
   train_config['train_ses'] = train_ses
   train_config['train_run'] = train_run
   train_config['train_cond_num'] = train_cond_num
   train_config['subj_list'] = subj_list
   train_config['method'] = method   # method used in modelling (see model.py)
   train_config['logalpha'] = log_alpha # alpha will be np.exp(log_alpha)
   train_config['cerebellum'] = cerebellum
   train_config['cortex'] = cortex
   train_config['parcellation'] = parcellation
   train_config['crossed'] = crossed
   train_config["validate_model"] = validate_model
   train_config["type"] = type
   train_config["cv_fold"] = cv_fold, #TO IMPLEMENT: "ses_id", "run", "dataset", "tasks"
   train_config['add_rest'] = add_rest
   train_config['cortical_act'] = cortical_act
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
            eval_ses = 'ses-s2',
            subj_list = 'all',
            eval_run = 'all',
            eval_cond_num = 'all',
            cerebellum = 'SUIT3',
            cortex = "fs32k",
            parcellation = "Icosahedron1002",
            crossed = "half", # or None
            type = "CondHalf",
            splitby = None,
            add_rest = True,
            std_cortex = 'parcel',
            std_cerebellum = 'global',
            cortical_act = 'ind'):
   """
   create a config file for evaluation
   Args:
      eval_dataset (str): evaluation dataset. Defaults to 'MDTB'.
      eval_ses (str): evaluation session. Defaults to 'ses-s2'.
      subj_list (str or list): List of subjects to evaluate. Defaults to 'all'.
      eval_run (str or list): List of runs to evaluate. Defaults to 'all'.
      eval_cond_num (str or list): List of conditions to evaluate. Defaults to 'all'.
      cerebellum (str): Atlas for cerebellum. Defaults to 'SUIT3'.
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
   eval_config['eval_run'] = eval_run
   eval_config['eval_cond_num'] = eval_cond_num
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
        X (nd-array):
        Y (nd-array):
    Returns:
        rmse_train (scalar), R_train (scalar)
    """
    Y_pred = model.predict(X)

    # get train rmse and R
    R_train, _ = ev.calculate_R(Y, Y_pred)
    R2_train,_ = ev.calculate_R2(Y, Y_pred)

    return R_train, R2_train

def validate_metrics(model, X, Y, cv_fold):
    """computes CV training metrics (rmse and R) on X and Y

    Args:
        model (class instance): must be fitted model
        X (nd-array):
        Y (nd-array):
        cv_fold (int): number of CV folds
    Returns:
        rmse_cv (scalar), R_cv (scalar)
    """

    # TO DO: implement train/validate splits for "sess", "run"
    r_cv_all = cross_val_score(model, X, Y, scoring=ev.calculate_R_cv, cv=cv_fold)

    return np.nanmean(r_cv_all)

def eval_metrics(Y, Y_pred, info):
    """Compute evaluation, returning summary and voxel data.

    Args:
        Y (np array):
        Y_pred (np array):
        Y_info (pd dataframe):
    Returns:
        dict containing evaluations (R, R2, noise).
    """
    # initialise dictionary
    data = {}

    # Add the evaluation
    data["R_eval"], data["R_vox"] = ev.calculate_R(Y=Y, Y_pred=Y_pred)

    # R between predicted and observed
    data["R2_eval"], data["R2_vox"] = ev.calculate_R2(Y=Y, Y_pred=Y_pred)

    # R2 between predicted and observed
    (
        data["noise_Y_R"],
        data["noise_Y_R_vox"],
        data["noise_Y_R2"],
        data["noise_Y_R2_vox"],
    ) = ev.calculate_reliability(Y=Y, dataframe = info)

    # Noise ceiling for predicted cerebellum (squared)
    (
        data["noise_X_R"],
        data["noise_X_R_vox"],
        data["noise_X_R2"],
        data["noise_X_R2_vox"],
    ) = ev.calculate_reliability(Y=Y_pred, dataframe = info)

    # calculate noise ceiling
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)

      data["noiseceiling_Y_R_vox"] = np.sqrt(data["noise_Y_R_vox"])
      data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"]) * np.sqrt(data["noise_X_R_vox"])
    return data

def cross_data(Y,info,mode):
   """Cross data across halves
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
   return Ys,infos

def std_data(Y,mode):
   if mode is None:
      return Y
   elif mode=='parcel':
      sc=np.sqrt(np.nansum(Y ** 2, 0) / Y.shape[0])
      return  np.nan_to_num(Y/sc)
   elif mode=='global':
      sc=np.sqrt(np.nansum(Y ** 2) / Y.size)
      return np.nan_to_num(Y/sc)
   elif mode=='norm':
      sc=np.linalg.norm(Y, ord='fro')
      return np.nan_to_num(Y/sc)
   else:
      raise ValueError('std_mode must be None, "voxel" or "global" or "norm"')

def train_model(config, save_path=None, mname=None):
   """
   training a specific model based on the config file created
   model will be trained on cerebellar voxels and average within cortical tessels.
   Args:
      config (dict)      - dictionary with configuration parameters
   Returns:
      conn_model_list (list)    - list of trained models on the list of subjects / log-alphas
      config (dict)             - dictionary containing info for training. Can be saved as json
      train_df (pd.DataFrame)   - dataframe containing training information
   """

   # get list of subjects
   config['subj_list'] = get_subj_list(config['subj_list'], config["train_dataset"])

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

   # Load the data
   YY, info, _ = fdata.get_dataset(gl.base_dir,
                                 config["train_dataset"],
                                 atlas=config["cerebellum"],
                                 sess=config["train_ses"],
                                 type=config["type"],
                                 subj=config['subj_list'].tolist())
   XX, info, _ = fdata.get_dataset(gl.base_dir,
                                 config["train_dataset"],
                                 atlas=config["cortex"],
                                 sess=config["train_ses"],
                                 type=config["type"],
                                 subj=config['subj_list'].tolist())
   # Average the cortical data over pacels
   X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(config['label_img'], unite_struct = False)
   # get the mean across tessels for cortical data
   XX, labels = fdata.agg_parcels(XX, X_atlas.label_vector,fcn=np.nanmean)

   # Remove Nans
   YY = np.nan_to_num(YY)
   XX = np.nan_to_num(XX)

   # Add rest condition?
   if config["add_rest"]:
      YY,_ = add_rest(YY,info)
      XX,info = add_rest(XX,info)

   # train only on some runs?
   if config["train_run"]!='all':
      if isinstance(config["train_run"], list):
         run_mask = info['run'].isin(config["train_run"])
         YY = YY[..., run_mask.values, :]
         XX = XX[..., run_mask.values, :]
         info = info[run_mask]

   # train only on some conds?
   if config['train_cond_num']!='all':
      if isinstance(config["train_cond_num"], list):
         cond_mask = info['cond_num'].isin(config["train_cond_num"])
         YY = YY[..., cond_mask.values, :]
         XX = XX[..., cond_mask.values, :]
         info = info[cond_mask]

   #Definitely subtract intercept across all conditions
   XX = (XX - XX.mean(axis=-2,keepdims=True))
   YY = (YY - YY.mean(axis=-2,keepdims=True))

   for i in range(XX.shape[0]):
      if 'std_cortex' in config.keys():
         XX[i,:,:] = std_data(XX[i,:,:],config['std_cortex'])
      if 'std_cerebellum' in config.keys():
         YY[i,:,:] = std_data(YY[i,:,:],config['std_cerebellum'])

      # cross the halves within each session
      if config["crossed"] is not None:
         YY[i,:,:] = cross_data(YY[i,:,:],info,config["crossed"])

   # loop over subjects and train models
   for i,sub in enumerate(config["subj_list"]):
      if config['cortical_act'] == 'ind':
         X=XX[i,:,:] # get the data for the subject
      elif config['cortical_act'] == 'avg':
         X=XX.mean(axis=0) # get average cortical data
      Y=YY[i,:,:] # get the data for the subject

      for la in config["logalpha"]:
         print(f'- Train {sub} {config["method"]} logalpha {la}')

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
         elif config["method"] == 'L2reg2':
            conn_model.fit(X, Y, info)
         else:
            conn_model.fit(X, Y)
         R_train,R2_train = train_metrics(conn_model, X, Y)
         # conn_model_list.append(conn_model)

         # collect train metrics ( R)
         model_info = {
                        "subj_id": sub,
                        "mname": mname_spec,
                        "R_train": R_train,
                        "R2_train": R2_train,
                        "num_regions": X.shape[1],
                        "logalpha": la
                        }

         # run cross validation and collect metrics (rmse and R)
         if config['validate_model']:
            R_cv = validate_metrics(conn_model, X, Y, config["cv_fold"][0])
            model_info.update({"R_cv": conn_model.R_cv})

         # Copy over all scalars or strings from config to eval dict:
         for key, value in config.items():
            if not isinstance(value, (list, dict,pd.Series,np.ndarray)):
               model_info.update({key: value})
         # Save the individuals info files
         cio.save_model(conn_model,model_info,save_path + "/" + mname_spec)
         train_info = pd.concat([train_info,pd.DataFrame(model_info)],ignore_index= True)
   train_info.to_csv(train_info_name,sep='\t')
   return config, conn_model_list, train_info

def get_model_names(train_dataset,train_ses,parcellation,method,ext_list):
   """ Makes a list of model dirs and model names, based on training set, etc.

   Args:
         train_dataset (str): trainign dataset 
         train_ses (str): Session 
         parcellation (str): Cortical parcellation
         method (str): 'L2regression', 'WTA', 'L1regression', 'NNlS', etc
         ext_list (list): List of extensions (numeric or string) to add to model name
   Returns:
         dirname (list): List of model directories 
         mname (list): List of model names 
   """   
   dirname=[] # Model directory name
   mname=[] # Model name - without the individual, average, or loo extension

   # Build list of to-be-evaluated models
   for a in ext_list:
      dirname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
      if a is None:
         mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
      if isinstance(a,int):
         mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_A{a}")
      elif isinstance(a,str):
         mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_{a}")
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
   
   return subj_list


def get_fitted_models(model_dirs,model_names,config):
   """Builds a list of fitted models from the saved files
   In case of individual-specific models (ind or loo), it builds a list of lists.

   Args:
       model_dirs (_type_): List of dirctory names for models 
       model_names (_type_): List of model names (without subject extension)
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
   elif config['model']=='avg':
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         fname = model_path + f"/{m}_avg"
         mo,inf = cio.load_model(fname)
         fitted_model.append(mo)
         train_info.append(inf)
   elif config['model']=='ind':
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         fm=[]
         ti = []
         for sub in config['subj_list']:
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
         ext = '_' + m.split('_')[-1]
         if 'L2reghalf' in d:
            fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                    cerebellum=config['cerebellum'],
                                    avrg_mode='loo-half')
         else:
            fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                    cerebellum=config['cerebellum'],
                                    avrg_mode='loo_sep')
         fitted_model.append(fm)
         train_info.append(fi)
   elif config['model']=='mix':
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         ext = '_' + m.split('_')[-1]
         fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                 cerebellum=config['cerebellum'],
                                 mix_subj=config['subj_list'],
                                 avrg_mode=config['model'],
                                 mix_param=config['mix_param'])
         fitted_model.append(fm)
         train_info.append(fi)
   elif config['model'].startswith('bayes'):
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         ext = '_' + m.split('_')[-1]
         fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                 cerebellum=config['cerebellum'],
                                 mix_subj=config['subj_list'],
                                 avrg_mode=config['model'])
         fitted_model.append(fm)
         train_info.append(fi)

   
   return fitted_model, train_info

def eval_model(model_dirs,model_names,eval_config,model_config):
   """
   evaluate group model on a specific dataset and session
   if model_config['model']=='avg' it will average the models across subjects
   if model_config['model']=='ind' it will evaluate each subejct individually
   if model_config['model']=='loo' it will average all other subjects
   if model_config['model']=='mix' it will do: p*subject + (1-p)*loo
   if model_config['model']=='bayes' it will integrate individual weights with bayes rule
   For 'ind', 'loo', and 'mix' training and evaluation dataset must be the same 
   Args:
      model_dirs (list)  - list of model directories
      model_names (list) - list of full model names (without .h5) to evaluate
      eval_config (dict)      - dictionary with evaluation parameters
      model_config (dict)     - dictionary with model parameters
   """
   # initialize eval dictionary
   eval_df = pd.DataFrame()
   eval_voxels = defaultdict(list)

   # get list of subjects
   eval_config["subj_list"] = get_subj_list(eval_config["subj_list"], eval_config["eval_dataset"])
   
   # get list of subject for model
   model_config["subj_list"] = get_subj_list(model_config["subj_list"], model_config["dataset"])

   # Get the list of fitted models
   fitted_model,train_info = get_fitted_models(model_dirs,model_names,model_config)

   # Load the data
   YY, info, _ = fdata.get_dataset(gl.base_dir,
                                    eval_config["eval_dataset"],
                                    atlas=eval_config["cerebellum"],
                                    sess=eval_config["eval_ses"],
                                    type=eval_config["type"],
                                    subj=eval_config["subj_list"].tolist())
   XX, info, _ = fdata.get_dataset(gl.base_dir,
                                    eval_config["eval_dataset"],
                                    atlas=eval_config["cortex"],
                                    sess=eval_config["eval_ses"],
                                    type=eval_config["type"],
                                    subj=eval_config["subj_list"].tolist())
   # Average the cortical data over parcels
   X_atlas, _ = at.get_atlas(eval_config['cortex'],gl.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(eval_config['label_img'], unite_struct = False)
   # get the mean across tessels for cortical data
   XX, labels = fdata.agg_parcels(XX, X_atlas.label_vector,fcn=np.nanmean)
   
   # Remove Nans
   YY = np.nan_to_num(YY)
   XX = np.nan_to_num(XX)

   # Add explicit rest to sessions
   if eval_config["add_rest"]:
      YY,_ = add_rest(YY,info)
      XX,info = add_rest(XX,info)

   # eval only on some runs?
   if eval_config["eval_run"]!='all':
      if isinstance(eval_config["eval_run"], list):
         run_mask = info['run'].isin(eval_config["eval_run"])
         YY = YY[...,run_mask.values, :]
         XX = XX[...,run_mask.values, :]
         info = info[run_mask]

   # eval only on some conds?
   if eval_config['eval_cond_num']!='all':
      if isinstance(eval_config["eval_cond_num"], list):
         cond_mask = info['cond_num'].isin(eval_config["eval_cond_num"])
         YY = YY[...,cond_mask.values, :]
         XX = XX[...,cond_mask.values, :]
         info = info[cond_mask]

   #Definitely subtract intercept across all conditions
   XX = (XX - XX.mean(axis=-2,keepdims=True))
   YY = (YY - YY.mean(axis=-2,keepdims=True))

   for i in range(XX.shape[0]):
      if 'std_cortex' in eval_config.keys():
         # Standardize the cortical data
         XX[i,:,:] = std_data(XX[i,:,:],eval_config['std_cortex'])
      if 'std_cerebellum' in eval_config.keys():
         YY[i,:,:] = std_data(YY[i,:,:],eval_config['std_cerebellum'])

      # cross the halves within each session
      if eval_config["crossed"] is not None:
         YY[i,:,:] = cross_data(YY[i,:,:],info,eval_config["crossed"])

   # Caluclated group reliability of subjects 
   group_noiseceil_lower = frel.between_subj_loo(YY)
   group_noiseceil_upper = frel.between_subj_avrg(YY)

   for i, sub in enumerate(eval_config["subj_list"]):
      print(f'- Evaluate {sub}')
      # Loop over models
      if eval_config['cortical_act'] == 'ind':
         X=XX[i,:,:] # get the data for the subject
      elif eval_config['cortical_act'] == 'avg':
         X=XX.mean(axis=0) # get average cortical data
      Y=YY[i,:,:] # get the data for the subject

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

def comb_eval(models=['Md_s1'],
              eval_data=["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC"],
              methods =['L2regression'],
              cerebellum='SUIT3',
              eval_t = 'eval',
              eval_type = None):
   """Combine different tsv files from different datasets into one dataframe

   Args:
       models (list): Strings of eval_ids to include. Defaults to ['Md_s1'].
       eval_data (list): Evaldatasets _description_. Defaults to ["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC"].
       cerebellum (str, optional): _description_. Defaults to 'SUIT3'.

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

               # Remove negative values from dd.noise_X_R
               dd.noise_X_R = dd.noise_X_R.apply(lambda x: np.nan if x < 0 else x)
               dd.noise_Y_R = dd.noise_Y_R.apply(lambda x: np.nan if x < 0 else x)
               dd['noiseceiling_Y']=np.sqrt(dd.noise_Y_R)
               dd['noiseceiling_XY']=np.sqrt(dd.noise_Y_R)*np.sqrt(dd.noise_X_R)
               dd['R_eval_adj'] = dd.R_eval/dd["noiseceiling_XY"]
               T.append(dd)
   df = pd.concat(T,ignore_index=True)
   return df


def calc_wopt_var(sub_weight_variance,
                  avrg_mode,
                  S):
   
   uncertainty = np.reciprocal(sub_weight_variance)
   wopt_variance_list = [np.nansum(uncertainty[s], axis=0) for s in range(S)]
   if 'vox' in avrg_mode:
      for wopt_var in wopt_variance_list:
         wopt_var[wopt_var == 0] = np.nan

   return wopt_variance_list


def calc_bayes_avrg(param_lists,
                    subject_list,
                    avrg_mode,
                    parameters=['coef_','coef_var']):
   # stack for numpy functions
   param_coef_ = np.stack(param_lists['coef_'], axis=0)
   
   # dimensions
   S = len(subject_list)
   n_vox, n_region = param_coef_[0].shape

   if 'loo' in avrg_mode:
      param_coef_var = np.stack(param_lists['coef_var'], axis=0)
      # calculate adjusted variance (vs: Sx(S-1))
      vg, vs = decompose_variance(param_coef_, np.nanmean(param_coef_var, axis=1)/n_region, model_type="loo")

      # reshape param_coef_var for loo
      idx = np.arange(24)[:, None]
      param_coef_var = param_coef_var[np.arange(24) != idx].reshape(S, S-1, n_vox)
      sub_var = vs[:, :, np.newaxis]*n_region + param_coef_var

      if not 'vox' in avrg_mode:
         sub_var = np.nanmean(sub_var, axis=-1)
         coef_norm = np.linalg.norm(param_coef_, axis=(1,2))[np.arange(24) != idx].reshape(S, S-1)
      else:
         coef_norm = np.linalg.norm(param_coef_, axis=2)[np.arange(24) != idx].reshape(S, S-1, n_vox)

      signal_norm2 = coef_norm**2 - n_vox*sub_var
      param_coef_ /= np.sqrt(signal_norm2).reshape(S, *([1]* (param_coef_.ndim - signal_norm2.ndim)))
      sub_var /= signal_norm2

      wopt_variance_list = calc_wopt_var(sub_weight_variance=sub_var,
                                         avrg_mode=avrg_mode,
                                         S=S)
      param_w_opt = {}
      if 'vox' in avrg_mode:
         # divide each weights by its variance
         P = [np.delete(param_coef_, s, axis=0) / sub_var[s, :, :, None] for s in range(S)]
         # sum over subjects and normalize
         param_w_opt['coef_'] = [np.nansum(P[s], axis=0) / wopt_variance_list[s][:, None] for s in range(S)]
      else:
         # divide each weights by its variance
         P = [np.delete(param_coef_, s, axis=0) / sub_var[s, :, None, None] for s in range(S)]
         # sum over subjects and normalize
         param_w_opt['coef_'] = [np.nansum(P[s], axis=0) / wopt_variance_list[s] for s in range(S)]

      param_w_opt['coef_'] = [np.nan_to_num(arr) for arr in param_w_opt['coef_']]
      param_w_opt['coef_var'] = wopt_variance_list
   elif 'half' in avrg_mode:
      param_coef_1 = np.stack(param_lists['coef_1'], axis=0)
      param_coef_2 = np.stack(param_lists['coef_2'], axis=0)
      vg, vs, vm = decompose_variance_half(np.stack((param_coef_1, param_coef_2), axis=1))
      sub_var = vs + vm/2

      coef_norm = np.linalg.norm(param_coef_, axis=(1,2))
      signal_norm2 = coef_norm**2 - n_vox*n_region*sub_var
      param_coef_ /= np.sqrt(signal_norm2).reshape(S, *([1]* (param_coef_.ndim - signal_norm2.ndim)))
      sub_var /= signal_norm2

      param_w_opt = {}
      wopt_variance = np.nan_to_num(np.nansum(1 / sub_var, axis=0))
      # divide each weights by its variance
      P = param_coef_ / sub_var[:, None, None]
      # sum over subjects and normalize
      param_w_opt['coef_'] = np.nan_to_num(np.nansum(P, axis=0) / wopt_variance)
      param_w_opt['coef_var'] = wopt_variance
   else:
      param_coef_var = np.stack(param_lists['coef_var'], axis=0)
      # calculate adjusted variance (vs: S)
      vg, vs = decompose_variance(param_coef_, np.nanmean(param_coef_var, axis=1)/n_region)
      sub_var = vs[:, np.newaxis]*n_region + param_coef_var

      if not 'vox' in avrg_mode:
         sub_var = np.nanmean(sub_var, axis=-1)
         coef_norm = np.linalg.norm(param_coef_, axis=(1,2))
      else:
         coef_norm = np.linalg.norm(param_coef_, axis=2)
   
      signal_norm2 = coef_norm**2 - n_vox*sub_var
      param_coef_ /= np.sqrt(signal_norm2).reshape(S, *([1]* (param_coef_.ndim - signal_norm2.ndim)))
      sub_var /= signal_norm2

      param_w_opt = {}
      wopt_variance = np.nan_to_num(np.nansum(1 / sub_var, axis=0))
      if 'vox' in avrg_mode:
         # divide each weights by its variance
         P = param_coef_ / sub_var[:, :, None]
         # sum over subjects and normalize
         param_w_opt['coef_'] = np.nansum(P, axis=0) / wopt_variance[:, None]
      else:
         # divide each weights by its variance
         P = param_coef_ / sub_var[:, None, None]
         # sum over subjects and normalize
         param_w_opt['coef_'] = np.nansum(P, axis=0) / wopt_variance
      param_w_opt['coef_'] = np.nan_to_num(param_w_opt['coef_'])
      param_w_opt['coef_var'] = wopt_variance

   return param_w_opt


def calc_avrg_model(train_dataset,
                    mname_base,
                    mname_ext,
                    cerebellum='SUIT3',
                    parameters=['coef_'],
                    avrg_mode='avrg_sep',
                    mix_param=[],
                    subj='all',
                    model_subj='all',
                    mix_subj='all'):
   """Get the fitted models from all the subjects in the training data set
      and create group-averaged model
   Args:
       train_dataset (str): _description_
       mname_base (str): Directory name for model (MDTB_all_Icosahedron1002_L2regression)
       mname_ext (str): Extension of name - typically logalpha
       (_A0)
       parameters (list): List of parameters to average
   """
   subject_list = get_subj_list(subj, train_dataset)
   model_subject_list = get_subj_list(model_subj, train_dataset)

   # get the directory where models are saved
   model_path = gl.conn_dir + f"/{cerebellum}/train/{mname_base}/"

   # Collect the parameters in lists
   if avrg_mode.startswith('bayes') & ('half' not in avrg_mode):
      parameters = ['coef_', 'coef_var']
   elif avrg_mode.startswith('bayes') & ('half' in avrg_mode):
      parameters = ['coef_', 'coef_1', 'coef_2']
   elif avrg_mode=='avg-half':
      parameters = ['coef_', 'coef_1', 'coef_2']
      avrg_mode = 'avrg_sep'
   elif avrg_mode=='loo-half':
      parameters = ['coef_', 'coef_1', 'coef_2']
      avrg_mode = 'loo_sep'
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
      df = pd.concat([df,pd.DataFrame(info,index=[0])],ignore_index=True)

      for p in parameters:
         param_lists[p].append(getattr(fitted_model,p))

   avrg_model = fitted_model
   if avrg_mode=='avrg_sep':
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         setattr(avrg_model,p,P.mean(axis=0))

   elif avrg_mode=='loo_sep':
      avrg_model = []
      subj_ind = np.arange(len(subject_list))
      for s,sub in enumerate(model_subject_list):
         avrg_model.append(copy(fitted_model))
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         for s,sub in enumerate(model_subject_list):
            sel_ind = list(subject_list).index(sub)
            setattr(avrg_model[s],p,P[subj_ind!=sel_ind].mean(axis=0))

   elif avrg_mode=='mix':
      avrg_model = []
      portion_value = mix_param / 100
      print(f"portion_value = {portion_value}")
      subj_ind = np.arange(len(subject_list))
      for s,sub in enumerate(subject_list):
         avrg_model.append(copy(fitted_model))
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         for s,sub in enumerate(subject_list):
            sel_subj = subject_list[subject_list.isin(mix_subj)].index.tolist()
            if s in sel_subj:
               sel_subj.remove(s)
            attr_value = P[sel_subj].mean(axis=0)*(1-portion_value) + P[subj_ind==s].mean(axis=0)*(portion_value)
            setattr(avrg_model[s],p,attr_value)

   elif avrg_mode.startswith('bayes'):
      param_w_opt = calc_bayes_avrg(parameters=parameters,
                              param_lists=param_lists,
                              subject_list=subject_list,
                              avrg_mode=avrg_mode)
      if 'loo' in avrg_mode:
         avrg_model = []
         for s,sub in enumerate(subject_list):
            avrg_model.append(copy(fitted_model))
         for s,(coef,var) in enumerate(zip(param_w_opt['coef_'], param_w_opt['coef_var'])):
            setattr(avrg_model[s], 'coef_', coef)
            setattr(avrg_model[s], 'coef_var', var)
      else:
         avrg_model = fitted_model
         setattr(avrg_model, 'coef_', param_w_opt['coef_'])
         setattr(avrg_model, 'coef_var', param_w_opt['coef_var'])

   elif avrg_mode=='avg-half':
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         setattr(avrg_model,p,P.mean(axis=0))
      setattr(avrg_model, 'coef_', (avrg_model.coef_1 + avrg_model.coef_2)/2)

         
   # Assemble the summary
   ## first fill in NoneTypes with Nans. This is a specific case for WTA
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


def decompose_variance_half(data):
    """ Decomposes variance of group, subject, and measurement noise. This is an upgraded version to handle subject-specific scaling.
    Args:
        data (ndarray (n_sub, n_rep, n_A, n_B)): the data to decompose, at least 2 for each dimension
    Returns:
        vg (ndarray (n_sub,)): group variance scaled for each subject
        vs (ndarray (n_sub,)): subject variance scaled for each subject
        vm (ndarray (n_sun,)): measurement noise variance scaled for each subject
    """

    n_sub, n_rep, n_A, n_B = data.shape
    n_features = n_A * n_B
    data = data.reshape((n_sub, n_rep, n_features))    # Shape: (n_sub, n_rep, n_features)

    product_matrices = np.einsum('srf,tkf->stkr', data, data) / n_features  # Shape: (n_sub, n_sub, n_rep, n_rep)

    # Masks
    mask_self_sub = np.eye(n_sub, dtype=bool)[:, :, None, None] # Shape: (n_sub, n_sub, 1, 1)
    mask_self_rep = np.eye(n_rep, dtype=bool)[None, None, :, :] # Shape: (1, 1, n_rep, n_rep)
    
    # Cross-subject (type 1)
    # Remove self-pairs by masking
    type_1 = np.where(mask_self_sub, 0, product_matrices)   # Set self-pairs to 0
    # Mean over repetitions
    SS_1 = np.nansum(type_1, axis=(2, 3)) / (n_rep**2)  # Shape: (n_sub, n_sub)

    # Within-subject, diff reps (type 2)
    # Remove other-pairs and self-reps by masking
    type_2 = np.where(mask_self_sub, product_matrices, 0)   # Set other-pairs to 0
    type_2 = np.where(mask_self_rep, 0, type_2) # Set self-reps to 0
    # Mean over repetitions
    SS_2 = np.diagonal(np.nansum(type_2, axis=(2,3)) / (n_rep**2-n_rep), axis1=0, axis2=1)    # Shape: (n_sub)

    # Within-subject, same reps (type 3)
    type_3 = np.where(mask_self_sub, product_matrices, 0)   # Set other-pairs to 0
    type_3 = np.where(mask_self_rep, type_3, 0) # Set other-reps to 0
    # Mean over repetitions
    SS_3 = np.diagonal(np.nansum(type_3, axis=(2,3)) / (n_rep), axis1=0, axis2=1)   # Shape: (n_sub)

    vm = SS_3 - SS_2
    vg = np.nansum(np.sqrt(SS_2[:, None] / SS_2) * SS_1, axis=1) / (n_sub-1)    # Shape: (n_sub)
    vs = SS_2 - vg

    return vg, vs, vm


def decompose_variance_scaled_from_SS(
    covariance_matrix: np.ndarray,
    dataset_vec: np.ndarray,
    sub_vec: np.ndarray,
    part_vec: np.ndarray,
    single_scaling: bool = False
) -> pd.DataFrame:
    """
    Decomposes variance components from a covariance matrix.
    Args:
        covariance_matrix (np.ndarray): A square covariance matrix.
        dataset_vec (np.ndarray): A vector containing dataset names for each row/column of the covariance matrix.
        sub_vec (np.ndarray): A vector containing subject IDs for each row/column of the covariance matrix.
        part_vec (np.ndarray): A vector containing partition IDs for each row/column of the covariance matrix.
        single_scaling (bool): If True, assumes a single scale factor for all subjects. Defaults to False.
    Returns:
        Q_df (pandas.DataFrame): DataFrame containing variance components:
            - train_dataset: Dataset names.
            - subj_id: Subject IDs.
            - sc: Scale factors for each subject.
            - v_u: Universal variance component.
            - v_d: Dataset variance component (dataset-specific).
            - v_s: Subject variance component (dataset-specific).
            - v_m: Measurement noise variance component (subject-specific).
    """

    N_SS = covariance_matrix.shape[0]

    # Identify unique subjects, datasets, and partitions
    subjects = [(dataset_vec[i], sub_vec[i]) for i in range(N_SS)]
    unique_subjects = list(dict.fromkeys(subjects))
    N_subj = len(unique_subjects)

    unique_datasets = list(dict.fromkeys(dataset_vec))
    N_datasets = len(unique_datasets)

    N_part = len(np.unique(part_vec))

    # ------------------------------
    # ------- Ckeck inputs ---------
    # ------------------------------
    if covariance_matrix.size == 0:
        raise ValueError("covariance_matrix cannot be empty.")

    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("The covariance_matrix must be a square 2D array.")

    if len(dataset_vec) != N_SS or len(sub_vec) != N_SS or len(part_vec) != N_SS:
        raise ValueError("Input vectors (dataset_vec, sub_vec, part_vec) must have the same length as the covariance matrix dimensions.")

    if N_part == 1:
        print(
            "The number of unique parts is 1. Subject variance (v_s) and measurement noise variance (v_m) cannot be distinguished. "
            "Returning v_i as v_s + v_m."
        )

    if N_datasets == 1:
        print(
            "The number of unique datasets is 1. Universal Variance (v_u) cannot be estimated. "
            "Returning v_g as v_u + v_d."
        )


    # Map (dataset, sub_id) to index
    subject_map = {sid: idx for idx, sid in enumerate(unique_subjects)}

    # ---------------------------------------
    # ----- Compute pairs and bad pairs -----
    # ---------------------------------------
    pairs_1 = []
    pairs_2 = []
    pairs_3 = []
    pairs_4 = []
    bad_pair_1 = 0
    bad_pair_2 = 0
    bad_pair_3 = 0
    bad_pair_4 = 0
    for i in range(N_SS):
        for k in range(i, N_SS):
            # cross-dataset pairs
            if dataset_vec[i] != dataset_vec[k]:
                if covariance_matrix[i, k] <= 0:
                    bad_pair_1 += 1
                    continue
                pairs_1.append((i, k))

            # same-dataset
            else:
                # cross-subject pairs
                if (sub_vec[i] != sub_vec[k]):
                    if covariance_matrix[i, k] <= 0:
                        bad_pair_2 += 1
                        continue
                    pairs_2.append((i, k))

                # same-subject
                else:
                    # cross-partition pairs
                    if (part_vec[i] != part_vec[k]):
                        if covariance_matrix[i, k] <= 0:
                            bad_pair_3 += 1
                            continue
                        pairs_3.append((i, k))

                    # same-partition pairs
                    else:
                        if covariance_matrix[i, k] <= 0:
                            bad_pair_4 += 1
                            continue
                        pairs_4.append((i, k))

    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)
    pairs_3 = np.array(pairs_3)
    pairs_4 = np.array(pairs_4)
    M_1 = len(pairs_1)
    M_2 = len(pairs_2)
    M_3 = len(pairs_3)
    M_4 = len(pairs_4)
    M = M_1 + M_2 + M_3 + M_4

    if N_datasets != 1:
        print(f"Bad pairs (cross-dataset): {bad_pair_1 / (M_1 + bad_pair_1) * 100:.2f}%")
    print(f"Bad pairs (cross-subject): {bad_pair_2 / (M_2 + bad_pair_2) * 100:.2f}%")
    if N_part != 1:
        print(f"Bad pairs (cross-partition): {bad_pair_3 / (M_3 + bad_pair_3) * 100:.2f}%")
    print(f"Bad pairs (same-partition): {bad_pair_4 / (M_4 + bad_pair_4) * 100:.2f}%")


    # -----------------------------------------------
    # ----- Construct A and y for least squares -----
    # -----------------------------------------------
    if single_scaling:
       N_scale = 1
    else:
       N_scale = N_subj
    if N_part == 1:
       A = np.zeros((M, N_scale + N_datasets + N_subj))
    else:
       A = np.zeros((M, N_scale + N_datasets + N_datasets + N_subj))
    y = np.zeros(M)

    # cross-dataset pairs
    for m, (i, k) in enumerate(pairs_1):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        s_k = subject_map[(dataset_vec[k], sub_vec[k])] if not single_scaling else 0
        # Set 1s for s_i, s_k, v_u
        A[m, s_i] += 1
        A[m, s_k] += 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])

    # same-dataset, cross-subject pairs
    for m, (i, k) in enumerate(pairs_2, start=M_1):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        s_k = subject_map[(dataset_vec[k], sub_vec[k])] if not single_scaling else 0
        # Set 1s for s_i, s_k
        A[m, s_i] += 1
        A[m, s_k] += 1
        # Set 1s for v_u + v_d
        d = unique_datasets.index(dataset_vec[i])
        A[m, N_scale+d] = 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])

    # same-dataset, same-subject, cross-partition pairs
    for m, (i, k) in enumerate(pairs_3, start=M_1 + M_2):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        # Set 1s for s_i, s_k
        A[m, s_i] = 2
        # Set 1s for v_u + v_d + v_s
        d = unique_datasets.index(dataset_vec[i])
        A[m, N_scale+N_datasets+d] = 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])

    # same-dataset, same-subject, same-partition pairs
    for m, (i, k) in enumerate(pairs_4, start=M_1 + M_2 + M_3):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        # Set 1s for s_i, s_k
        A[m, s_i] = 2
        # Set 1s for v_u + v_d + v_s + v_m
        A[m, -(N_subj-subject_map[(dataset_vec[i], sub_vec[i])])] = 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])


    # -------------------------------------------------------
    # ----- Solve least squares and extract components ------
    # -------------------------------------------------------
    x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # Extract parameters
    sc = np.exp(x[:N_scale])                                          # scales
    type_1 = np.exp(x[N_scale:N_scale+N_datasets])                    # V_u + V_d
    type_2 = np.exp(x[N_scale+N_datasets:N_scale+N_part*N_datasets])  # V_u + V_d + V_s
    type_3 = np.exp(x[N_scale+N_part*N_datasets:])                    # V_u + V_d + V_s + V_m

    if N_datasets == 1:
        v_g = type_1
    else:
        v_u = 1
        v_d = type_1 - v_u
    if N_part == 1:
        v_i = type_3 - type_1[[unique_datasets.index(ds) for ds,_ in unique_subjects]]
    else:
        v_s = type_2 - type_1
        v_m = type_3 - type_2[[unique_datasets.index(ds) for ds,_ in unique_subjects]]


    # ----------------------------------
    # -------- Create DataFrame --------
    # ----------------------------------
    train_dataset = [sid[0] for sid in subject_map.keys()]
    subj_id = [sid[1] for sid in subject_map.keys()]

    if single_scaling:
        sc = [sc[0]] * len(train_dataset)

    data_dict = {
        'train_dataset': train_dataset,
        'subj_id': subj_id,
        'sc': sc
    }
    if N_datasets == 1:
        data_dict['v_g'] = v_g[[unique_datasets.index(ds) for ds, _ in unique_subjects]]
    else:
        data_dict['v_u'] = [v_u] * len(train_dataset)
        data_dict['v_d'] = v_d[[unique_datasets.index(ds) for ds, _ in unique_subjects]]
    if N_part == 1:
        data_dict['v_i'] = v_i
    else:
        data_dict['v_s'] = v_s[[unique_datasets.index(ds) for ds, _ in unique_subjects]]
        data_dict['v_m'] = v_m

    Q_df = pd.DataFrame(data_dict)

    return Q_df


def decompose_variance(data, vm_hat, model_type=None):
   """ Decomposes variance of group, subject, and measurement noise.
      This is an upgraded version to handle subject-specific scaling.
      With the vm_hat already estimated, there is no need for different observations.
   Args:
      data (ndarray (n_sub, n_A, n_B)): the data to decompose
      vm_hat (ndarray (n_sub)): estimated variance of measurement noise of subjects
      model_type (str): either None or 'loo':
         if 'loo': the output will be stretched by subject size
   Returns:
      vg (ndarray (n_sub,)): group variance scaled for each subject
      vs (ndarray (n_sub,)): subject variance scaled for each subject
      vm (ndarray (n_sun,)): measurement noise variance scaled for each subject
      if model_type is 'loo': outputs will be (n_sub, n_sub-1) shape
   """

   n_sub, n_A, n_B = data.shape
   n_features = n_A * n_B
   data = data.reshape((n_sub, n_features))

   product_matrices = np.einsum('sf,kf->sk', data, data) / n_features   # Shape: (n_sub, n_sub)

   if model_type == 'loo':
      n_sub_loo = n_sub - 1
      vg = np.zeros((n_sub, n_sub - 1))
      vs = np.zeros((n_sub, n_sub - 1))
      for s in range(n_sub):
         product_matrices_loo = np.delete(np.delete(product_matrices, s, axis=0), s, axis=1)

         # Masks
         mask_self_sub = np.eye(n_sub_loo, dtype=bool) # Shape: (n_sub, n_sub)
         
         # Cross-subject (type 1)
         SS_1 = np.where(mask_self_sub, 0, product_matrices_loo)   # Set self-pairs to 0

         # Within-subject, same reps (type 3)
         SS_3 = np.diag(product_matrices_loo)   # Set other-pairs to 0

         SS_2 = SS_3 - np.delete(vm_hat, s, axis=0)

         vg[s] = np.nansum(np.sqrt(SS_2[:, None] / SS_2) * SS_1, axis=1) / (n_sub_loo-1)    # Shape: (n_sub)
         vs[s] = SS_2 - vg[s]
   elif model_type is None:
      # Masks
      mask_self_sub = np.eye(n_sub, dtype=bool) # Shape: (n_sub, n_sub)
      
      # Cross-subject (type 1)
      SS_1 = np.where(mask_self_sub, 0, product_matrices)   # Set self-pairs to 0

      # Within-subject, same reps (type 3)
      SS_3 = np.diag(product_matrices)   # Set other-pairs to 0

      SS_2 = SS_3 - vm_hat

      vg = np.nansum(np.sqrt(SS_2[:, None] / SS_2) * SS_1, axis=1) / (n_sub-1)    # Shape: (n_sub)
      vs = SS_2 - vg
   else:
      raise ValueError("model_type should be 'loo' or not given")
   return vg, vs

