"""Main module for training and evaluating connectivity models.

   @authors: Ladan Shahshahani, Maedbh King, Jörn Diedrichsen
"""
# TODO: implement the weighting option  
# TODO: Handling crossing sessions (half or ses) - right now it only uses half

from audioop import cross
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import json
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import nibabel as nb
import Functional_Fusion as ff
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.matrix as fm
import cortico_cereb_connectivity.globals as gl

import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.evaluation as ev

import warnings

# warnings.filterwarnings("ignore")

def get_train_config(
                     train_dataset = "MDTB", 
                     train_ses = "ses-s1", 
                     method = "L2regression",
                     log_alpha = 8, 
                     cerebellum = "SUIT3",
                     cortex = "fs32k",
                     parcellation = "Icosahedron1002",
                     type = "CondHalf",
                     cv_fold = 4,
                     crossed = "half", # or None 
                     validate_model = True,
                     ):
   """get_train_config
   Function to create a config dictionary containing the info for the training

   Args:
       dataset (str, optional): _description_. Defaults to "MDTB".
       ses_id (str, optional): _description_. Defaults to "ses-s1".
       method (str, optional): _description_. Defaults to "L2regression".
       log_alpha (int, optional): _description_. Defaults to 8.
       cerebellum (str, optional): _description_. Defaults to "SUIT3".
       cortex (str, optional): _description_. Defaults to "fs32k".
       parcellation (str, optional): _description_. Defaults to "Icosahedron-1002_Sym.32k".
       mode (str, optional): _description_. Defaults to "crossed".
       type (str, optional): _description_. Defaults to "CondHalf".
       cv_fold (int, optional): _description_. Defaults to 4.
       cross_over (str, optional): _another option: or dataset name if you want to integrate over sessions of the dataset_. Defaults to "half".

   Returns:
       _type_: _description_
   """
   train_config = {}
   train_config['train_dataset'] = train_dataset # name of the dataset to be used in 	   	   
   train_config['train_ses'] = train_ses   
   train_config['method'] = method   # method used in modelling (see model.py)
   train_config['logalpha'] = log_alpha # alpha will be np.exp(log_alpha)
   train_config['cerebellum'] = cerebellum
   train_config['cortex'] = cortex
   train_config['parcellation'] = parcellation
   train_config['crossed'] = crossed
   # train_config['weighting'] = weighting
   train_config["validate_model"] = validate_model
   train_config["type"] = type 
   train_config["cv_fold"] = cv_fold, #TO IMPLEMENT: "ses_id", "run", "dataset", "tasks"
   train_config['subj_list'] = "all"
      
   # get label images for left and right hemisphere
   train_config['label_img'] = []
   for hemi in ['L', 'R']:
      train_config['label_img'].append(gl.atlas_dir + f'/tpl-{train_config["cortex"]}' + f'/{train_config["parcellation"]}.{hemi}.label.gii')

   return train_config

def get_eval_config(eval_dataset = 'MDTB',
            eval_ses = 'ses-s2', 
            cerebellum = 'SUIT3',
            cortex = "fs32k",
            parcellation = "Icosahedron1002",
            crossed = "half", # or None
            type = "CondHalf",
            splitby = None):
   """
   create a config file for evaluation
   """
   eval_config = {}
   eval_config['eval_dataset'] = eval_dataset
   eval_config['eval_ses'] = eval_ses
   eval_config['cerebellum'] = cerebellum
   eval_config['cortex'] = cortex
   eval_config['parcellation'] = parcellation
   eval_config['crossed'] = crossed
   eval_config["splitby"] = splitby
   eval_config["type"] = type 
   eval_config["cv_fold"] = None, #TO IMPLEMENT: "sess", "run" (None is "tasks")
   eval_config['subj_list'] = "all"

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
    rmse_train = mean_squared_error(Y, Y_pred, squared=False)
    R_train, _ = ev.calculate_R(Y, Y_pred)
    return rmse_train, R_train

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
    # get cv rmse and R
    rmse_cv_all = np.sqrt(cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv=cv_fold) * -1)

    # TO DO: implement train/validate splits for "sess", "run"
    r_cv_all = cross_val_score(model, X, Y, scoring=ev.calculate_R_cv, cv=cv_fold)

    return np.nanmean(rmse_cv_all), np.nanmean(r_cv_all)

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

    # Noise ceiling for cerebellum (squared)
    (
        data["noise_X_R"],
        data["noise_X_R_vox"],
        data["noise_X_R2"],
        data["noise_X_R2_vox"],
    ) = ev.calculate_reliability(Y=Y_pred, dataframe = info)

    # calculate noise ceiling
    data["noiseceiling_Y_R_vox"] = np.sqrt(data["noise_Y_R_vox"])
    data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"] * np.sqrt(data["noise_X_R_vox"]))

    return data

def train_model(config):
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
   # get dataset class 
   dataset = fdata.get_dataset_class(gl.base_dir, 
                                    dataset=config["train_dataset"])
   
   ## loop over sessions chosen through train_id and concatenate data
   info_list = []
   
   if not isinstance(config['subj_list'],(list,pd.Series,np.ndarray)):
      if config["subj_list"]=='all':
         T = dataset.get_participants()
         config["subj_list"] = T.participant_id

   # initialize training dict
   conn_model_list = []
   train_info = pd.DataFrame()

   # Generate model name and create directory
   mname = f"{config['train_dataset']}_{config['train_ses']}_{config['parcellation']}_{config['method']}"
   save_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',
                                  mname)
   # check if the directory exists
   try:
      os.makedirs(save_path)
   except OSError:
      pass

   # Loop over subjects 
   for i, sub in enumerate(config["subj_list"]):
      YY, info, _ = fdata.get_dataset(gl.base_dir,
                                    config["train_dataset"],
                                    atlas=config["cerebellum"],
                                    sess=config["train_ses"],
                                    type=config["type"],
                                    subj=str(sub))
      XX, info, _ = fdata.get_dataset(gl.base_dir,
                                    config["train_dataset"],
                                    atlas=config["cortex"],
                                    sess=config["train_ses"],
                                    type=config["type"],
                                    subj=str(sub))
      # Average the cortical data over pacels   
      X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
      # get the vector containing tessel labels
      X_atlas.get_parcel(config['label_img'], unite_struct = False)
      # get the mean across tessels for cortical data
      XX, labels = fdata.agg_parcels(XX, X_atlas.label_vector,fcn=np.nanmean)

      # Remove Nans 
      Y = np.nan_to_num(YY[0,:,:]) 
      X = np.nan_to_num(XX[0,:,:]) 

      # cross the sessions
      if config["crossed"] is not None:
         if config["crossed"]=='half':
            Y = np.r_[Y[info["half"] == 2, :], Y[info["half"] == 1, :]]

      for la in config["logalpha"]:
      # loop over subjects and train models
         print(f'- Train {sub} {config["method"]} logalpha {la}')

         # Generate new model
         alpha = np.exp(la) # get alpha
         conn_model = getattr(model, config["method"])(alpha)

         # Fit model, get train and validate metrics
         conn_model.fit(X, Y)
         conn_model.rmse_train, conn_model.R_train = train_metrics(conn_model, X, Y)
         conn_model_list.append(conn_model)

         mname_spec = f"{mname}_A{la}_{sub}"
         # collect train metrics (rmse and R)
         model_info = {
                        "subj_id": sub,
                        "mname": mname_spec,
                        "rmse_train": conn_model.rmse_train,
                        "R_train": conn_model.R_train,
                        "num_regions": X.shape[1],
                        "logalpha": la
                        }

         # run cross validation and collect metrics (rmse and R)
         if config['validate_model']:
            conn_model.rmse_cv, conn_model.R_cv = validate_metrics(conn_model, X, Y, config["cv_fold"][0])
            model_info.update({"rmse_cv": conn_model.rmse_cv,
                            "R_cv": conn_model.R_cv
                           })

         # Copy over all scalars or strings from config to eval dict:
         for key, value in config.items():
            if not isinstance(value, (list, dict,pd.Series,np.ndarray)):
               model_info.update({key: value})
         # Save the individuals info files 
         dd.io.save(save_path + "/" + mname_spec + ".h5",
                     conn_model, compression=None)
         with open(save_path + "/" + mname_spec + ".json", "w") as fp:
            json.dump(model_info, fp, indent=4)

         train_info = pd.concat([train_info,pd.DataFrame(model_info)],ignore_index= True)
   train_info.to_csv(save_path + "/" + mname + ".tsv",sep='\t')
   return config, conn_model_list, train_info

def eval_model(model_dirs,model_names,config, avg = True):
   """
   evaluate group model on a specific dataset and session
   Args:
      model_dirs (list)  - list of model directories 
      model_names (list) - list of full model names (without .h5) to evaluate 
      config (dict)      - dictionary with evaluation parameters
   """
   # initialize eval dictionary
   eval_df = pd.DataFrame()
   eval_voxels = defaultdict(list)

   # get dataset class
   dataset = fdata.get_dataset_class(gl.base_dir, 
                                    dataset=config["eval_dataset"])
   
   # get list of subjects
   if config["subj_list"]=='all':
      T = dataset.get_participants()
      config["subj_list"] = T.participant_id

   # Load all the models to evaluate:
   fitted_model = []
   train_info = []
   if avg:
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         fname = model_path + f"/{m}_avg.h5"
         json_name = model_path + f"/{m}_avg.json"
         fitted_model.append(dd.io.load(fname))
      
         # Load json file
         with open(json_name) as json_file:
            train_info.append(json.load(json_file))



   # loop over subjects
   for i, sub in enumerate(config["subj_list"]):
      print(f'- Evaluate {sub}')

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

      # cross the sessions
      if config["crossed"] is not None:
         if config["crossed"]=='half':
            Y = np.r_[Y[info["half"] == 2, :], Y[info["half"] == 1, :]]

      # If not average, load the model for each subject
      if not avg:
         fitted_model = []
         train_info = []
         for d,m in zip(model_dirs,model_names):
            model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
            fname = model_path + f"/{m}_{sub}.h5"
            json_name = model_path + f"/{m}_{sub}.json"
            fitted_model.append(dd.io.load(fname))
      
            with open(json_name) as json_file:
               train_info.append(json.load(json_file))
      
      # Loop over models
      for j, (fm, tinfo) in enumerate(zip(fitted_model, train_info)):
         # Get model predictions
         Y_pred = fm.predict(X)
      
         # 
         eval_sub = {"eval_subj": sub,
                  "num_regions": X.shape[1]}

         # Copy over all scalars or strings to eval_all dataframe:
         for key, value in tinfo.items():
            if not isinstance(value,(list,pd.Series,np.ndarray)):
               eval_sub.update({key: value})
         for key, value in config.items():
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

         # don't save voxel data to summary
         eval_df = pd.concat([eval_df,pd.DataFrame(eval_sub)],ignore_index= True)

   return eval_df, eval_voxels

def calc_avrg_model(train_dataset,
                    mname_base,
                    mname_ext,
                    cerebellum='SUIT3',
                    parameters=['coef_','scale_']):
   """Get the fitted models from all the subjects in the training data set 
      and create group-averaged model 
   Args:
       train_dataset (str): _description_
       mname_base (str): Directory name for mode (MDTB_all_Icosahedron1002_L2regression) 
       mname_ext (str): Extension of name - typically logalpha
       (A0)
       parameters (list): List of parameters to average
   """

   # get the dataset class the model was trained on
   # To get the list of subjects 
   tdata = fdata.get_dataset_class(gl.base_dir, dataset=train_dataset)  
   T = tdata.get_participants()
   subject_list = T.participant_id
   
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
      # load the model
      fname = model_path + f"/{mname_base}_{mname_ext}_{sub}.h5"
      info_name = model_path + f"/{mname_base}_{mname_ext}_{sub}.json"
      fitted_model = dd.io.load(fname)
      
      # load the json file
      with open(info_name) as json_file:
         info = json.load(json_file)
         df = pd.concat([df,pd.DataFrame(info,index=[0])],ignore_index=True)
         
      for p in parameters:
         param_lists[p].append(getattr(fitted_model,p))

   avrg_model = fitted_model
   for p in parameters:
      P = np.stack(param_lists[p],axis=0)
      setattr(avrg_model,p,P.mean(axis=0))
   
   dd.io.save(model_path + f"/{mname_base}_{mname_ext}_avg.h5",
      avrg_model, compression=None)
   # Assemble the summary
   dict = {'train_dataset': df.train_dataset[0],
           'train_ses': df.train_ses[0],
           'train_type': df.type[0],
           'cerebellum': df.cerebellum[0],
           'cortex': df.cortex[0],
           'method': df.method[0],
           'logalpha': float(df.logalpha[0]),
           'R_train': df.R_train.mean(),
           'rmse_train': df.rmse_train.mean(),
           'R_cv': df.R_cv.mean(),
           'rmse_cv': df.rmse_cv.mean(),
           }
   # save dict as json
   with open(model_path + f"/{mname_base}_{mname_ext}_avg.json", 'w') as fp:
      json.dump(dict, fp, indent=4)
