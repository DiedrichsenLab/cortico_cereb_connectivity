"""Main module for training and evaluating connectivity models.

   @authors: Ladan Shahshahani, Maedbh King, Jörn Diedrichsen
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sys
# sys.path.append('../Functional_Fusion') 
sys.path.append('..')
import nibabel as nb
import Functional_Fusion as ff
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.matrix as fm
import prepare_data as prep

import model as model
import evaluation as ev

import warnings

warnings.filterwarnings("ignore")

# OPEN ISSUES: 
# 2. Handling crossing sessions (half or ses) - right now it only uses half

# get training config dictionary
def get_train_config(
                     dataset = "MDTB", 
                     ses_id = "ses-s1", 
                     method = "L2regression",
                     log_alpha = 8, 
                     cerebellum = "SUIT3",
                     cortex = "fs32k",
                     parcellation = "Icosahedron-1002_Sym.32k",
                     mode = "crossed",  
                     type = "CondHalf",
                     cv_fold = 4,
                     # weighting = True, 
                     validate_model = True,
                     ):
   """
   create a config file for training

   """
   train_config = {}
   train_config['dataset'] = dataset # name of the dataset to be used in training models
   train_config['ses_id'] = ses_id   
   train_config['method'] = method   # method used in modelling (see model.py)
   train_config['log_alpha'] = log_alpha # alpha will be np.exp(log_alpha)
   train_config['cerebellum'] = cerebellum
   train_config['cortex'] = cortex
   train_config['parcellation'] = parcellation
   train_config['mode'] = mode 
   # train_config['weighting'] = weighting
   train_config["validate_model"] = validate_model
   train_config["type"] = type 
   train_config["cv_fold"] = cv_fold, #TO IMPLEMENT: "ses_id", "run", "dataset", "tasks"
   train_config['name'] = f"{parcellation}_{ses_id}_{method}_logalpha_{log_alpha}"
   
   # get the cortical parcellation you want to use in modelling
   train_config['label_img'] = []
   for hemi in ['L', 'R']:
      train_config['label_img'].append(prep.atlas_dir + f'/tpl-{train_config["cortex"]}' + f'/{train_config["parcellation"]}.{hemi}.label.gii')
   

   return train_config

# get evaluation config dictionary
def get_eval_config(
   dataset = "MDTB", 
   train_id = "ses-s1",
   eval_id = "ses-s2", 
   method = "L2Regression",
   log_alpha = 8, 
   cerebellum = "SUIT3",
   cortex = "fs32k",
   parcellation = "Icosahedron-1002_Sym.32k",
   mode = "crossed",  
   type = "CondHalf",
   # weighting = True, 
   splitby = None,
):
   """
   create a config file for evaluation

   """
   eval_config = {}
   eval_config['dataset'] = dataset
   eval_config['train_id'] = train_id
   eval_config['eval_id'] = eval_id
   eval_config['method'] = method
   eval_config['log_alpha'] = log_alpha #
   eval_config['cerebellum'] = cerebellum
   eval_config['cortex'] = cortex
   eval_config['parcellation'] = parcellation
   eval_config['mode'] = mode 
   # train_config['weighting'] = weighting
   eval_config["splitby"] = splitby
   eval_config["type"] = type 
   eval_config["cv_fold"] = None, #TO IMPLEMENT: "sess", "run" (None is "tasks")
   # eval_config['name'] = f"{parcellation}_{eval_id}_{method}_logalpha_{log_alpha}"
   eval_config['name'] = f"{parcellation}_{train_id}_{method}_logalpha_{log_alpha}"

   # get label images for left and right hemisphere
   eval_config['label_img'] = []
   for hemi in ['L', 'R']:
      eval_config['label_img'].append(prep.atlas_dir + f'/tpl-{eval_config["cortex"]}' + f'/{eval_config["parcellation"]}.{hemi}.label.gii')


   return eval_config

# get train metrics
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

# cross validating model
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

# calculate and return evaluation metrics
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
    data["R2"], data["R2_vox"] = ev.calculate_R2(Y=Y, Y_pred=Y_pred)

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

# training model
def train_model(config, group = True, save_tensor = False):
   """
   training a specific model based on the config file created
   Args: 
      config (dict)      - dictionary with configuration parameters
      group (bool)       - fit the model using "group" data (data averaged over subjects)
      save_tensor (bool) - create and save the tensor (contains all the subjects data)
   Returns:
      conn_model (model object) - trained model  
      train_df (pd.DataFrame)   - datafarme containing training information
   """
   # save tensors?
   if save_tensor:
      # get data tensor for SUIT3
        prep.save_data_tensor(dataset = config["dataset"],
                        atlas=config['cerebellum'],
                        ses_id=config['ses_id'],
                        type=config['type'], 
                        )

        # get data tensor for fs32k
        prep.save_data_tensor(dataset = config["dataset"],
                        atlas=config['cortex'],
                        ses_id=config['ses_id'],
                        type=config['type'], 
                        )
   
   # get dataset class 
   Data = fdata.get_dataset_class(prep.base_dir, dataset=config["dataset"])
   # get info
   info = Data.get_info(config['ses_id'],config['type'])
   # load data tensor for cortex and cerebellum atlases
   Y_file = prep.conn_dir + config['dataset'] + f"/{config['dataset']}_{config['cerebellum']}_{config['ses_id']}_{config['type']}.npy"
   Y_tensor = np.load(Y_file)
   # load data tensor for fs32k
   X_file = prep.conn_dir + config['dataset'] + f"/{config['dataset']}_{config['cortex']}_{config['ses_id']}_{config['type']}.npy"
   X_tensor = np.load(X_file)


   # get cortical atlas object(will be used to aggregate data within tessellation)
   X_atlas, _ = at.get_atlas(config['cortex'],prep.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(config['label_img'], unite_struct = False)

   # get participants for the dataset
   if group: # will just use the group averaged dataset
      subject_list = ["group"]
   else:
      T = Data.get_participants()
      subject_list = T.participant_id

   # initialize training dict
   train_dict = defaultdict(list)

   # loop over subjects and train models
   for i, sub in enumerate(subject_list):
      print(f'- Train {sub} {config["method"]} log alpha {config["log_alpha"]}')
      # get the slice of tensor corresponding to the current subject
      X = X_tensor[i, :, :]
      Y = Y_tensor[i, :, :]
    
      # get the mean across tessels for cortical data
      X = fdata.agg_parcels(X, X_atlas.label_vector,fcn=np.nanmean)

      Y = np.nan_to_num(Y) # replace nans first 
      X = np.nan_to_num(X) # replace nans first
      # Generate new model
      alpha = np.exp(config["log_alpha"]) # get alpha
      conn_model = getattr(model, config["method"])(alpha)

      # cross the sessions
      if config["mode"] == "crossed":
         Y = np.r_[Y[info.half == 2, :], Y[info.half == 1, :]]

      # Fit model, get train and validate metrics
      conn_model.fit(X, Y)
      conn_model.rmse_train, conn_model.R_train = train_metrics(conn_model, X, Y)

      # collect train metrics (rmse and R)
      model_info = {
                        "subj_id": sub,
                        "rmse_train": conn_model.rmse_train,
                        "R_train": conn_model.R_train,
                        "num_regions": X.shape[1]
                        }

      # run cross validation and collect metrics (rmse and R)
      if config['validate_model']:
         conn_model.rmse_cv, conn_model.R_cv = validate_metrics(conn_model, X, Y, config["cv_fold"][0])
         model_info.update({"rmse_cv": conn_model.rmse_cv,
                            "R_cv": conn_model.R_cv
                           })

      # Copy over all scalars or strings from config to eval dict:
      for key, value in config.items():
         if not isinstance(value, (list, dict)):
               model_info.update({key: value})

      for k, v in model_info.items():
            train_dict[k].append(v)

      # get directory to save the trained model
      save_path = os.path.join(prep.conn_dir,config['dataset'],'train', config['name'])
      # check if the directory exists
      try:
         os.makedirs(save_path)
      except OSError:
         pass

      fname = save_path + f"/{config['method']}_alpha{config['log_alpha']}_{sub}.h5"
      dd.io.save(fname, conn_model, compression=None)
   
   return conn_model, pd.DataFrame.from_dict(train_dict)

# evaluating model
def eval_model(config, group = True, save_tensor = False, save = False):
   """
   """

   # initialize eval dictionary
   eval_dict = defaultdict(list)
   eval_voxels = defaultdict(list)

   # save tensors?
   if save_tensor:
      # get data tensor for SUIT3
        prep.get_data_tensor(dataset = config["dataset"],
                        atlas='SUIT3',
                        ses_id=config['eval_id'],
                        type=config['type'], 
                        save = save_tensor)

        # get data tensor for fs32k
        prep.get_data_tensor(dataset = config["dataset"],
                        atlas='fs32k',
                        ses_id=config['eval_id'],
                        type=config['type'], 
                        save= save_tensor)

   # get dataset class 
   Data = fdata.get_dataset_class(prep.base_dir, dataset=config["dataset"])
   # get info
   info = Data.get_info(config['eval_id'],config['type'])
   # load data tensor for cortex and cerebellum atlases
   Y_file = prep.conn_dir + config['dataset'] + f"/{config['dataset']}_{config['cerebellum']}_{config['eval_id']}_{config['type']}.npy"
   Y_tensor = np.load(Y_file)

   # load data tensor for fs32k
   X_file = prep.conn_dir + config['dataset']+ f"/{config['dataset']}_{config['cortex']}_{config['eval_id']}_{config['type']}.npy"
   X_tensor = np.load(X_file)

   # get cortical atlas object(will be used to aggregate data within tessellation)
   X_atlas, _ = at.get_atlas(config['cortex'],prep.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(config['label_img'], unite_struct = False)


   # get participants for the dataset
   if group:
      subject_list = ["group"]
   else:
      T = Data.get_participants()
      subject_list = T.participant_id

   # loop over subjects
   for i, sub in enumerate(subject_list):
      print(f'- Evaluate {sub} {config["method"]} log alpha {config["log_alpha"]}')

      # get the slice of tensor corresponding to the current subject
      X = X_tensor[i, :, :]
      Y = Y_tensor[i, :, :]
    
      # get the mean across tessels for cortical data
      X = fdata.agg_parcels(X, X_atlas.label_vector,fcn=np.nanmean)

      Y = np.nan_to_num(Y) # replace nans first 
      X = np.nan_to_num(X) # replace nans first

      # get model name
      model_path = prep.conn_dir + config['dataset']+ f'/train/{config["name"]}'
      # check if the directory exists
      try:
         fname = model_path + f"/{config['method']}_alpha{config['log_alpha']}_{sub}.h5"
         fitted_model = dd.io.load(fname)
      except:
         print(f"Run the model first")
         
      # Get model predictions
      Y_pred = fitted_model.predict(X)
      if config["mode"] == "crossed":
         Y_pred = np.r_[Y_pred[info.half == 2, :], Y_pred[info.half == 1, :]]
      
      # get rmse
      rmse = mean_squared_error(Y, Y_pred, squared=False)
      eval_sub = {"rmse_eval": rmse,
                  "subj_id": sub,
                  "num_regions": X.shape[1]}

      # Copy over all scalars or strings to eval_all dataframe:
      for key, value in config.items():
         if type(value) is not list:
               eval_sub.update({key: value})

      # add evaluation (summary)
      evals = eval_metrics(Y=Y, Y_pred=Y_pred, info = info)
      eval_sub.update(evals)

      # add evaluation (voxels)
      for k, v in eval_sub.items():
            if "vox" in k:
               eval_voxels[k].append(v)

      # don't save voxel data to summary
      eval_sub = {k: v for k, v in eval_sub.items() if "vox" not in k}

      # append data for each subj
      for k, v in eval_sub.items():
         eval_dict[k].append(v)

      # get eval name
      eval_path = prep.conn_dir + config['dataset']+ f'/{sub}/eval/'
      # Save the maps
      
   return pd.DataFrame.from_dict(eval_dict), eval_voxels