"""Main module for training and evaluating connectivity models.

   @authors: Ladan Shahshahani, Maedbh King, Jörn Diedrichsen
"""
# TODO: Change variables in the dictionary to accomodate cross dataset integration
# TODO: for each alpha, get the group average weight across training subject (maybe if group option is selected)
# TODO: For each alpha get the group average scale across training subjects (again if group option is selected)
# TODO: implement the weighting option  
# TODO: Reorganize the directory structure??
# TODO: choose a better naming for the model. Current name is too long
# TODO: Handling crossing sessions (half or ses) - right now it only uses half

from audioop import cross
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import re
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

warnings.filterwarnings("ignore")

def get_train_config(
                     train_dataset = "MDTB", 
                     ses_id = "ses-s1", 
                     method = "L2regression",
                     log_alpha = 8, 
                     cerebellum = "SUIT3",
                     cortex = "fs32k",
                     parcellation = "Icosahedron-1002_Sym.32k",
                     mode = "crossed",  
                     type = "CondHalf",
                     cv_fold = 4,
                     cross_over = "half", # or sess 
                     # weighting = True, 
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
   train_config['train_dataset'] = train_dataset # name of the dataset to be used in 	   	   train_config['train_id'] = ses_id   
   train_config['method'] = method   # method used in modelling (see model.py)
   train_config['logalpha'] = log_alpha # alpha will be np.exp(log_alpha)
   train_config['cerebellum'] = cerebellum
   train_config['cortex'] = cortex
   train_config['parcellation'] = parcellation
   train_config['mode'] = mode 
   train_config['cross_over'] = cross_over
   # train_config['weighting'] = weighting
   train_config["validate_model"] = validate_model
   train_config["type"] = type 
   train_config["cv_fold"] = cv_fold, #TO IMPLEMENT: "ses_id", "run", "dataset", "tasks"
   train_config['name'] = f"{parcellation}_{ses_id}_{method}_logalpha_{log_alpha}"
   train_config['subj_list'] = "all"
   
   
   # get the cortical parcellation you want to use in modelling
   train_config['label_img'] = []
   for hemi in ['L', 'R']:
      train_config['label_img'].append(gl.atlas_dir + f'/tpl-{train_config["cortex"]}' + f'/{train_config["parcellation"]}.{hemi}.label.gii')
   

   return train_config

def get_eval_config(
   eval_dataset = "MDTB", 
   train_dataset = "MDTB",
   train_id = "ses-s1", # or "all" if you have used the whole dataset
   eval_id = "ses-s2",  # or "all" if you have used the whole dataset
   method = "L2Regression",
   log_alpha = 8, 
   cerebellum = "SUIT3",
   cortex = "fs32k",
   parcellation = "Icosahedron-1002_Sym.32k",
   mode = "crossed",  
   cross_over = "half", # or sess
   type = "CondHalf",
   # weighting = True, 
   splitby = None,
):
   """
   create a config file for evaluation

   """
   eval_config = {}
   eval_config['train_dataset'] = train_dataset
   eval_config['eval_dataset'] = eval_dataset
   eval_config['train_id'] = train_id
   eval_config['eval_id'] = eval_id
   eval_config['method'] = method
   eval_config['log_alpha'] = log_alpha #
   eval_config['cerebellum'] = cerebellum
   eval_config['cortex'] = cortex
   eval_config['parcellation'] = parcellation
   eval_config['mode'] = mode 
   eval_config['cross_over'] = cross_over
   # train_config['weighting'] = weighting
   eval_config["splitby"] = splitby
   eval_config["type"] = type 
   eval_config["cv_fold"] = None, #TO IMPLEMENT: "sess", "run" (None is "tasks")

   # # some housekeeping
   # if cross_over == "sess":
   #    # overwrite train
   eval_config['name'] = f"{parcellation}_{train_id}_{method}_logalpha_{log_alpha}"

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


#TODO: train on individual eval on individual from the same data set cross_over to "half". 
#TODO: train on individual, get group weights, eval on indivuals from a different dataset set cross_over to "sess"

def train_model(config):
   """
   training a specific model based on the config file created
   Args: 
      config (dict)      - dictionary with configuration parameters
      group (bool)       - fit the model using "group" data (data averaged over subjects)
                           NOTE: this is different from "group-level" connectivity weights. Those can be estimated 
                           on the individual level and then averaged.
      save_tensor (bool) - create and save the tensor (contains all the subjects data)
   Returns:
      conn_model_list (list)    - list of trained models on the list of subjects
      config (dict)             - dictionary containing info for training. Can be saved as json
      train_df (pd.DataFrame)   - datafarme containing training information
   """
   # get dataset class 
   Data = fdata.get_dataset_class(gl.base_dir, dataset=config["train_dataset"])
   
   # load data tensor for cortex and cerebellum atlases
   ## loop over sessions chosen through train_id and concatenate data
   tensor_X_list = []
   tensor_Y_list = []
   info_list = []
   if config["cross_over"] == "sess": # if you want to train the model over the whole dataset
      
      # get the data over all the sessions
      for ses, ses_id in enumerate(Data.sessions):

         YY, info, _ = fdata.get_dataset(gl.base_dir,config["train_dataset"],atlas="SUIT3",sess=ses_id,type=config["type"], info_only=False)
         XX, info, _ = fdata.get_dataset(gl.base_dir,config["train_dataset"],atlas="fs32k",sess=ses_id,type=config["type"], info_only=False)
         tensor_X_list.append(XX)
         tensor_Y_list.append(YY)
         # add a number for session
         info["ses_num"] = ses+1
         info_list.append(info)
   else: # you want to train the model on a specific session of the dataset
      YY, info, _ = fdata.get_dataset(gl.base_dir,config["train_dataset"],atlas="SUIT3",sess=config["train_id"],type=config["type"], info_only=False)
      XX, info, _ = fdata.get_dataset(gl.base_dir,config["train_dataset"],atlas="fs32k",sess=config["train_id"],type=config["type"], info_only=False)
      tensor_X_list.append(XX)
      tensor_Y_list.append(YY)
      info["ses_num"] = info["half"]
      info_list.append(info)
      
   # concatenate data across conditions over the selected sessions
   tensor_X = np.concatenate(tensor_X_list, axis = 1) # axis 1 is the conditions
   tensor_Y = np.concatenate(tensor_Y_list, axis = 1) # axis 1 is the conditions
   
   # concatenate the dataframes representing info
   info = pd.concat(info_list, axis = 0)
   # create a column to show unified condition numbers 
   ## this will be useful for when you want to integrate over sessions
   info["cond_num_uni"] = np.arange(1, len(info.index)+1)
   
   # get cortical atlas object(will be used to aggregate data within tessellation)
   # NOTE: model will be trained on cerebellar voxels and average within cortical tessels.
   X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(config['label_img'], unite_struct = False)

   #
   # get the mean across tessels for cortical data
   tensor_X, labels = fdata.agg_parcels(tensor_X, X_atlas.label_vector,fcn=np.nanmean)

   # get participants for the dataset
   T = Data.get_participants()
   config["subject_list"] = T.participant_id

   # initialize training dict
   train_dict = defaultdict(list)

   # loop over subjects and train models
   conn_model_list = []
   for i, sub in enumerate(config["subject_list"]):
      print(f'- Train {sub} {config["method"]} log alpha {config["logalpha"]}')
      # get the slice of tensor corresponding to the current subject
      X = tensor_X[i, :, :]
      Y = tensor_Y[i, :, :]

      # replace nans in X and Y
      Y = np.nan_to_num(Y) 
      X = np.nan_to_num(X) 
      # Generate new model
      alpha = np.exp(config["logalpha"]) # get alpha
      conn_model = getattr(model, config["method"])(alpha)

      # cross the sessions
      if config["mode"] == "crossed":
         Y = np.r_[Y[info["ses_num"] == 2, :], Y[info["ses_num"] == 1, :]]

      # Fit model, get train and validate metrics
      conn_model.fit(X, Y)
      conn_model.rmse_train, conn_model.R_train = train_metrics(conn_model, X, Y)
      conn_model_list.append(conn_model)

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
      save_path = os.path.join(gl.conn_dir,config['train_dataset'],'train', config['name'])
      # check if the directory exists
      try:
         os.makedirs(save_path)
      except OSError:
         pass

      fname = save_path + f"/{config['method']}_{config['train_id']}_logalpha{config['logalpha']}_{sub}.h5"
      dd.io.save(fname, conn_model, compression=None)
   
   return config, conn_model_list, pd.DataFrame.from_dict(train_dict)

def eval_model(config, avg = True, save = False):
   """
   evaluate models
   """

   # initialize eval dictionary
   eval_dict = defaultdict(list)
   eval_voxels = defaultdict(list)

   # get dataset class 
   Data = fdata.get_dataset_class(gl.base_dir, dataset=config["eval_dataset"])
   
   # load data tensor for cortex and cerebellum atlases
   ## loop over sessions chosen through train_id and concatenate data
   tensor_X_list = []
   tensor_Y_list = []
   info_list = []
   if config["cross_over"] == "sess": # if you want to train the model over the whole dataset
      
      # get the data over all the sessions
      for ses, ses_id in enumerate(Data.sessions):

         YY, info, _ = fdata.get_dataset(gl.base_dir,config["eval_dataset"],atlas="SUIT3",sess=ses_id,type=config["type"], info_only=False)
         XX, info, _ = fdata.get_dataset(gl.base_dir,config["eval_dataset"],atlas="fs32k",sess=ses_id,type=config["type"], info_only=False)
         tensor_X_list.append(XX)
         tensor_Y_list.append(YY)
         # add a number for session
         info["ses_num"] = ses+1
         info_list.append(info)
   else: # you want to train the model on a specific session of the dataset
      YY, info, _ = fdata.get_dataset(gl.base_dir,config["eval_dataset"],atlas="SUIT3",sess=config["eval_id"],type=config["type"], info_only=False)
      XX, info, _ = fdata.get_dataset(gl.base_dir,config["eval_dataset"],atlas="fs32k",sess=config["eval_id"],type=config["type"], info_only=False)
      tensor_X_list.append(XX)
      tensor_Y_list.append(YY)
      info["ses_num"] = info["half"]
      info_list.append(info)
      
   # concatenate data across conditions over the selected sessions
   tensor_X = np.concatenate(tensor_X_list, axis = 1) # axis 1 is the conditions
   tensor_Y = np.concatenate(tensor_Y_list, axis = 1) # axis 1 is the conditions
   
   # concatenate the dataframes representing info
   info = pd.concat(info_list, axis = 0)
   # create a column to show unified condition numbers
   info["cond_num_uni"] = np.arange(1, len(info.index)+1)
   
   # get cortical atlas object(will be used to aggregate data within tessellation)
   X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
   # get the vector containing tessel labels
   X_atlas.get_parcel(config['label_img'], unite_struct = False)

   # get the mean across tessels for cortical data
   tensor_X, labels = fdata.agg_parcels(tensor_X, X_atlas.label_vector,fcn=np.nanmean)

   # get participants for the dataset
   T = Data.get_participants()
   subject_list = T.participant_id

   # loop over subjects
   for i, sub in enumerate(subject_list):
      print(f'- Evaluate {sub} {config["method"]} log alpha {config["log_alpha"]}')

      # get the slice of tensor corresponding to the current subject
      X = tensor_X[i, :, :]
      Y = tensor_Y[i, :, :]

      Y = np.nan_to_num(Y) # replace nans first 
      X = np.nan_to_num(X) # replace nans first

      # get model name
      model_path = gl.conn_dir + f"/{config['train_dataset']}/train/{config['name']}/"
      # check if the directory exists
      if avg: 
         fname = model_path + f"/{config['method']}_{config['train_id']}_logalpha{config['log_alpha']}_avg.h5"
      else:
         fname = model_path + f"/{config['method']}_{config['train_id']}_logalpha{config['log_alpha']}_{sub}.h5"
         
      fitted_model = dd.io.load(fname)
       
      # Get model predictions
      Y_pred = fitted_model.predict(X)
      # cross the sessions
      if config["mode"] == "crossed":
         Y = np.r_[Y[info["ses_num"] == 2, :], Y[info["ses_num"] == 1, :]]
      
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
      eval_path = gl.conn_dir + config['eval_dataset']+ f'/{sub}/eval/'
      # Save the maps
      
   return pd.DataFrame.from_dict(eval_dict), eval_voxels

def get_group_weights(config, fcn = np.nanmean, fold = "train"):
   """
   loop over the model for each subject, get weight and calculate group measure
   """

   # get the dataset class the model was trained on
   Data = fdata.get_dataset_class(gl.base_dir, dataset=config["dataset"])  
   # get the directory where models are saved
   model_path = gl.conn_dir + f"/{config['dataset']}/{fold}/{config['name']}"

   # get the list of subjects
   T = Data.get_participants()
   subject_list = T.participant_id

   # loop over subjects and load the model
   weight_list = []
   for sub in subject_list:
      print(f"- getting weights for {sub}")
      # load the model
      fname = model_path + f"/{config['method']}_alpha{config['log_alpha']}_{sub}.h5"
      fitted_model = dd.io.load(fname)

      # get the weights and append it to the list
      ## a new axis is added to represent subject
      weight_list.append(fitted_model.coef_[np.newaxis, ...])

   # concatenate weights in the list
   weight_array = np.concatenate(weight_list, axis = 0)

   # apply function to get group weights
   weight_group = fcn(weight_array,axis = 0)
   return weight_group