"""Main module for training and evaluating connectivity models.

   @authors: Ladan Shahshahani, Maedbh King, Jörn Diedrichsen
"""
import os
import numpy as np
import time
import deepdish as dd
import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from Functional_Fusion.dataset import *

import nibabel as nb

import connectivity.data as cdata
import connectivity.model as model
import connectivity.evaluation as ev

import warnings

warnings.filterwarnings("ignore")

np.seterr(divide="ignore", invalid="ignore")

# OPEN ISSUES: 
# 1. Where to save the train and evaluation results?


# make sure you have extrated data in functional fusion framework before running these function
# calculating training metrics
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

# validating train metrics
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

# train models on a dataset
def train_model(base_dir,
                atlas_dir, 
                model_name, 
                model_alpha, 
                train_dataset = "Demand",
                ses_id = 1,
                type = "CondHalf", 
                cortex = "Icosahedron-42_Sym",
                cerebellum = "SUIT3", 
                validate = True, 
                cv_fold = 4, 
                train_mode = 'crossed'):
    """
    train a model on a specific dataset

    """
    D = train_dataset(base_dir, train_dataset)
    # Create an isntance of the connectivity Dataset class
    Y = cdata(experiment="train_dataset", ses_id = 1, atlas="SUIT3", type = "CondHalf", subj_id="s02")

    # loop through participants
    for s in D.participant_id:

        print(f"- Training model {model_name} param {model_alpha}")

        # load cerebellar data
        Y = D.load_Y(subj_id = s)

        # load cortical data
        ## get label img
        # label_img = os.path.join(atlas_dir, 'tpl-fs32k', f'Icosahedron-642_Sym.32k.{hemi[idx]}.label.gii')
        X = D.load_X(subj_id = s, cortex = cortex)



        # create an instance of the model
        new_model = getattr(model, model_name)(model_alpha)
        # models.append(new_model)        

        # cross the sessions
        if train_mode == "crossed":
            Y = np.r_[Y[info.sess == 2, :], Y[info.sess == 1, :]]

        # Fit model, get train and validate metrics
        new_model.fit(X, Y)
        new_model.rmse_train, new_model.R_train = train_metrics(new_model, X, Y)

        # initialize a dictionary with training info
        model_info = {
                "model_name":new_model.name,
                "alpha": new_model.alpha, 
                # get cortex name and cerebellum name as atlases/parcellations
                "rmse_train": new_model.rmse_train,
                "R_train": new_model.R_train,
                "num_regions": X.shape[1]
                }

        # run cross validation and collect metrics (rmse and R)
        if validate:
            new_model.rmse_cv, new_model.R_cv = validate_metrics(new_model, X, Y, info, cv_fold)
            model_info.update({"rmse_cv": new_model.rmse_cv,
                        "R_cv": new_model[-1].R_cv
                        })

        # Save trained data to the disk
        fname = os.path.join("PATH TO CONN MODEL", f"{dataset}_{cortex}_{cerebellum}_{s}_{model_name}_alpha_{model_alpha}.h5")
        dd.io.save(fname, new_model, compression=None)
    return

def eval_model():
    """
    Evaluates the model
    """
    return


# def eval_models(config):
#     """Evaluates a specific model class on X and Y data from a specific experiment for subjects listed in config.

#     Args:
#         config (dict): Evaluation configuration, returned from get_default_eval_config()
#     Returns:
#         models (pd dataframe): evaluation of different models on the data
#     """

#     eval_all = defaultdict(list)
#     eval_voxels = defaultdict(list)

#     for idx, subj in enumerate(config["subjects"]):

#         print(f"Evaluating model on {subj}")

#         # get data
#         Y, Y_info, X, X_info = _get_XYdata(config=config, exp=config["eval_exp"], subj=subj)

#         # Get the model from file
#         fname = _get_model_name(train_name=config["name"], exp=config["train_exp"], subj_id=subj)
#         fitted_model = dd.io.load(fname)

#         # Get model predictions
#         Y_pred = fitted_model.predict(X)
#         if config["mode"] == "crossed":
#             Y_pred = np.r_[Y_pred[Y_info.sess == 2, :], Y_pred[Y_info.sess == 1, :]]

#         # get rmse
#         rmse = mean_squared_error(Y, Y_pred, squared=False)
#         data = {"rmse_eval": rmse,
#                 "subj_id": subj,
#                 "num_regions": X.shape[1]}

#         # Copy over all scalars or strings to eval_all dataframe:
#         for key, value in config.items():
#             if type(value) is not list:
#                 data.update({key: value})

#         # add evaluation (summary)
#         evals = _get_eval(Y=Y, Y_pred=Y_pred, Y_info=Y_info, X_info=X_info)
#         data.update(evals)

#         # add evaluation (voxels)
#         if config["save_maps"]:
#             for k, v in data.items():
#                 if "vox" in k:
#                     eval_voxels[k].append(v)

#         # don't save voxel data to summary
#         data = {k: v for k, v in data.items() if "vox" not in k}

#         # add model timestamp
#         # add date/timestamp to dict (to keep track of models)
#         timestamp = time.ctime(os.path.getctime(fname))
#         data.update({'timestamp': timestamp})

#         # append data for each subj
#         for k, v in data.items():
#             eval_all[k].append(v)

#     # Return list of models
#     return pd.DataFrame.from_dict(eval_all), eval_voxels

# def _get_eval(Y, Y_pred, Y_info, X_info):
#     """Compute evaluation, returning summary and voxel data.

#     Args:
#         Y (np array):
#         Y_pred (np array):
#         Y_info (pd dataframe):
#         X_info (pd dataframe):
#     Returns:
#         dict containing evaluations (R, R2, noise).
#     """
#     # initialise dictionary
#     data = {}

#     # Add the evaluation
#     data["R_eval"], data["R_vox"] = ev.calculate_R(Y=Y, Y_pred=Y_pred)

#     # R between predicted and observed
#     data["R2"], data["R2_vox"] = ev.calculate_R2(Y=Y, Y_pred=Y_pred)

#     # R2 between predicted and observed
#     (
#         data["noise_Y_R"],
#         data["noise_Y_R_vox"],
#         data["noise_Y_R2"],
#         data["noise_Y_R2_vox"],
#     ) = ev.calculate_reliability(Y=Y, dataframe=Y_info)

#     # Noise ceiling for cerebellum (squared)
#     (
#         data["noise_X_R"],
#         data["noise_X_R_vox"],
#         data["noise_X_R2"],
#         data["noise_X_R2_vox"],
#     ) = ev.calculate_reliability(Y=Y_pred, dataframe=X_info)

#     # calculate noise ceiling
#     data["noiseceiling_Y_R_vox"] = np.sqrt(data["noise_Y_R_vox"])
#     data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"] * np.sqrt(data["noise_X_R_vox"]))

#     # # Noise ceiling for cortex (squared)
#     #     pass

#     return data


