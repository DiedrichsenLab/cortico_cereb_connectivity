import os
import numpy as np
import time
import deepdish as dd
import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import connectivity.io as cio
from connectivity import data as cdata
import connectivity.constants as const
import connectivity.model as model
import connectivity.evaluation as ev

import warnings

warnings.filterwarnings("ignore")

np.seterr(divide="ignore", invalid="ignore")

"""Main module for training and evaluating connectivity models.

   @authors: Maedbh King, Jörn Diedrichsen

  Typical usage example:

  config = get_default_train_config()
  models = train_models(config=config)

  config = get_default_eval_config()
  models = eval_models(config)
"""

def train_models(config, save=False):
    """Trains a specific model class on X and Y data from a specific experiment for subjects listed in config.

    Args:
        config (dict): Training configuration, returned from get_default_train_config()
        save (bool): Optional; Save fitted models automatically to disk.
    Returns:
        models (list): list of trained models for subjects listed in config.
        train_all (pd dataframe): dataframe containing
    """

    dirs = const.Dirs(exp_name=config["train_exp"], glm=config["glm"])
    models = []
    train_all = defaultdict(list)
    # Store the training configuration in model directory
    if save:
        fpath = os.path.join(dirs.conn_train_dir, config["name"])
        cio.make_dirs(fpath)
        cio.save_dict_as_JSON(os.path.join(fpath, "train_config.json"), config)

    # Loop over subjects and train
    for subj in config["subjects"]:
        print(f"Training model on {subj}")

        # get data
        Y, Y_info, X, X_info = _get_XYdata(config=config, exp=config["train_exp"], subj=subj)

        # Generate new model and put in the list
        new_model = getattr(model, config["model"])(**config["param"])
        models.append(new_model)

        # cross the sessions
        if config["mode"] == "crossed":
            Y = np.r_[Y[Y_info.sess == 2, :], Y[Y_info.sess == 1, :]]

        # Fit model, get train and validate metrics
        models[-1].fit(X, Y)
        models[-1].rmse_train, models[-1].R_train = train_metrics(models[-1], X, Y)

        # collect train metrics (rmse and R)
        data = {
            "subj_id": subj,
            "rmse_train": models[-1].rmse_train,
            "R_train": models[-1].R_train,
            "num_regions": X.shape[1]
            }

        # run cross validation and collect metrics (rmse and R)
        if config['validate_model']:
            models[-1].rmse_cv, models[-1].R_cv = validate_metrics(models[-1], X, Y, X_info, config["cv_fold"])
            data.update({"rmse_cv": models[-1].rmse_cv,
                        "R_cv": models[-1].R_cv
                        })

        # Copy over all scalars or strings from config to eval dict:
        for key, value in config.items():
            if not isinstance(value, (list, dict)):
                data.update({key: value})

        # Save the fitted model to disk if required
        if save:
            fname = _get_model_name(train_name=config["name"], exp=config["train_exp"], subj_id=subj)
            dd.io.save(fname, models[-1], compression=None)

            # add date/timestamp to dict (to keep track of models)
            timestamp = time.ctime(os.path.getctime(fname))
            data.update({'timestamp': timestamp})

        for k, v in data.items():
            train_all[k].append(v)

    return models, pd.DataFrame.from_dict(train_all)

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

def validate_metrics(model, X, Y, X_info, cv_fold):
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

def eval_models(config):
    """Evaluates a specific model class on X and Y data from a specific experiment for subjects listed in config.

    Args:
        config (dict): Evaluation configuration, returned from get_default_eval_config()
    Returns:
        models (pd dataframe): evaluation of different models on the data
    """

    eval_all = defaultdict(list)
    eval_voxels = defaultdict(list)

    for idx, subj in enumerate(config["subjects"]):

        print(f"Evaluating model on {subj}")

        # get data
        Y, Y_info, X, X_info = _get_XYdata(config=config, exp=config["eval_exp"], subj=subj)

        # Get the model from file
        fname = _get_model_name(train_name=config["name"], exp=config["train_exp"], subj_id=subj)
        fitted_model = dd.io.load(fname)

        # Get model predictions
        Y_pred = fitted_model.predict(X)
        if config["mode"] == "crossed":
            Y_pred = np.r_[Y_pred[Y_info.sess == 2, :], Y_pred[Y_info.sess == 1, :]]

        # get rmse
        rmse = mean_squared_error(Y, Y_pred, squared=False)
        data = {"rmse_eval": rmse,
                "subj_id": subj,
                "num_regions": X.shape[1]}

        # Copy over all scalars or strings to eval_all dataframe:
        for key, value in config.items():
            if type(value) is not list:
                data.update({key: value})

        # add evaluation (summary)
        evals = _get_eval(Y=Y, Y_pred=Y_pred, Y_info=Y_info, X_info=X_info)
        data.update(evals)

        # add evaluation (voxels)
        if config["save_maps"]:
            for k, v in data.items():
                if "vox" in k:
                    eval_voxels[k].append(v)

        # don't save voxel data to summary
        data = {k: v for k, v in data.items() if "vox" not in k}

        # add model timestamp
        # add date/timestamp to dict (to keep track of models)
        timestamp = time.ctime(os.path.getctime(fname))
        data.update({'timestamp': timestamp})

        # append data for each subj
        for k, v in data.items():
            eval_all[k].append(v)

    # Return list of models
    return pd.DataFrame.from_dict(eval_all), eval_voxels

def _get_eval(Y, Y_pred, Y_info, X_info):
    """Compute evaluation, returning summary and voxel data.

    Args:
        Y (np array):
        Y_pred (np array):
        Y_info (pd dataframe):
        X_info (pd dataframe):
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
    ) = ev.calculate_reliability(Y=Y, dataframe=Y_info)

    # Noise ceiling for cerebellum (squared)
    (
        data["noise_X_R"],
        data["noise_X_R_vox"],
        data["noise_X_R2"],
        data["noise_X_R2_vox"],
    ) = ev.calculate_reliability(Y=Y_pred, dataframe=X_info)

    # calculate noise ceiling
    data["noiseceiling_Y_R_vox"] = np.sqrt(data["noise_Y_R_vox"])
    data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"] * np.sqrt(data["noise_X_R_vox"]))

    # # Noise ceiling for cortex (squared)
    #     pass

    return data

def _get_XYdata(config, exp, subj):
    """get X and Y data for exp and subj

    Args:
        config (dict): must contain keys for glm, Y_data, X_data, averaging, weighting
        exp (str): 'sc1' or 'sc2'
        subj (str): default subjs are set in constants.py
    Returns:
        Y (nd array), Y_info (pd dataframe), X (nd array), X_info (pd dataframe)
    """

    # Get cerebellar data and load mat
    Ydata = cdata.Dataset(
        experiment=exp,
        glm=config["glm"],
        subj_id=subj,
        roi=config["Y_data"],
    )
    Ydata.load_mat()

    # get dataframe
    df = Ydata.get_info()

    exp_num = int(re.findall('(\d+)', exp)[0])

    # figure out splitby
    subset = None
    if 'splitby' in config:
        if config['splitby']!='all':
            subset = (df['split']==config['splitby'])
        else:
            subset = df['task']>0 # Should be all true

    # Include the instructions?
    if not config['incl_inst']:
        subset = subset & (df['inst']==0)

    Y, Y_info = Ydata.get_data(averaging=config["averaging"], weighting=config["weighting"], subset=subset)

    # Get cortical data and load mat
    Xdata = cdata.Dataset(
        experiment=exp,
        glm=config["glm"],
        subj_id=subj,
        roi=config["X_data"],
    )
    Xdata.load_mat()
    X, X_info = Xdata.get_data(averaging=config["averaging"], weighting=config["weighting"], subset=subset)

    return Y, Y_info, X, X_info

def _get_model_name(train_name, exp, subj_id):
    """returns path/name for connectivity training model outputs.

    Args:
        train_name (str): Name of trained model
        exp (str): Experiment name
        subj_id (str): Subject id
    Returns:
        fpath (str): Full path and name to connectivity output for model training.
    """

    dirs = const.Dirs(exp_name=exp)
    fname = f"{train_name}_{subj_id}.h5"
    fpath = os.path.join(dirs.conn_train_dir, train_name, fname)
    return fpath
