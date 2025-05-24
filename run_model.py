"""Main module for training and evaluating connectivity models.
   Designed to work together with Functional_Fusion package.
   Dataset, session, and parcellation names are as in Functional_Fusion.
   The main work is being done by train_model and eval_model functions.
   @authors: Ladan Shahshahani, Maedbh King, Jörn Diedrichsen
"""

# from audioop import cross
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.evaluation as ev
from copy import copy, deepcopy
import warnings

# warnings.filterwarnings("ignore")

def get_train_config(train_dataset = "MDTB",
                     train_ses = "ses-s1",
                     train_run = 'all',
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
                     add_rest = False,
                     std_cortex = None,
                     std_cerebellum = None,
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
      add_rest (bool): Add rest condition to each session and half. Defaults to False.
      std_cortex(): z-Standardize the cortical data. (Defaults to None)
      std_cerebelum(): z-Standardize the cortical data. (Defaults to None)
   Returns:
      dict: Dictionary containing the default training configuration
   """
   train_config = {}
   train_config['train_dataset'] = train_dataset # name of the dataset to be used in
   train_config['train_ses'] = train_ses
   train_config['train_run'] = train_run
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
   train_config['std_cortex'] = std_cortex
   train_config['std_cerebellum'] = std_cerebellum
   train_config['append'] = False

   # get label images for left and right hemisphere
   train_config['label_img'] = []
   for hemi in ['L', 'R']:
      train_config['label_img'].append(gl.atlas_dir + f'/tpl-{train_config["cortex"]}' + f'/{train_config["parcellation"]}.{hemi}.label.gii')

   return train_config

def get_eval_config(eval_dataset = 'MDTB',
            eval_ses = 'ses-s2',
            subj_list = 'all',
            cerebellum = 'SUIT3',
            cortex = "fs32k",
            parcellation = "Icosahedron1002",
            crossed = "half", # or None
            type = "CondHalf",
            splitby = None,
            add_rest = False,
            std_cortex = 'parcel',
            std_cerebellum = 'global',
            model = 'avg'):
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
   eval_config['add_rest'] = add_rest
   eval_config["splitby"] = splitby
   eval_config["type"] = type
   eval_config['subj_list'] = subj_list
   eval_config['model'] = model
   eval_config['add_rest'] = add_rest
   eval_config['std_cortex'] = std_cortex
   eval_config['std_cerebellum'] = std_cerebellum

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
      data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"] * np.sqrt(data["noise_X_R_vox"]))
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
       Y (_type_): _description_
       info (_type_): _description_

   Returns:
       _type_: _description_
   """
   Y_list = []
   info_list = []
   if not('cond_name' in info.columns):
      info['cond_name']=info.task_name
   for s in np.unique(info.sess):
      for h in np.unique(info.half):
         indx = (info.sess==s) & (info.half==h)
         if any([i.startswith('rest') for i in info[indx].cond_name]):
            Y_list.append(Y[indx,:])
            info_list.append(info[indx])
         else:
            Yp = np.zeros((indx.sum()+1,Y.shape[1]))
            Yp[0:-1,:] = Y[indx,:]
            Yp = Yp - Yp.mean(axis=0)
            Y_list.append(Yp)
            inf = info[indx]
            newD = {'cond_name':['rest'],
                    'sess':[inf.sess.iloc[0]],
                    'half':[inf.half.iloc[0]]}
            inf = pd.concat([inf,pd.DataFrame(newD)],ignore_index=True)
            info_list.append(inf)
   Ys = np.concatenate(Y_list,axis=0)
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
   else:
      raise ValueError('std_mode must be None, "voxel" or "global"')

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
   # get dataset class
   dataset = fdata.get_dataset_class(gl.base_dir,
                                    dataset=config["train_dataset"])

   ## loop over sessions chosen through train_id and concatenate data
   info_list = []

   T = dataset.get_participants()
   if config["subj_list"] is None:
      config["subj_list"] = T.participant_id
   elif isinstance(config['subj_list'],(list,pd.Series,np.ndarray)):
      if isinstance(config['subj_list'][0],str):
         pass
      else: # Numerical 
         config["subj_list"] = T.participant_id.iloc[config['subj_list']]
   elif config["subj_list"]=='all':
      config["subj_list"] = T.participant_id
   else:
      raise ValueError('config["subj_list"] must be a list of str, integers or "all"')

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

      # Add rest condition?
      if config["add_rest"]:
         Y,_ = add_rest(Y,info)
         X,info = add_rest(X,info)

      # train only on some runs?
      if config["train_run"]!='all':
         if isinstance(config["train_run"], list):
            run_mask = info['run'].isin(config["train_run"])
            Y = Y[run_mask.values, :]
            X = X[run_mask.values, :]
            info = info[run_mask]

      #Definitely subtract intercept across all conditions
      X = (X - X.mean(axis=0))
      Y = (Y - Y.mean(axis=0))

      if 'std_cortex' in config.keys():
         X = std_data(X,config['std_cortex'])
      if 'std_cerebellum' in config.keys():
         Y = std_data(Y,config['std_cerebellum'])

      # cross the halves within each session
      if config["crossed"] is not None:
         Y = cross_data(Y,info,config["crossed"])

      for la in config["logalpha"]:
      # loop over subjects and train models
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
         conn_model.fit(X, Y)
         R_train,R2_train = train_metrics(conn_model, X, Y)
         conn_model_list.append(conn_model)

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

def get_fitted_models(model_dirs,model_names,config):
   """Builds a list of fitted models from the saved files
   In case of individual-specific models (ind or loo), it builds a list of lists.

   Args:
       model_dirs (_type_): List of dirctory names for models 
       model_names (_type_): List of model names (without subject extension)
       config (dict): Dictonary with evaluation parameters

   Returns:
       fitted_models (list): _description_
       train_info (list): information on each trained model
   """
   # Load all the models to evaluate:
   fitted_model = []
   train_info = []
   num_subj = len(config['subj_list'])

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
         fm,fi = calc_avrg_model(config['eval_dataset'],d,ext,
                                 subj=config['subj_list'],
                                 avrg_mode='loo_sep')
         fitted_model.append(fm)
         train_info.append(fi)

   return fitted_model, train_info

def eval_model(model_dirs,model_names,config):
   """
   evaluate models on a specific dataset and session
   if config['model']=='avg' it will average the models across all subjects and evaluate 
   if config['model']=='ind' it will evaluate each model on the corresponding subject
   if config['model']=='loo' it will average all other subjects 
   For 'ind' and 'loo' training and evaluation dataset must be the same
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

   # Get the list of fitted models
   fitted_model,train_info = get_fitted_models(model_dirs,model_names,config)

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

      # Add explicit rest to sessions
      if config["add_rest"]:
         Y,_ = add_rest(Y,info)
         X,info = add_rest(X,info)

      #Definitely subtract intercept across all conditions
      X = (X - X.mean(axis=0))
      Y = (Y - Y.mean(axis=0))

      if 'std_cortex' in config.keys():
         X = std_data(X,config['std_cortex'])
      if 'std_cerebellum' in config.keys():
         Y = std_data(Y,config['std_cerebellum'])

      # cross the halves within each session
      if config["crossed"] is not None:
         Y = cross_data(Y,info,config["crossed"])

      # Loop over models
      for j, (fm, tinfo) in enumerate(zip(fitted_model, train_info)):

         # Use subject-specific model? (indiv or loo)
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


def calc_avrg_model(train_dataset,
                    mname_base,
                    mname_ext,
                    cerebellum='SUIT3',
                    parameters=['scale_','coef_'],
                    avrg_mode='avrg_sep',
                    subj='all'):
   """Get the fitted models from all the subjects in the training data set
      and create group-averaged model
   Args:
       train_dataset (str): _description_
       mname_base (str): Directory name for model (MDTB_all_Icosahedron1002_L2regression)
       mname_ext (str): Extension of name - typically logalpha
       (_A0)
       parameters (list): List of parameters to average
   """

   # get the dataset class the model was trained on
   # To get the list of subjects
   tdata = fdata.get_dataset_class(gl.base_dir, dataset=train_dataset)
   T = tdata.get_participants()

   if isinstance(subj,(list,pd.Series)):
      subject_list = subj
   elif isinstance(subj,np.ndarray):
      subject_list = T.participant_id.iloc[subj]
   elif isinstance(subj,str):
      if subj=='all':
         subject_list = T.participant_id
      else:
         subject_list = [subj]
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
      for s,sub in enumerate(subject_list):
         avrg_model.append(copy(fitted_model))
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         for s,sub in enumerate(subject_list):
            setattr(avrg_model[s],p,P[subj_ind!=s].mean(axis=0))

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