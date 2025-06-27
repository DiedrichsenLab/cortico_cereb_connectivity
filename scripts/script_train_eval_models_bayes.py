"""
script for training models
@ Ali Shahbazi
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.atlas_map as at
import nitools as nt
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.model as c_model
import cortico_cereb_connectivity.evaluation as ev
import json

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
                 crossed = "half",
                 type = "CondHalf",
                 train_ses = 'all',
                 cond_num = 'all',
                 dataset = "MDTB",
                 add_rest = True,
                 parcellation = "Icosahedron1002",
                 subj_list = "all",
                 cerebellum='MNISymC3',
                 method = "L2reg",
                 validate_model = False,
                 std_cortex = None,
                 std_cerebellum = 'global',
                 cortical_cerebellar_act = 'ind',
                 save_path=None,
                 mname=None):

   if std_cortex is None:
      std_cortex = 'global' if dataset=='Somatotopic' or dataset=='WMFS' else 'parcel'

   config = rm.get_train_config(log_alpha=logalpha_list,
                                 crossed=crossed,
                                 type=type,
                                 cond_num=cond_num,
                                 subj_list=subj_list,
                                 cerebellum=cerebellum,
                                 parcellation=parcellation,
                                 train_dataset=dataset,
                                 method=method,
                                 train_ses=train_ses,
                                 add_rest=add_rest,
                                 validate_model=validate_model,
                                 std_cortex=std_cortex,
                                 std_cerebellum=std_cerebellum,
                                 cortical_cerebellar_act=cortical_cerebellar_act,
                                 append=False)
   
   # get the subject list
   config["subj_list"] = rm.get_subj_list(subj_list, dataset)

   if mname is None:
      config, conn_list, df_tmp = rm.train_model(config)
   else:
      config, conn_list, df_tmp = rm.train_model(config, mname=mname)
   return df_tmp


def avrg_model(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
               train_data = "MDTB",
               train_ses= "all",
               train_run='all',
               parcellation = 'Icosahedron1002',
               method='L2reg',
               type='CondHalf',
               cerebellum='MNISymC3',
               parameters=['coef_'],
               avrg_mode = 'avrg_sep',
               avg_id = 'avg',
               model_path=None,
               mname_base=None):

   if mname_base is None:
      mname_base = f"{train_data}_{train_ses}_{parcellation}_{method}"
   if model_path is None:
      model_path = gl.conn_dir + f"/{cerebellum}/train/{mname_base}/"
   for la in logalpha_list:
      if la is not None:
         # Generate new model
         mname_ext = f"_A{la}"
      else:
         mname_ext = f""

      avrg_model,info = rm.calc_avrg_model(train_data,
                         mname_base,
                         mname_ext,
                         cerebellum=cerebellum,
                         parameters=parameters,
                         avrg_mode=avrg_mode)
      cio.save_model(avrg_model,info,model_path + f"/{mname_base}{mname_ext}_{avg_id}")


def bayes_avrg_model(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
               train_data = "MDTB",
               train_ses= "all",
               train_run='all',
               parcellation = 'Icosahedron1002',
               method='L2reg',
               type='CondHalf',
               cerebellum='MNISymC3',
               parameters=['coef_','coef_var'],
               avrg_mode = 'bayes',
               avg_id = 'bayes'):

   mname_base = f"{train_data}_{train_ses}_{parcellation}_{method}"
   model_path = gl.conn_dir + f"/{cerebellum}/train/{mname_base}/"
   for la in logalpha_list:
      if la is not None:
         # Generate new model
         mname_ext = f"_A{la}"
      else:
         mname_ext = f""

      if 'half' in method:
         avrg_mode = 'bayes_half'
      avrg_model,info = rm.calc_avrg_model(train_data,
                         mname_base,
                         mname_ext,
                         cerebellum=cerebellum,
                         parameters=parameters,
                         avrg_mode=avrg_mode)
      cio.save_model(avrg_model,info,model_path + f"/{mname_base}{mname_ext}_{avg_id}")


def eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
                train_dataset = "MDTB",
                train_ses = "all",
                cond_num = 'all',
                method = "L2reg",
                parcellation = "Icosahedron1002",
                cerebellum='MNISymC3',
                eval_dataset = ["MDTB"],
                eval_type = ["CondHalf"],
                eval_ses  = "all",
                run = 'all',
                eval_id = 'MD_s1',
                crossed = 'half',
                add_rest = True,
                std_cortex = None,
                std_cerebellum = 'global',
                cortical_act = 'avg',
                subj_list = "all",
                model_subj_list = "all",
                model = 'avg',
                mix_param = None,
                append = False,
                dir_extra = '',
                ):
   """_summary_

   Args:
       logalpha_list (list): logalpha or other extension
       type (str): _description_. Defaults to "CondHalf".
       train_dataset (str): _description_. Defaults to "MDTB".
       train_ses (str): _description_. Defaults to "ses-s1".
       method (str): _description_. Defaults to "L2regression".
       parcellation (str): _description_. Defaults to "Icosahedron1002".
       cerebellum (str, optional): _description_. Defaults to 'SUIT3'.
       eval_dataset (list): _description_. Defaults to ["Demand"].
       eval_type (list): _description_. Defaults to ["CondHalf"].
       eval_ses (str): _description_. Defaults to "all".
       eval_id (str): _description_. Defaults to 'Md_s1'.
       model (str): _description_. Defaults to 'avg'.
       mix_param (list): Percentage of subject weights if model mix. Defaults to [].
       append (bool): Append to existing tsv file? Defaults to False.

   Returns:
       _type_: _description_
   """

   for i,ed in enumerate(eval_dataset):
      if std_cortex is None:
         std_cortex = 'global' if ed=='Somatotopic' or ed=='WMFS' else 'parcel'

      eval_config = rm.get_eval_config(eval_dataset=ed,
                                       eval_ses=eval_ses,
                                       run=run,
                                       cond_num=cond_num,
                                       parcellation=parcellation,
                                       crossed=crossed, # "half", # or None
                                       type=eval_type[i],
                                       cerebellum=cerebellum,
                                       splitby=None,
                                       add_rest=add_rest,
                                       std_cortex=std_cortex,
                                       std_cerebellum=std_cerebellum,
                                       subj_list=subj_list,
                                       cortical_act=cortical_act)
      
      model_config = rm.get_model_config(dataset=train_dataset,
                                         subj_list=model_subj_list,
                                         model=model,
                                         cerebellum=cerebellum,
                                         mix_param=mix_param)

      dirname=[]
      mname=[]
      for a in logalpha_list:
         dirname.append(f"{train_dataset}_{train_ses}_{eval_config['parcellation']}_{method}{dir_extra}")
         mname.append(f"{train_dataset}_{train_ses}_{eval_config['parcellation']}_{method}{dir_extra}_A{a}")

      df, df_voxels = rm.eval_model(dirname,mname,eval_config,model_config)
      save_path = gl.conn_dir+ f"/{cerebellum}/eval"

      if not os.path.isdir(save_path):
         os.mkdir(save_path)
      else:
         pass
      ename = eval_config['eval_dataset']
      if eval_config['eval_ses'] != 'all':
         ses_code = eval_config['eval_ses'].split('-')[1]
         ename = eval_config['eval_dataset'] + ses_code
      file_name = save_path + f"/{ename}_{method}_{eval_id}.tsv"
      if os.path.isfile(file_name) & append:
         dd = pd.read_csv(file_name, sep='\t')
         df = pd.concat([dd, df], ignore_index=True)
      df.to_csv(file_name, index = False, sep='\t')
   return df,df_voxels


def eval_region_models(ext_list = [0, 2, 4, 6, 8, 10, 12],
                       train_dataset = "MDTB",
                       train_ses = "all",
                       train_run = 'all',
                       method = "L2reghalf",
                       parcellation = "Icosahedron1002",
                       cerebellum='MNISymC3',
                       eval_dataset = ["MDTB"],
                       eval_type = ["CondHalf"],
                       eval_ses  = "all",
                       eval_run='all',
                       eval_id = 'region',
                       crossed = 'half',
                       add_rest = True,
                       std_cortex = None,
                       std_cerebellum = 'global',
                       subj_list = "all",
                       model_subj_list = "all",
                       model = 'avg-half',
                       mix_param = [],
                       append = False):
   """_summary_

   Args:
      ext_list (list): logalpha or other extension
      type (str): _description_. Defaults to "CondHalf".
      train_dataset (str): _description_. Defaults to "MDTB".
      train_ses (str): _description_. Defaults to "ses-s1".
      method (str): _description_. Defaults to "L2regression".
      parcellation (str): _description_. Defaults to "Icosahedron1002".
      cerebellum (str, optional): _description_. Defaults to 'SUIT3'.
      eval_dataset (list): _description_. Defaults to ["Demand"].
      n_regions (int): Number of regions to evaluate. Defaults to 16.
      eval_type (list): _description_. Defaults to ["CondHalf"].
      eval_ses (str): _description_. Defaults to "all".
      eval_id (str): _description_. Defaults to 'Md_s1'.
      model (str): _description_. Defaults to 'avg'.
      mix_param (list): Percentage of subject weights if model mix. Defaults to [].
      append (bool): Append to existing tsv file? Defaults to False.

   Returns:
       _type_: _description_
   """

   for i,ed in enumerate(eval_dataset):

      if std_cortex is None:
         std_cortex = 'global' if ed=='Somatotopic' or ed=='WMFS' else 'parcel'

      config = rm.get_eval_config(train_dataset=train_dataset,
                                 eval_dataset=ed,
                                 eval_ses=eval_ses,
                                 eval_run=eval_run,
                                 parcellation=parcellation,
                                 crossed=crossed, # "half", # or None
                                 type=eval_type[i],
                                 cerebellum=cerebellum,
                                 splitby=None,
                                 add_rest=add_rest,
                                 std_cortex=std_cortex,
                                 std_cerebellum=std_cerebellum,
                                 subj_list=subj_list,
                                 model_subj_list=model_subj_list,
                                 model=model,
                                 mix_param=mix_param)
      
      config["subj_list"] = fdata.get_dataset_class(gl.base_dir, dataset=config["eval_dataset"]).get_participants().participant_id

      dirname=[]
      mname=[]
      for a in ext_list:
         dirname.append(f"{train_dataset}_{train_ses}_{config['parcellation']}_{method}")
         mname.append(f"{train_dataset}_{train_ses}_{config['parcellation']}_{method}_A{a}")

      # initialize eval dictionary
      eval_df = pd.DataFrame()
      eval_voxels = defaultdict(list)
      conn_model_list = []
      conn_info_list = []

      if config['model'] == 'avg-half':
         for d,m in zip(dirname,mname):
            model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
            fname = model_path + f"/{m}_{config['model']}"
            mo,inf = cio.load_model(fname)
            conn_model_list.append(mo)
            conn_info_list.append(inf)
      elif config['model'] == 'loo':
         conn_model_list, conn_info_list = rm.get_fitted_models(dirname,mname,config)
      
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
            Y,_ = rm.add_rest(Y,info)
            X,info = rm.add_rest(X,info)

         # eval only on some runs?
         if config["eval_run"]!='all':
            if isinstance(config["eval_run"], list):
               run_mask = info['run'].isin(config["eval_run"])
               Y = Y[run_mask.values, :]
               X = X[run_mask.values, :]
               info = info[run_mask]

         #Definitely subtract intercept across all conditions
         X = (X - X.mean(axis=0))
         Y = (Y - Y.mean(axis=0))

         if 'std_cortex' in config.keys():
            X = rm.std_data(X,config['std_cortex'])
         if 'std_cerebellum' in config.keys():
            Y = rm.std_data(Y,config['std_cerebellum'])

         # cross the halves within each session
         if config["crossed"] is not None:
            Y = rm.cross_data(Y,info,config["crossed"])

         # Get the atlas for the cerebellum
         atlas, _ = at.get_atlas(config["cerebellum"])
         atlas_dir = '/cifs/diedrichsen/data/FunctionalFusion/Atlases/tpl-MNI152NLin2009cSymC'
         atlas_fname = 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii'
         U = atlas.read_data(f'{atlas_dir}/{atlas_fname}')
         U = U.T
         atlas_labels = np.argmax(U, axis=1)+1
         _, _, atlas_names = nt.read_lut(f'{atlas_dir}/atl-NettekovenSym32.lut')
         
         # Loop over models
         for conn_mo, conn_info in zip(conn_model_list, conn_info_list):
            for r in range(32):
               # Use subject-specific model? (indiv or loo or mix)
               if (isinstance(conn_mo,list)):
                  fitM = conn_mo[i]
               else:
                  fitM = conn_mo

               if (isinstance(conn_info,list)):
                  ti = conn_info[i]
               else:
                  ti = conn_info

               # make a subset of Y based on regions
               Y_region = Y[:, atlas_labels == r+1]
               # make a subset of W based on regions
               W_region = fitM.coef_[atlas_labels == r+1, :]

               Y_pred = X @ W_region.T

               eval_sub = {"eval_subj": sub, "num_regions": X.shape[1], "cereb_region_num": r+1,
                           "cereb_region_name": atlas_names[r+1]}

               # Copy over all scalars or strings to eval_all dataframe:
               for key, value in ti.items():
                  if not isinstance(value,(list,pd.Series,np.ndarray)):
                     eval_sub.update({key: value})
               for key, value in config.items():
                  if not isinstance(value,(list,pd.Series,np.ndarray)):
                     eval_sub.update({key: value})

               # add evaluation (summary)
               evals = rm.eval_metrics(Y=Y_region, Y_pred=Y_pred, info = info)

               # add evaluation (voxels)
               for k, v in evals.items():
                  if "vox" in k:
                     eval_voxels[k].append(v)
                  else:
                     eval_sub[k]=v

               # don't save voxel data to summary
               eval_df = pd.concat([eval_df,pd.DataFrame(eval_sub,index=[0])],ignore_index= True)


      save_path = gl.conn_dir+ f"/{cerebellum}/eval"

      if not os.path.isdir(save_path):
         os.mkdir(save_path)
      else:
         pass
      ename = config['eval_dataset']
      if config['eval_ses'] != 'all':
         ses_code = config['eval_ses'].split('-')[1]
         ename = config['eval_dataset'] + ses_code
      file_name = save_path + f"/{ename}_{method}_{eval_id}.tsv"
      if os.path.isfile(file_name) & append:
         dd = pd.read_csv(file_name, sep='\t')
         eval_df = pd.concat([dd, eval_df], ignore_index=True)
      eval_df.to_csv(file_name, index = False, sep='\t')


def fuse_models_loso(train_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    train_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    eval_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    eval_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    logalpha=[8, 8, 8, 8, 8, 10, 8],
                    weight=[1, 1, 1, 1, 1, 1, 1],
                    model='avg',   # "avg" or "bayes"
                    method='L2reghalf',   # should be L2reghalf if weight='voxel-rel'
                    parcellation='Icosahedron1002',
                    cerebellum='MNISymC3',
                    eval_id='Fus-avg'):
    
   # First load all basic dataset models
   coef_list = []
   num_subj = []
   rel_map_list = []
   loo_weights = [1] * len(train_datasets)
   for i, (la, tdata, tses) in enumerate(zip(logalpha, train_datasets, train_ses)):
      print(f'Loading avg model for {tdata} - {tses}')
      mname = f"{tdata}_{tses}_{parcellation}_{method}"
      model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
      m = mname + f"_A{la}_{model}"
      fname = model_path + f"/{m}"
      conn_mo, info = cio.load_model(fname)
      coef_list.append(conn_mo.coef_)

      # Calculate the reliability map for the model
      if isinstance(weight, str) and 'voxel-rel' in weight and method=='L2reghalf':
         _, R_vox = ev.calculate_R(conn_mo.coef_1.T, conn_mo.coef_2.T)
         rel_map_list.append(2*R_vox / (R_vox + 1))

      T = fdata.get_dataset_class(gl.base_dir, dataset=tdata).get_participants()
      # Get the number of subjects in the dataset
      num_subj.append(len(T.participant_id))

   for i, edata in enumerate(eval_datasets):
      print(f'\n Fusing models for {edata}')
      # Next get all individual models from the evaluation dataset  
      indx = train_datasets.index(edata)
      dataset = fdata.get_dataset_class(gl.base_dir, dataset=edata)
      T = dataset.get_participants()

      mname = f"{edata}_{train_ses[indx]}_{parcellation}_{method}"
      mext = f"_A{logalpha[indx]}"
      config = rm.get_eval_config(eval_dataset=edata,
                                  eval_ses=eval_ses[i],
                                  parcellation=parcellation,
                                  crossed='half', # "half", # or None
                                  type='CondHalf',
                                  cerebellum=cerebellum,
                                  add_rest=True,
                                  std_cortex='global' if edata=='Somatotopic' or edata=='WMFS' else 'parcel',
                                  std_cerebellum='global',
                                  subj_list=list(T.participant_id),
                                  model='loo')
      
      # Get individual models for the individual dataset
      ind_models,info = rm.get_fitted_models([mname],[mext],config)
      # Get the first (and only) model
      ind_models = ind_models[0]
      info = info[0]
      
      # adjust the evaluation dataset weight with loo
      if weight == 'num_subj':
         loo_weights = num_subj.copy()
         loo_weights[indx] -= 1
         loo_weights /= np.sum(loo_weights)
         loo_weights *= len(train_datasets)
      elif isinstance(weight, list) and all(isinstance(w, int) for w in weight):
         loo_weights = weight

      # Get the model for the fused dataset
      loo_fuse_models = []
      Coef = np.stack(coef_list,axis=0)
      rel_map = np.stack(rel_map_list, axis=0) if rel_map_list else None
      for s,m in enumerate(ind_models):
         print('Fuse model for subject',T.participant_id.iloc[s])
         Coef_dup = Coef.copy()
         if rel_map is not None:
            rel_map_dup = rel_map.copy()
         # Find where train_data is equal to ed
         Coef_dup[indx,:,:] = m.coef_
      
         weight_norm = np.sqrt(np.nansum((Coef_dup**2),axis=(1,2),keepdims=True))
         if np.any(np.isinf(weight_norm)):
            inf_indices = np.where(np.isinf(weight_norm))[0]
            print(f"Warning: weight_norm contains inf values for dataset {train_datasets[inf_indices]}.")
         wCoef = Coef_dup/weight_norm
         wCoef = wCoef * np.array(loo_weights).reshape(-1,1,1)

         if isinstance(weight, str) and 'voxel-rel' in weight and method=='L2reghalf':
            _, R_vox = ev.calculate_R(m.coef_1.T, m.coef_2.T)
            rel_map_dup[indx,:] = 2*R_vox / (R_vox + 1)

            if 'WTA' in weight:
               # Perform winner-take-all on rel_map_dup along axis 0
               winner_mask = np.zeros_like(rel_map_dup, dtype=bool)
               winner_indices = np.argmax(rel_map_dup, axis=0)
               winner_mask[winner_indices, np.arange(rel_map_dup.shape[1])] = True
               rel_map_dup = np.where(winner_mask, 1, 0)

            wCoef = wCoef * rel_map_dup[:,:,np.newaxis]

         setattr(m,'coef_',np.nanmean(wCoef,axis=0))
         loo_fuse_models.append(m)

      # Needs to be wrapped in a list, as it is one model with a specific model per subject
      config['model'] = [loo_fuse_models]
      info['extension'] = eval_id
      info['logalpha'] = logalpha
      info['weight'] = loo_weights

      config['train_info'] = [info]
      df, df_voxels = rm.eval_model(None,None,config)
      save_path = gl.conn_dir+ f"/{cerebellum}/eval"

      ename = config['eval_dataset']
      if config['eval_ses'] != 'all':
         ses_code = config['eval_ses'].split('-')[1]
         ename = config['eval_dataset'] + ses_code
      file_name = save_path + f"/{ename}_{method}_{eval_id}.tsv"
      df['train_dataset'] = ['Fusion'] * df.shape[0]
      df['model'] = ['loo']*df.shape[0]
      df['logalpha'] = [logalpha]*df.shape[0]
      df['weight'] = [loo_weights]*df.shape[0]
      df.to_csv(file_name, index = False, sep='\t')
      

def fuse_models_lodo(train_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    train_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    eval_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    eval_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    add_rest=[False, False, True, True, True, False, True],
                    std_cortex=['parcel', 'parcel', 'global', 'parcel', 'global', 'parcel', 'parcel'],
                    logalpha=[8, 8, 8, 8, 8, 10, 8],
                    model='avg',   # "avg" or "bayes"
                    method='L2reghalf',
                    parcellation='Icosahedron1002',
                    cerebellum='MNISymC3',
                    cortical_act='avg',
                    eval_id='Fus-avg',
                    save_model=True):
   
   # get other datasets models and weights
   if model == 'avg':
      coef_list = []
      for i, (la, tdata, tses) in enumerate(zip(logalpha, train_datasets, train_ses)):
         print(f'Loading avg model for {tdata} - {tses}')
         mname = f"{tdata}_{tses}_{parcellation}_{method}"
         model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
         m = mname + f"_A{la}_{model}"
         fname = model_path + f"/{m}"
         conn_mo, info = cio.load_model(fname)
         coef_list.append(conn_mo.coef_)
   elif model == 'bayes':
      # Decompose Variances and get weights
      data = np.load("/home/UWO/ashahb7/Github/bayes_temp/bestSTD_product_matrix.npz", allow_pickle=True)
      indices = []
      for ds, la in zip(train_datasets, logalpha):
         indices.append(np.where((data['dataset_vec'] == ds) & (data['logalpha_vec'] == la))[0])
      indices = np.concatenate(indices)
      product_la = data['product_matrix'][np.ix_(indices, indices)]
      dataset_vec_la = data['dataset_vec'][indices]
      sub_vec_la = data['sub_vec'][indices]
      part_vec_la = data['part_vec'][indices]
      # Solve
      print('Decomposing variances ...')
      full_var_decom_df = rm.decompose_variance_from_SS_2(product_la, dataset_vec_la, sub_vec_la, part_vec_la)

      # Get the models
      coef_list = []
      for i, (la, tdata, tses) in enumerate(zip(logalpha, train_datasets, train_ses)):
         print(f'\nLoading models for {tdata} - {la} - {tses} ...')
         mname = f"{tdata}_{tses}_{parcellation}_{method}"
         model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
         T = fdata.get_dataset_class(gl.base_dir, dataset=tdata).get_participants()

         for s,subj in enumerate(T.participant_id):
            print('  subject -',subj)
            m = mname + f"_A{la}_{subj}"
            fname = model_path + f"/{m}"
            conn_mo, info = cio.load_model(fname)

            # find the corresponding weight in the decomposition
            row = full_var_decom_df[
               (full_var_decom_df['train_dataset'] == tdata) &
               (full_var_decom_df['subj_id'] == subj)
            ]

            if row['v_m'].values[0] > 0 and row['v_s'].values[0] > 0 and row['v_d'].values[0] > 0:
               weight = 1 / (row['sc'].values[0]) / (row['v_m'].values[0] + row['v_s'].values[0] + row['v_d'].values[0])
            else:
               weight = 0

            if s == 0:
               dataset_coef = conn_mo.coef_ * weight
            else:
               dataset_coef += conn_mo.coef_ * weight

         coef_list.append(dataset_coef)

   all_df = pd.DataFrame()
   for i, edata in enumerate(eval_datasets):
      print(f'\nFusing models for {edata} ...')
      T = fdata.get_dataset_class(gl.base_dir, dataset=edata).get_participants()
      eval_config = rm.get_eval_config(eval_dataset=edata,
                                  eval_ses=eval_ses[i],
                                  parcellation=parcellation,
                                  crossed='half', # "half", # or None
                                  type='CondHalf',
                                  cerebellum=cerebellum,
                                  add_rest=add_rest[i],
                                  std_cortex=std_cortex[i],
                                  std_cerebellum='global',
                                  subj_list=list(T.participant_id),
                                  cortical_act=cortical_act)
      
      model_config = rm.get_model_config(dataset=edata,
                                         subj_list=T.participant_id,
                                         model=model,
                                         cerebellum=cerebellum)

      # sum over the models of other datasets
      coef_to_sum = coef_list.copy()
      if edata in train_datasets:
         indx = train_datasets.index(edata)
         coef_to_sum[indx] = np.zeros_like(coef_to_sum[indx])

      fuse_model = getattr(c_model, method)()
      setattr(fuse_model,'coef_',np.nansum(coef_to_sum, axis=0))
      if save_model:
         ev_dscode = gl.dscode[gl.datasets.index(edata)]
         train_dscode = ''.join(gl.dscode).replace(ev_dscode, '')
         if 'Ht' in train_dscode:
            train_dscode = train_dscode.replace('Ht', '')
         mname = f"{train_dscode}_{parcellation}_{method}"
         model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
         fname = model_path + f"/" + mname + f"_{model}"
         cio.save_model(fuse_model, info, fname)

      model_config['model'] = [fuse_model]
      info['extension'] = eval_id
      info['logalpha'] = logalpha
      model_config['train_info'] = [info]
      df, _ = rm.eval_model(None,None,eval_config, model_config)
      df['train_dataset'] = ['Fusion'] * df.shape[0]
      df['model'] = ['lodo']*df.shape[0]
      df['logalpha'] = [logalpha]*df.shape[0]

      all_df = pd.concat([all_df, df], ignore_index=True)

   save_path = gl.conn_dir+ f"/{cerebellum}/eval"
   file_name = save_path + f"/all_{method}_{eval_id}.tsv"
   all_df.to_csv(file_name, index = False, sep='\t')


def fuse_voxel_lodo(train_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    train_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    eval_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    eval_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    logalpha=[6, 6, 6, 2, 2, 8, 6],
                    method='L2reghalf',
                    parcellation='Icosahedron1002',
                    cerebellum='MNISymC3',
                    eval_id="Fus-lodo-voxel"):
   coef_list = []
   rel_map_list = []
   for i, (la, tdata, tses) in enumerate(zip(logalpha, train_datasets, train_ses)):
      print(f'Loading avg-half model for {tdata} - {tses}')
      mname = f"{tdata}_{tses}_{parcellation}_{method}"
      model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
      m = mname + f"_A{la}_avg-half"
      fname = model_path + f"/{m}"
      conn_mo, info = cio.load_model(fname)

      _, R_vox = ev.calculate_R(conn_mo.coef_1.T, conn_mo.coef_2.T)
      rel_map_list.append(2*R_vox / (R_vox + 1))
      coef_list.append((conn_mo.coef_1 + conn_mo.coef_2) / 2)
      

   all_df = pd.DataFrame()
   for i, edata in enumerate(eval_datasets):
      print(f'\nVoxel fusing models for {edata} ...')
      T = fdata.get_dataset_class(gl.base_dir, dataset=edata).get_participants()
      eval_config = rm.get_eval_config(eval_dataset=edata,
                                  eval_ses=eval_ses[i],
                                  parcellation=parcellation,
                                  crossed='half', # "half", # or None
                                  type='CondRun',
                                  cerebellum=cerebellum,
                                  add_rest=True,
                                  std_cortex='global' if edata=='Somatotopic' or edata=='WMFS' else 'parcel',
                                  std_cerebellum='global',
                                  subj_list=list(T.participant_id),
                                  cortical_act='avg')

      model_config = rm.get_model_config()
                                        
      # sum over the models of other datasets
      coef_to_sum = np.stack(coef_list, axis=0)
      rel_map_to_sum = np.stack(rel_map_list, axis=0)
      if edata in train_datasets:
         indx = train_datasets.index(edata)
         coef_to_sum[indx] = np.zeros_like(coef_to_sum[indx])     # shape: (n_datasets, n_voxels, n_regions)
         rel_map_to_sum[indx] = np.zeros_like(rel_map_to_sum[indx])

      # Normalize rel_map_to_sum columns to sum to 1
      rel_map_to_sum /= np.nansum(rel_map_to_sum, axis=0, keepdims=True)   # shape: (n_datasets, n_voxels)

      # Get the model for the fused dataset
      voxel_fuse_model = getattr(c_model, method)()
      wCoef = np.nansum(coef_to_sum * rel_map_to_sum[:, :, np.newaxis], axis=0)  # shape: (n_voxels, n_regions)
      setattr(voxel_fuse_model, 'coef_', wCoef)

      # Evaluate
      model_config['model'] = [voxel_fuse_model]
      info['extension'] = eval_id
      info['logalpha'] = logalpha
      model_config['train_info'] = [info]
      df, _ = rm.eval_model(None, None, eval_config, model_config)
      df['train_dataset'] = ['Fusion'] * df.shape[0]
      df['model'] = ['voxel-fusion']*df.shape[0]
      df['logalpha'] = [logalpha]*df.shape[0]

      all_df = pd.concat([all_df, df], ignore_index=True)

   # Save the results
   save_path = gl.conn_dir+ f"/{cerebellum}/eval"
   file_name = save_path + f"/all_{method}_{eval_id}.tsv"
   all_df.to_csv(file_name, index = False, sep='\t')


def fuse_all_models(train_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    train_ses=['all', 'ses-localizer', 'all', 'all', 'all', 'all', 'all'],
                    logalpha=[6, 6, 4, 4, 2, 8, 6],
                    weight=[1, 1, 1, 1, 1, 1, 1],
                    method='L2reghalf',
                    model='avg',
                    parcellation='Icosahedron1002',
                    cerebellum='MNISymC3',
                    fuse_id='Fus-all',
                    save=True):

   # Load all avg models of datasets
   coef_list = []
   for i, (la, tdata, tses) in enumerate(zip(logalpha, train_datasets, train_ses)):
      print(f'Loading avg model for {tdata} - {tses}')
      mname = f"{tdata}_{tses}_{parcellation}_{method}"
      model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
      fname = model_path + f"/{mname}_A{la}_{model}"
      conn_mo, conn_inf = cio.load_model(fname)
      coef_list.append(conn_mo.coef_)

   # Average datasetmeans
   Coef = np.stack(coef_list,axis=0)
   weight_norm = np.sqrt(np.nansum((Coef**2),axis=(1,2),keepdims=True))
   wCoef = Coef/weight_norm
   wCoef = wCoef * np.array(weight).reshape(-1,1,1)

   fuse_model = getattr(c_model, method)()
   setattr(fuse_model, 'coef_', np.nanmean(wCoef, axis=0))

   # Make the info
   fuse_info = conn_inf.copy()
   fuse_info['train_dataset'] = f'Fusion-{model}' 
   fuse_info['extension'] = fuse_id
   fuse_info['logalpha'] = logalpha
   
   # Save the model
   if save:
      mname = f"Fusion_{parcellation}_{method}"
      model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
      if os.path.exists(model_path) == False:
         os.makedirs(model_path)
      cio.save_model(fuse_model, fuse_info, model_path + f"/{mname}_{fuse_id}")

   return fuse_model, fuse_info


def train_global_model(train_dscode=''.join(gl.dscode),
                       method='L2reg',
                       cerebellum='MNISymC3',
                       mname=None,
                       logalpha_list=[0, 2, 4, 6, 8, 10, 12]):
   """Train a global model for the given dataset and session by concatenating dataset models."""

   if mname is None:
      mname = f"{train_dscode}_Icosahedron1002_{method}"

   config = rm.get_train_config(train_dataset = train_dscode,
                                method = method,
                                log_alpha = logalpha_list,
                                cerebellum = cerebellum,
                                validate_model = False)
   
   rm.train_global_model(config, mname=mname)


def eval_global_model(train_dscode='MdWfIbDeNiSoScLa',
                      eval_dataset='HCPur100',
                      parcellation='Icosahedron1002',
                      cerebellum='MNISymC3',
                      method='L2reg',
                      mname=None,
                      logalpha_list=[0, 2, 4, 6, 8, 10, 12],
                      append=False,
                      eval_id='MdWfIbDeNiSoScLa-global-Cavg'):
   """Evaluate a global model for the given dataset and session"""
   
   if mname is None:
      mname = f"{train_dscode}_Icosahedron1002_{method}"
   
   indx = gl.datasets.index(eval_dataset)

   eval_config = rm.get_eval_config(eval_dataset=eval_dataset,
                                    eval_ses=gl.sessions[indx],
                                    add_rest=gl.add_rest[indx],
                                    std_cortex=gl.std_cortex[indx],
                                    cortical_act='avg',
                                    cerebellum=cerebellum)
   
   model_config = rm.get_model_config(dataset=train_dscode,
                                      cerebellum=cerebellum)
   
   df_all = pd.DataFrame()
   mname = f"{train_dscode}_{parcellation}_{method}"
   model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
   for la in logalpha_list:
      m = mname + f"_A{la}_global"
      fname = model_path + f"/{m}"
      conn_mo, info = cio.load_model(fname)
      model_config['model'] = [conn_mo]
      model_config['logalpha'] = la
      info['extension'] = eval_id
      model_config['train_info'] = [info]

      df, _ = rm.eval_model(None, None, eval_config, model_config)
      df['model'] = ['global']*df.shape[0]

      df_all = pd.concat([df_all, df], ignore_index=True)

   save_path = gl.conn_dir+ f"/{cerebellum}/eval"
   if not os.path.isdir(save_path):
      os.mkdir(save_path)
   else:
      pass
   ename = eval_config['eval_dataset']
   if eval_config['eval_ses'] != 'all':
      ses_code = eval_config['eval_ses'].split('-')[1]
      ename = eval_config['eval_dataset'] + ses_code
   file_name = save_path + f"/{ename}_{method}_{eval_id}.tsv"
   if os.path.isfile(file_name) & append:
      dd = pd.read_csv(file_name, sep='\t')
      df_all = pd.concat([dd, df_all], ignore_index=True)
   df_all.to_csv(file_name, index = False, sep='\t')


def eval_mix_model(train_dscode='WfIbDeNiSoScLa',
                   eval_dataset='MDTB',
                   eval_ses='all',
                   add_rest=False,
                   std_cortex='parcel',
                   std_cerebellum='global',
                   run='all',
                   parcellation='Icosahedron1002',
                   cerebellum='MNISymC3',
                   method='L2reg',
                   crossed='half',
                   eval_type='CondHalf',
                   global_logalpha=8,
                   logalpha_list=[8],
                   cond_num='rnd_eval',
                   subj_list='all',
                   cortical_act='ind',
                   eval_id='WfIbDeNiSoScLa-mix-Cind',
                   dir_extra='_CV',
                   mix_params=np.linspace(0,100,11),
                   append=False):
   
   # Load the fusion lodo model
   print(f'Loading Fusion global model for {train_dscode}')
   mname = f"{train_dscode}_{parcellation}_{method}"
   model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
   fname = model_path + f"/{mname}_A{global_logalpha}_global"
   fuse_mo, _ = cio.load_model(fname)

   eval_config = rm.get_eval_config(eval_dataset=eval_dataset,
                                    eval_ses=eval_ses,
                                    run=run,
                                    cond_num=cond_num,
                                    parcellation=parcellation,
                                    crossed=crossed, # "half", # or None
                                    type=eval_type,
                                    cerebellum=cerebellum,
                                    splitby=None,
                                    add_rest=add_rest,
                                    std_cortex=std_cortex,
                                    std_cerebellum=std_cerebellum,
                                    subj_list=subj_list,
                                    cortical_act=cortical_act)
   
   dirname=[]
   mname=[]
   for a in logalpha_list:
      dirname.append(f"{train_dataset}_{train_ses}_{eval_config['parcellation']}_{method}{dir_extra}")
      mname.append(f"{train_dataset}_{train_ses}_{eval_config['parcellation']}_{method}{dir_extra}_A{a}")

   for mix_p in mix_params:
      model_config = rm.get_model_config(dataset=eval_dataset,
                                       subj_list=subj_list,
                                       model='mix',
                                       cerebellum=cerebellum,
                                       mix_param=mix_p)
      model_config['mix_model'] = fuse_mo

      df, _ = rm.eval_model(dirname,mname,eval_config,model_config)
      df_all = pd.concat([df_all, df], ignore_index=True)

   save_path = gl.conn_dir+ f"/{cerebellum}/eval"
   if not os.path.isdir(save_path):
      os.mkdir(save_path)
   else:
      pass
   ename = eval_config['eval_dataset']
   if eval_config['eval_ses'] != 'all':
      ses_code = eval_config['eval_ses'].split('-')[1]
      ename = eval_config['eval_dataset'] + ses_code
   file_name = save_path + f"/{ename}_{method}_{eval_id}.tsv"
   if os.path.isfile(file_name) & append:
      dd = pd.read_csv(file_name, sep='\t')
      df_all = pd.concat([dd, df_all], ignore_index=True)
   df_all.to_csv(file_name, index = False, sep='\t')


if __name__ == "__main__":
   do_train = False
   do_eval = False
   do_region_eval = False
   do_loso_fuse = False
   do_lodo_fuse = False
   do_voxel_lodo_fuse = False
   do_fuse_all = False
   do_train_global = False
   do_eval_global = False
   do_fuse_lodo_mix = True
   method = 'L2reg'
   cereb_atlas = 'MNISymC3'
   global_best_la = [8, 6, 8, 8, 8, 4, 8, 8, 8]
   
   # models = ["loo", "bayes", "bayes_vox"]
   # models = ["ind"]
   # models = ["avg", "bayes"]
   # models = ["bayes"]
   # models = [["avg"], ["bayes"]]
   # models = [['avg']]
   # models = ['avg', 'loo']
   # models = ["loo", "bayes-loo"]
   # models = ['avg']
   # models = ['loo']
   # models = ['avg-half']
   # models = ['group']
   # models = ['group', 'avg']
   models = ['mix']; mix_params = np.linspace(0,100,11)

   train_types = {
      'MDTB':        ('all',                 False,   'parcel',   8),
      'Language':    ('ses-localizer',       False,   'parcel',   8),
      'Social':      ('ses-social',          False,   'parcel',   8),
      'WMFS':        ('all',                 True,    'global',   8),
      'Demand':      ('all',                 True,    'parcel',   8),
      'Somatotopic': ('all',                 True,    'global',   10),
      'Nishimoto':   ('all',                 False,   'parcel',   10),
      'IBC':         ('all',                 True,    'parcel',   6),
   }

   eval_types = {
      'MDTB':        ('all',                 False,   'parcel',   models),
      'Language':    ('ses-localizer',       False,   'parcel',   models),
      'Social':      ('ses-social',          False,   'parcel',   models),
      'WMFS':        ('all',                 True,    'global',   models),
      'Demand':      ('all',                 True,    'parcel',   models),
      'Somatotopic': ('all',                 True,    'global',   models),
      'Nishimoto':   ('all',                 False,   'parcel',   models),
      'IBC':         ('all',                 True,    'parcel',   models),
      'HCPur100':    ('all',                 True,    'parcel',   models),
   }

   for train_dataset, (train_ses, add_rest, std_cortex, best_la) in train_types.items():
      if do_train:
         if models[0] == 'mix':
            cond_num = 'rnd_train'
         else:
            cond_num = 'all'

         print(f'Train: {train_dataset} - individual')
         train_models(dataset=train_dataset, train_ses=train_ses, add_rest=add_rest, std_cortex=std_cortex,
                      method=method, cerebellum=cereb_atlas, cond_num=cond_num,
                      mname=f"{train_dataset}_{train_ses}_Icosahedron1002_{method}_CV",
                      logalpha_list=[best_la])

         # print(f'Train: {train_dataset} - avg')
         # avrg_model(train_data=train_dataset, train_ses=train_ses, method=method, cerebellum=cereb_atlas,)
                  #   avrg_mode='avg-half')

         # print(f'Train: {train_dataset} - group')
         # train_models(dataset=train_dataset, train_ses=train_ses, add_rest=add_rest, std_cortex=std_cortex,
         # method=method, cerebellum=cereb_atlas,
         #              cortical_cerebellar_act='avg',)

         # print(f'Train: {train_dataset} - bayes')
         # bayes_avrg_model(train_data=train_dataset, train_ses=train_ses, method=method, cerebellum=cereb_atlas)

      for eval_dataset, (eval_ses, add_rest, std_cortex, models) in eval_types.items():
         if do_eval:
            for model in models:
               if (train_dataset == eval_dataset) & (isinstance(model, list)) & (model[0]=='avg'):
                  continue
               elif (train_dataset == eval_dataset) & ((model=='group') | (model=='avg')):
                  continue
               elif (train_dataset != eval_dataset) & (("loo" in model) | (model=='mix')):
                  continue

               print(f'Train: {train_dataset} - Eval: {eval_dataset} - {model}')
               if isinstance(model, list):
                  eval_id = train_dataset+"-"+str(model[0])
               else:
                  eval_id = train_dataset+"-"+model
               
               if model == 'ind':
                  D = fdata.get_dataset_class(gl.base_dir, train_dataset)
                  T = D.get_participants()
                  eval_id = train_dataset+"-"+model
                  model = list(T['participant_id'])
               
               if model=='mix':
                  for p in mix_params:
                     eval_models(train_dataset=train_dataset, train_ses=train_ses, eval_dataset=[eval_dataset], eval_ses=eval_ses,
                              add_rest=add_rest, std_cortex=std_cortex, model=model, method=method,
                              cortical_act='avg', eval_id=eval_id+"-CV-Cavg", logalpha_list=[best_la],
                              mix_param=p, cond_num='rnd_eval', dir_extra='_CV', append=True)
               else:
                  eval_models(train_dataset=train_dataset, train_ses=train_ses, eval_dataset=[eval_dataset], eval_ses=eval_ses,
                              add_rest=add_rest, std_cortex=std_cortex, model=model, method=method,
                              cortical_act='avg', eval_id=eval_id+"-Cavg")
         
         if do_region_eval:
            if train_dataset != eval_dataset:
               print(f'Eval: {train_dataset} - {eval_dataset} - region')
               eval_id = train_dataset+"-avg-region"
               eval_region_models(train_dataset=train_dataset, train_ses=train_ses, eval_dataset=[eval_dataset], eval_ses=eval_ses,
                                 add_rest=add_rest, std_cortex=std_cortex, method=method, ext_list=[best_la], eval_id=eval_id)
            else:
               print(f'Eval: {train_dataset} - {eval_dataset} - region')
               eval_id = train_dataset+"-loo-region"
               eval_region_models(train_dataset=train_dataset, train_ses=train_ses, eval_dataset=[eval_dataset], eval_ses=eval_ses,
                                 add_rest=add_rest, std_cortex=std_cortex, model='loo', method=method, ext_list=[best_la], eval_id=eval_id)
               
   if do_loso_fuse:
      for model in models:
         eval_id = "Fus06-bestSTD-voxel-WTA"# + model
         fuse_models_loso(train_datasets=list(train_types.keys()),
                         train_ses=[value[0] for value in train_types.values()],
                         eval_datasets=list(eval_types.keys()),
                         eval_ses=[value[0] for value in eval_types.values()],
                         logalpha=[value[1] for value in train_types.values()],
                        #  weight=[1]*len(train_types),
                         weight='voxel-rel-WTA',
                         model=model,   # "avg" or "bayes"
                         method=method,
                         parcellation='Icosahedron1002',
                         cerebellum=cereb_atlas,
                         eval_id=eval_id)
         
   if do_lodo_fuse:
      for model in models:
         eval_id = "Fus-lodo-" + model + "-Cavg"
         fuse_models_lodo(train_datasets=list(train_types.keys()),
                           train_ses=[value[0] for value in train_types.values()],
                           eval_datasets=list(eval_types.keys()),
                           eval_ses=[value[0] for value in eval_types.values()],
                           add_rest=[value[1] for value in eval_types.values()],
                           std_cortex=[value[2] for value in eval_types.values()],
                           logalpha=[value[3] for value in train_types.values()],
                           model=model,   # "avg" or "bayes"
                           method=method,
                           parcellation='Icosahedron1002',
                           cerebellum=cereb_atlas,
                           cortical_act='avg',
                           eval_id=eval_id,
                           save_model=True)

   if do_voxel_lodo_fuse:
      eval_id = "Fus-lodo-voxel"
      fuse_voxel_lodo(train_datasets=list(train_types.keys()),
                        train_ses=[value[0] for value in train_types.values()],
                        eval_datasets=list(eval_types.keys()),
                        eval_ses=[value[0] for value in eval_types.values()],
                        logalpha=[value[3] for value in train_types.values()],
                        method=method,
                        parcellation='Icosahedron1002',
                        cerebellum=cereb_atlas,
                        eval_id=eval_id)
      
   if do_fuse_all:
      for model in models:
         print(f'Fusing all models for {model}')
         fuse_id = f'Fus-{model}'
         fuse_all_models(train_datasets=list(train_types.keys()),
                        train_ses=[value[0] for value in train_types.values()],
                        logalpha=[value[3] for value in train_types.values()],
                        weight=[1]*len(train_types),
                        method=method,
                        model=model,
                        parcellation='Icosahedron1002',
                        cerebellum=cereb_atlas,
                        fuse_id=fuse_id)
   
   if do_train_global:
      print(f'Training global models')
      for ds in list(train_types.keys()):
         d = gl.datasets.index(ds)
         train_dscode = ''.join(gl.dscode[:d]+gl.dscode[d+1:])
         train_dscode = train_dscode.replace('Ht', '')
         print(f'{train_dscode}:')
         # train_dscode = ''.join(gl.dscode).replace('Ht', '')
         train_global_model(train_dscode=train_dscode,
                           method=method,
                           cerebellum=cereb_atlas,
                           mname=None,
                           logalpha_list=[0, 2, 4, 6, 8, 10, 12])
            
   if do_eval_global:
      for ds in list(train_types.keys()):
         print(f'Evaluating global model for {ds}')
         d = gl.datasets.index(ds)
         train_dscode = ''.join(gl.dscode[:d]+gl.dscode[d+1:]).train_dscode.replace('Ht', '')
         # train_dscode = ''.join(gl.dscode).replace('Ht', '')
         eval_id = train_dscode + '-global-Cavg'
         eval_global_model(train_dscode=train_dscode,
                           eval_dataset=ds,
                           cerebellum=cereb_atlas,
                           method=method,
                           logalpha_list=[0, 2, 4, 6, 8, 10, 12],
                           eval_id=eval_id)
               
   if do_fuse_lodo_mix:
      for i, ds in enumerate(list(train_types.keys())):
         print(f'Mixing Fusion model with {ds}')
         d = gl.datasets.index(ds)
         train_dscode = ''.join(gl.dscode[:d]+gl.dscode[d+1:]).replace('Ht', '')
         eval_id = train_dscode + '-mix-Cind'
         eval_mix_model(train_dscode=train_dscode,
                        eval_dataset=ds,
                        cerebellum=cereb_atlas,
                        method=method,
                        logalpha=global_best_la[i],
                        cond_num='rnd_eval',
                        cortical_act='ind',
                        dir_extra='_CV',
                        mix_params=mix_params,
                        eval_id=eval_id)
         

