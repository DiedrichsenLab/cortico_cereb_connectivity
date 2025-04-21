"""
script for training models
@ Ali Shahbazi
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.scripts.script_fuse_models as sfm
import json

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
                 crossed = "half",
                 type = "CondHalf",
                 train_ses = 'all',
                 dataset = "MDTB",
                 add_rest = True,
                 parcellation = "Icosahedron1002",
                 subj_list = "all",
                 cerebellum='MNISymC3',
                 method = "L2reg",
                 validate_model = False):

   config = rm.get_train_config(log_alpha=logalpha_list,
                                 crossed=crossed,
                                 type=type,
                                 subj_list=subj_list,
                                 cerebellum=cerebellum,
                                 parcellation=parcellation,
                                 train_dataset=dataset,
                                 method=method,
                                 train_ses=train_ses,
                                 train_run='all',
                                 add_rest=add_rest,
                                 validate_model=validate_model,
                                 std_cortex='parcel',
                                 std_cerebellum='global',
                                 append=False)
   
   dataset = fdata.get_dataset_class(gl.base_dir, dataset=config["train_dataset"])
   # get the list of trained connectivity models and training summary
   T = dataset.get_participants()
   if subj_list is None:
      config["subj_list"] = T.participant_id
   elif subj_list=='all':
      config["subj_list"] = T.participant_id
   elif isinstance(subj_list[0],str):
      config["subj_list"] = subj_list
   else:
      config["subj_list"] = T.participant_id.iloc[subj_list]

   config, conn_list, df_tmp = rm.train_model(config)
   return df_tmp


def avrg_model(logalpha_list = [2, 4, 6, 8, 10, 12],
               train_data = "MDTB",
               train_ses= "all",
               train_run='all',
               parcellation = 'Icosahedron1002',
               method='L2reg',
               type='CondHalf',
               cerebellum='MNISymC3',
               parameters=['coef_'],
               avrg_mode = 'avrg_sep',
               avg_id = 'avg'):

   mname_base = f"{train_data}_{train_ses}_{parcellation}_{method}"
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


def bayes_avrg_model(logalpha_list = [2, 4, 6, 8, 10, 12],
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


def eval_models(ext_list = [2, 4, 6, 8, 10, 12],
                train_dataset = "MDTB",
                train_ses = "all",
                train_run = 'all',
                method = "L2reg",
                parcellation = "Icosahedron1002",
                cerebellum='MNISymC3',
                eval_dataset = ["MDTB"],
                eval_type = ["CondHalf"],
                eval_ses  = "ses-s2",
                eval_run='all',
                eval_id = 'MD_s1',
                crossed = 'half',
                add_rest = True,
                std_cortex = 'parcel',
                std_cerebellum = 'global',
                subj_list = "all",
                model_subj_list = "all",
                model = 'avg',
                mix_param = [],
                append = False
                ):
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

      dirname=[]
      mname=[]
      for a in ext_list:
         dirname.append(f"{train_dataset}_{train_ses}_{config['parcellation']}_{method}")
         mname.append(f"{train_dataset}_{train_ses}_{config['parcellation']}_{method}_A{a}")

      df, df_voxels = rm.eval_model(dirname,mname,config)
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
         df = pd.concat([dd, df], ignore_index=True)
         # df = df.append(dd,ignore_index=True) # pd.append is deprecated
      df.to_csv(file_name, index = False, sep='\t')
   return df,df_voxels


def fuse_models_loo(train_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    train_ses=['all', 'ses-localizer_cond', 'all', 'all', 'all', 'all', 'all'],
                    eval_datasets=['MDTB', 'Language', 'WMFS', 'Demand', 'Somatotopic', 'Nishimoto', 'IBC'],
                    eval_ses=['all', 'ses-localizer_cond', 'all', 'all', 'all', 'all', 'all'],
                    logalpha=[8, 8, 8, 8, 8, 10, 8],
                    weight=[1, 1, 1, 1, 1, 1, 1],
                    model='avg',   # "avg" or "bayes"
                    method='L2reghalf',
                    parcellation='Icosahedron1002',
                    cerebellum='MNISymC3',
                    eval_id='Fus-avg'):
    
   # First load all basic dataset models
   coef_list = []
   num_subj = []
   for i, (la,tdata, tses) in enumerate(zip(logalpha, train_datasets, train_ses)):
      mname = f"{tdata}_{tses}_{parcellation}_{method}"
      model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
      m = mname + f"_A{la}_{model}"
      fname = model_path + f"/{m}"
      conn_mo, info = cio.load_model(fname)
      coef_list.append(conn_mo.coef_)

      T = fdata.get_dataset_class(gl.base_dir, dataset=tdata).get_participants()
      # Get the number of subjects in the dataset
      num_subj.append(len(T.participant_id))      

   for i, edata in enumerate(eval_datasets):
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
                                  std_cortex='parcel' if edata=='MDTB' else 'global',
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
      else:
         loo_weights = weight

      # Get the model for the fused dataset
      loo_fuse_models = []
      Coef = np.stack(coef_list,axis=0)
      for s,m in enumerate(ind_models):
         print('Fuse model for subject',T.participant_id.iloc[s])
         Coef_dup = Coef.copy()
         # Find where train_data is equal to ed
         Coef_dup[indx,:,:] = m.coef_
      
         weight_norm = np.sqrt(np.nansum((Coef_dup**2),axis=(1,2),keepdims=True))
         wCoef = Coef_dup/weight_norm
         wCoef = wCoef * np.array(loo_weights).reshape(-1,1,1)
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
      df['train_dataset'] = 'Fusion'*df.shape[0]
      df['model'] = ['loo']*df.shape[0]
      df['logalpha'] = [logalpha]*df.shape[0]
      df['weight'] = [loo_weights]*df.shape[0]
      df.to_csv(file_name, index = False, sep='\t')
      

if __name__ == "__main__":
   do_train = False
   do_eval = False
   do_fuse = False
   method = 'L2reg'
   
   # models = ["loo", "bayes", "bayes_vox"]
   # models = ["ind"]
   # models = ["avg", "bayes"]
   # models = [["avg"], ["bayes"]]
   # models = [['avg']]
   # models = ["loo", "bayes-loo"]
   models = ['avg']
   # models = ['loo']

   train_types = {
      'MDTB':        ('all',                 8),
      # 'Language':    ('ses-localizer_cond',  8),
      'WMFS':        ('all',                 8),
      'Demand':      ('all',                 8),
      'Somatotopic': ('all',                 8),
      'Nishimoto':   ('all',                 10),
      'IBC':         ('all',                 8),
   }

   eval_types = {
      'MDTB':        ('all',                 models),
      # 'Language':    ('ses-localizer_cond',  models),
      'WMFS':        ('all',                 models),
      'Demand':      ('all',                 models),
      'Somatotopic': ('all',                 models),
      'Nishimoto':   ('all',                 models),
      'IBC':         ('all',                 models),
   }

   for train_dataset, (train_ses, best_la) in train_types.items():
      if do_train:
         print(f'Train: {train_dataset} - individual')
         train_models(dataset=train_dataset, train_ses=train_ses, method=method)
         print(f'Train: {train_dataset} - avg')
         avrg_model(train_data=train_dataset, train_ses=train_ses, method=method)
         # print(f'Train: {train_dataset} - bayes')
         # bayes_avrg_model(train_data=train_dataset, train_ses=train_ses, method=method)

      if do_eval:
         for eval_dataset, (eval_ses, models) in eval_types.items():
            for model in models:
               if (train_dataset == eval_dataset) & isinstance(model, list):
                  continue
               elif (train_dataset != eval_dataset) & ("loo" in model):
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
               
               eval_models(train_dataset=train_dataset, train_ses=train_ses, eval_dataset=[eval_dataset], eval_ses=eval_ses,
                           model=model, method=method, ext_list=[2, 4, 6, 8, 10, 12], eval_id=eval_id)
               
   if do_fuse:
      for model in models:
         eval_id = "Fus06-bestSTD-" + model
         fuse_models_loo(train_datasets=list(train_types.keys()),
                         train_ses=[value[0] for value in train_types.values()],
                         eval_datasets=list(eval_types.keys()),
                         eval_ses=[value[0] for value in eval_types.values()],
                         logalpha=[value[1] for value in train_types.values()],
                         weight=[1]*len(train_types),
                        #  weight='num_subj',
                        #  model=model,   # "avg" or "bayes"
                         method=method,
                         parcellation='Icosahedron1002',
                         cerebellum='MNISymC3',
                         eval_id=eval_id)

   # ED=['Demand','IBC','MDTB','Somatotopic','WMFS','Nishimoto']
   # ED=['Somatotopic']
   # for ed in ED:
   #    sfm.eval_fusion_loo(eval_data=ed,weight=[1,0,1,1,1,1,1],eval_id='Fu06-loo-replication')
