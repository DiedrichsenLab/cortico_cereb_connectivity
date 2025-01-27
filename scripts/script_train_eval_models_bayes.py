"""
script for training models
@ Ali Shahbazi, Ladan Shahshahani, Joern Diedrichsen
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import re
import sys
from collections import defaultdict
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as cm
import cortico_cereb_connectivity.cio as cio
import json

def train_models(logalpha_list = [2, 4, 6, 8, 10, 12],
                 crossed = "half",
                 type = "CondHalf",
                 train_ses = 'ses-s1',
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
   dataset = fdata.get_dataset_class(gl.base_dir,
                                    dataset=config["train_dataset"])
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
               train_ses= "ses-s1",
               train_run='all',
               parcellation = 'Icosahedron1002',
               method='L2reg',
               type='CondHalf',
               cerebellum='MNISymC3',
               parameters=['coef_','coef_var'],
               avrg_mode = 'avrg_sep',
               avg_id = 'avg'):

   mname_base = f"{train_data}_{type}_{train_ses}_run-{train_run}_{parcellation}_{method}"
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


def eval_models(ext_list = [2, 4, 6, 8, 10, 12],
                train_dataset = "MDTB",
                train_ses = "ses-s1",
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
      config = rm.get_eval_config(eval_dataset = ed,
                                 eval_ses = eval_ses,
                                 eval_run=eval_run,
                                 parcellation=parcellation,
                                 crossed = crossed, # "half", # or None
                                 type = eval_type[i],
                                 cerebellum=cerebellum,
                                 splitby = None,
                                 add_rest = add_rest,
                                 std_cortex=std_cortex,
                                 std_cerebellum=std_cerebellum,
                                 subj_list = subj_list,
                                 model_subj_list = model_subj_list,
                                 model = model,
                                 mix_param = mix_param)

      dirname=[]
      mname=[]
      for a in ext_list:
         dirname.append(f"{train_dataset}_{config['type']}_{train_ses}_run-{train_run}_{config['parcellation']}_L2reg")
         mname.append(f"{train_dataset}_{config['type']}_{train_ses}_run-{train_run}_{config['parcellation']}_L2reg_A{a}")

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


if __name__ == "__main__":
   eval_across_dataset = True
   do_train = False
   models = ["loo", "bayes", "bayes_vox"]
   # models = ["ind"]

   train_types = {
      'MDTB':        ('ses-s1'),
      'WMFS':        ('ses-01'),
      'Nishimoto':   ('ses-01'),
   }

   eval_types = {
      'MDTB':        ('ses-s2',     models),
      'WMFS':        ('ses-02',     models),
      'Nishimoto':   ('ses-02',     models),
   }

   for train_dataset, (train_ses) in train_types.items():
      if do_train:
         train_models(dataset=train_dataset, train_ses=train_ses)
         avrg_model(train_data=train_dataset, train_ses=train_ses)

      for eval_dataset, (eval_ses, models) in eval_types.items():
         if not eval_across_dataset:
            if train_dataset != eval_dataset:
               continue

         for model in models:
            print(f'Train: {train_dataset} - Eval: {eval_dataset} - {model}')
            eval_id = train_dataset+"-"+model
            if model == 'ind':
               D = fdata.get_dataset_class(gl.base_dir, train_dataset)
               T = D.get_participants()
               eval_id = train_dataset+"-"+model
               model = list(T['participant_id'])
            
            eval_models(train_dataset=train_dataset, train_ses=train_ses, eval_dataset=[eval_dataset], eval_ses=eval_ses,
                        model=model, ext_list=[2, 4, 6, 8, 10, 12], eval_id=eval_id)

