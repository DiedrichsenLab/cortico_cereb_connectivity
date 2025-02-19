"""
script for training models
@ Ladan Shahshahani, Joern Diedrichsen Jan 30 2023 12:57
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
import json

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
                 crossed = "half",
                 type = "CondHalf",
                 train_ses = 'all',
                 dataset = "MDTB",
                 add_rest = True,
                 parcellation = "Icosahedron1002",
                 subj_list = "all",
                 cerebellum='SUIT3',
                 method = "L2regression",
                 validate_model = True):

   config = rm.get_train_config(log_alpha = logalpha_list,
                                crossed = crossed,
                                type = type,
                                cerebellum=cerebellum,
                                parcellation=parcellation,
                                train_dataset = dataset,
                                method = method,
                                train_ses=train_ses,
                                add_rest=add_rest,
                                validate_model=validate_model)
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

   config, conn_list, df_tmp =rm.train_model(config)
   return df_tmp

def avrg_model(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
               train_data = "MDTB",
               train_ses= "ses-s1",
               parcellation = 'Icosahedron1002',
               method='L2Regression',
               cerebellum='SUIT3',
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
      dd.io.save(model_path + f"/{mname_base}{mname_ext}_{avg_id}.h5",
         avrg_model, compression=None)
      with open(model_path + f"/{mname_base}{mname_ext}_{avg_id}.json", 'w') as fp:
         json.dump(info, fp, indent=4)


def eval_models(ext_list = [0, 2, 4, 6, 8, 10, 12],
                train_dataset = "MDTB",
                train_ses = "ses-s1",
                method = "L2regression",
                parcellation = "Icosahedron1002",
                cerebellum='SUIT3',
                eval_dataset = ["Demand"],
                eval_type = ["CondHalf"],
                eval_ses  = "all",
                eval_id = 'Md_s1',
                crossed = 'half',
                add_rest = False,
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
                                 parcellation = parcellation,
                                 crossed = crossed, # "half", # or None
                                 type = eval_type[i],
                                 cerebellum=cerebellum,
                                 splitby = None,
                                 add_rest = add_rest,
                                 subj_list = subj_list,
                                 model_subj_list = model_subj_list,
                                 model = model,
                                 mix_param = mix_param)

      dirname=[]
      mname=[]

      for a in ext_list:
         dirname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
         if a is None:
            mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
         if isinstance(a,int):
            mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_A{a}")
         elif isinstance(a,str):
            mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_{a}")

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

def train_all():
   ED=["MDTB","WMFS","Nishimoto","IBC",'Somatotopic','Demand','HCP'] #
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf","CondHalf", "CondHalf","Tseries"]
   for ed,et in zip(ED,ET):
      train_models(dataset = ed,method='L2regression',
                  train_ses = 'all',
                  cerebellum='SUIT3',
                  validate_model=False,
                  type = et,
                  crossed='half',
                  add_rest=True,
                  logalpha_list = [-4,-2,0,2,4,6,8,10,12])

def train_all_wta():
   ED=['HCP'] # ["MDTB","WMFS","Nishimoto","IBC",'Somatotopic','Demand','HCP'] #
   ET=['Tseries'] # ["CondHalf","CondHalf", "CondHalf", "CondHalf","CondHalf", "CondHalf","Tseries"]
   for ed,et in zip(ED,ET):
      if et=='Tseries':
         ar= False
         cr=None
      else:
         ar= True
         cr= 'half'
      train_models(dataset = ed,method='WTA',
                  train_ses = 'all',
                  cerebellum='SUIT3',
                  parcellation = "Icosahedron1002",
                  validate_model=False,
                  type = et,
                  crossed=cr,
                  add_rest=ar,
                  logalpha_list = [None])

def avrg_all():
   ED=["MDTB","WMFS", "Nishimoto", "IBC",'Somatotopic','Demand']
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf","CondHalf", "CondHalf"]
   for ed,et in zip(ED,ET):
      avrg_model(train_data = ed,

                 train_ses= "all",
                 cerebellum='SUIT3',
                 logalpha_list = [-4,-2,0,2,4,6,8,10,12])

def avrg_all_wta():
   ED=["MDTB","WMFS", "Nishimoto", "IBC",'Somatotopic','Demand']
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf","CondHalf", "CondHalf"]
   for ed,et in zip(ED,ET):
      avrg_model(train_data = ed,
               method='WTA',
                 train_ses= "all",
                 cerebellum='SUIT3',
                 logalpha_list = [None])

def eval_all():
   TD = ["HCP"]
   ED=["MDTB","WMFS", "Nishimoto", "IBC",'Somatotopic','Demand']
   tID = ["Hc"] # ['Md','Wm','Ni','Ib','So','De']
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf",'CondHalf','CondHalf']
   for td,tid in zip(TD,tID):
      eval_models(eval_dataset = ED, eval_type = ET,
                  crossed='half',
                  train_dataset=td,
                  add_rest=True,
                  ext_list = [-4,-2,0,2,4,6,8,10,12],
                  train_ses="all",eval_id = tid)

def eval_all_loo():
   ED=["MDTB","WMFS", "Nishimoto", "IBC",'Somatotopic','Demand']
   eID = ['Md-loo','Wm-loo','Ni-loo','Ib-loo','So-loo','De-loo']
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf",'CondHalf','CondHalf']
   for ed,et,eid in zip(ED,ET,eID):
      eval_models(eval_dataset = [ed], eval_type = [et],
                  crossed='half',
                  train_dataset=ed,
                  train_ses="all",
                  ext_list = [-4,-2,0,2,4,6,8,10,12],
                  eval_id = eid,
                  add_rest=True,
                  model='loo')

if __name__ == "__main__":
   """
   TD=["MDTB"] # ["MDTB","WMFS", "Nishimoto", "IBC"]
   tID = ['Md-rest']
   ED=["Demand","WMFS"]
   ET=["CondHalf","CondHalf"]
   for et,tid in zip(TD,tID):
      eval_models(eval_dataset = ED, eval_type = ET,
                  crossed='half',
                  train_dataset=et,
                  ext_list = [8],
                  add_rest=True,
                  train_ses="all",eval_id = tid)
   """
   # train_all()
   # avrg_all()
   #
   avrg_model(train_data = 'HCP',
               method='WTA',
                 train_ses= "all",
                 cerebellum='SUIT3',
                 logalpha_list = [None])
   # avrg_all_wta()