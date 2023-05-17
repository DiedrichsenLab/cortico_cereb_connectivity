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
# sys.path.append('../cortico-cereb_connectivity')
# sys.path.append('..')
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as cm

def train_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                 crossed = "half", 
                 type = "CondHalf",
                 train_ses = 'all',
                 dataset = "MDTB",
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
               cerebellum='SUIT3'):

   mname = f"{train_data}_{train_ses}_{parcellation}_{method}"
   for la in logalpha_list: 
      if la is not None:
         # Generate new model
         mname_ext = f"A{la}"
      else:
         mname_ext = f""

      rm.calc_avrg_model(train_data,mname,mname_ext,cerebellum=cerebellum)



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
                model = 'avg',
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
       append (bool): Append to existing tsv file? Defaults to False.

   Returns:
       _type_: _description_
   """
   for i,ed in enumerate(eval_dataset):
      config = rm.get_eval_config(eval_dataset = ed,
                                 eval_ses = eval_ses, 
                                 parcellation = parcellation,
                                 crossed = "half", # or None
                                 type = eval_type[i],
                                 cerebellum=cerebellum,
                                 splitby = None,
                                 model=model)

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
         df = df.append(dd,ignore_index=True)
      df.to_csv(file_name, index = False, sep='\t')
   return df,df_voxels

if __name__ == "__main__":
   # train_models(train_ses = 'all',dataset = "HCP",type='Tseries',crossed=None,cerebellum='SUIT3',validate_model=False,logalpha_list = [-4,-2])
   # avrg_model(train_data = "HCP",train_ses= "all",logalpha_list = [-2])
   ED=["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC"]
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf", "CondHalf", "CondHalf"]
   # 
   #    train_models(train_ses = 'all',dataset = ed,cerebellum='MNISymC2')
      # avrg_model(train_data = ed,train_ses= "all",cerebellum='MNISymC2')
   # eval_models(eval_dataset = ED, eval_type = ET,train_dataset="HCP", train_ses="all",eval_id = 'Hc')
   # eval_models(eval_dataset = ED, train_dataset="Demand", train_ses="all",eval_id = 'De')
   # eval_models(eval_dataset = ED, train_dataset="Nishimoto", train_ses="all",eval_id = 'Ni')
   # eval_models(eval_dataset = ED, train_dataset="WMFS", train_ses="all",eval_id = 'Wm')
   # eval_models(eval_dataset = ED, train_dataset="Somatotopic", train_ses="all",eval_id = 'So',eval_type=ET)

   # for ed,et in zip(ED,ET):
   # train_models(dataset = 'HCP',method='L2noscale',
   #                train_ses = 'all',cerebellum='SUIT3',validate_model=False,
   #                 type = 'TSeries',crossed=None,
   #                 logalpha_list = [-4,-2,0,2,4,6,8,10,12],
   #                 subj_list=np.arange(63,100))
   eval_models(train_dataset="MDTB", train_ses="all",
             eval_dataset = ['MDTB'], eval_ses="ses-s2",
             eval_id = 'Md-avg',model='avg',ext_list=[8])
   # eval_models(train_dataset="MDTB", train_ses="all",
   #           eval_dataset = ['MDTB'], eval_ses="ses-s2",
   #           eval_id = 'Md-loo',model='loo',ext_list=[8])
   # All individual models
   # D = fdata.get_dataset_class(gl.base_dir,'MDTB')
   # Get participants info
   # T = D.get_participants()
   # subj_name = list(T['participant_id'])
   # eval_models(train_dataset="MDTB", train_ses="all",
   #          eval_dataset = ['MDTB'], eval_ses="ses-s2",
   #          eval_id = 'Md-ind',model=subj_name,ext_list=[8])
   #    avrg_model(train_data = 'MDTB', method='L2regression',train_ses= "all",cerebellum='SUIT3',logalpha_list = [0,2,4,6,8,10,12])
   # avrg_model(train_data = 'Somatotopic', method='L2noscale',train_ses= "all",cerebellum='SUIT3',logalpha_list = [-4,-2,0,2,4,6,8,10])
   # eval_models(eval_dataset = ['MDTB'], eval_type = ['CondHalf'],
   #               train_dataset="Somatotopic", train_ses="all",
   #               eval_id = 'So',model='avg',
   #               method='L2noscale',cerebellum='SUIT3',
   #               ext_list=['A-4','A-2','A0','A2','A4','A6','A8','A10'])
   # df=rm.comb_eval(models=['De','So'],methods=['L2regression','L2noscale'],# eval_data=['MDTB'])
   # pass
  # eval_models(eval_dataset = ['MDTB'], train_dataset="WMFS", train_ses="all",eval_id = 'Wm_loo',model='loo')
   # eval_models(eval_dataset = ['WMFS'], train_dataset="WMFS", train_ses="all",eval_id = 'Wm_loo',model='loo')
   # eval_models(eval_dataset = ['Nishimoto'], train_dataset="Nishimoto", train_ses="all",eval_id = 'Ni_loo',model='loo')
   # eval_models(eval_dataset = ['Demand'], train_dataset="Demand", train_ses="all",eval_id = 'De_loo',model='loo')
   # eval_models(eval_dataset = ['IBC'], train_dataset="IBC", train_ses="all",eval_id = 'Ib_loo',model='loo')
   # eval_models(eval_dataset = ['Somatotopic'], train_dataset="Somatotopic", train_ses="all",eval_id = 'So_loo',model='loo')
   # eval_models(eval_dataset = ED, eval_type=ET,train_dataset="HCP", train_ses="all",eval_id = 'Hc',method='L2noscale')
   # eval_models(eval_dataset = ['MDTB'], eval_type=['CondHalf'],train_dataset="MDTB", train_ses="all",eval_id = 'Md')
   # eval_models(eval_dataset = ED, eval_type=ET,train_dataset="Demand", train_ses="all",eval_id = 'De',method='L2noscale')
   # eval_models(eval_dataset = ED, eval_type=ET,train_dataset="Nishimoto", train_ses="all",eval_id = 'Ni',method='L2noscale')
   # eval_models(eval_dataset = ED, eval_type=ET,train_dataset="WMFS", train_ses="all",eval_id = 'Wm',method='L2noscale')
   # eval_models(eval_dataset = ED, eval_type=ET,train_dataset="Somatotopic", train_ses="all",eval_id = 'So',method='L2noscale')
   # eval_models(eval_dataset = ED, eval_type=ET,train_dataset="IBC", train_ses="all",eval_id = 'Ib',method='L2noscale')
   # for ed in ED:
   #    train_models(train_ses = 'all',dataset = ed,cerebellum='MNISymC2')
      # avrg_model(train_data = ed,train_ses= "all",cerebellum='MNISymC2')
   # eval_models(eval_dataset = ED, eval_type = ET,train_dataset="HCP", train_ses="all",eval_id = 'Hc')
   
   # train_models(train_ses = 'all',dataset = "MDTB")
   # avrg_model(train_data = "MDTB",train_ses= "all")
   # train_models(train_ses = 'all',dataset = "MDTB")
   # avrg_model(train_data = "MDTB",train_ses= "all")
   