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
                 train_ses = 'ses-s1',
                 dataset = "MDTB"):
      
   config = rm.get_train_config(log_alpha = logalpha_list, 
                                crossed = crossed,
                                type = type, 
                                train_dataset = dataset,
                                train_ses=train_ses)
   dataset = fdata.get_dataset_class(gl.base_dir, 
                                    dataset=config["train_dataset"]) 
         # get the list of trained connectivity models and training summary
   T = dataset.get_participants()
   config["subj_list"] = T.participant_id[80:]
   config, conn_list, df_tmp =rm.train_model(config)



def avrg_model(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
            train_data = "MDTB",
            train_ses= "ses-s1",
            parcellation = 'Icosahedron1002',
            method='L2Regression'):
    mname = f"{train_data}_{train_ses}_{parcellation}_{method}"

    for la in logalpha_list: 
        mname_ext = f"A{la}"
        rm.calc_avrg_model(train_data,mname,mname_ext)


def eval_models(logalpha_list = [0, 2, 4, 6, 8, 10, 12], 
                type = "CondHalf",
                train_dataset = "MDTB",
                train_ses = "ses-s1",
                method = "L2Regression",
                parcellation = "Icosahedron1002", 
                eval_dataset = ["Demand"],
                eval_type = ["CondHalf"],
                eval_ses  = "all",
                eval_id = 'Md_s1' 
                ):
   for i,ed in enumerate(eval_dataset):
      config = rm.get_eval_config(eval_dataset = ed,
               eval_ses = eval_ses, 
               parcellation = parcellation,
               crossed = "half", # or None
               type = eval_type[i],
               splitby = None)

      dirname=[]
      mname=[]

      for a in logalpha_list:
         dirname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}")
         mname.append(f"{train_dataset}_{train_ses}_{parcellation}_{method}_A{a}")
      df, df_voxels = rm.eval_model(dirname,mname,config)
      save_path = gl.conn_dir+ f"/eval"

      if not os.path.isdir(save_path):
         os.mkdir(save_path)
      else:
         pass

      file_name = save_path + f"/{config['eval_dataset']}_eval_{eval_id}.tsv"
      df.to_csv(file_name, index = False, sep='\t')
   return df,df_voxels

if __name__ == "__main__":
   # train_models(train_ses = 'all',dataset = "HCP",type='Tseries',crossed=None)
   # avrg_model(train_data = "HCP",train_ses= "all")
   ED=["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC"]
   ET=["CondHalf","CondHalf", "CondHalf", "CondHalf", "CondHalf", "CondHalf"]
   # eval_models(eval_dataset = ED, train_dataset="MDTB", train_ses="all",eval_id = 'Md')
   # eval_models(eval_dataset = ED, train_dataset="MDTB", train_ses="ses-s1",eval_id = 'Mds1')
   # for ed in ED[1:]:
   #    train_models(train_ses = 'all',dataset = ed)
   #    avrg_model(train_data = ed,train_ses= "all")
   eval_models(eval_dataset = ED, eval_type = ET,train_dataset="HCP", train_ses="all",eval_id = 'Hc')
   # eval_models(eval_dataset = ED, train_dataset="Demand", train_ses="all",eval_id = 'De')
   # eval_models(eval_dataset = ED, train_dataset="Nishimoto", train_ses="all",eval_id = 'Ni')
   # eval_models(eval_dataset = ED, train_dataset="WMFS", train_ses="all",eval_id = 'Wm')
   
   # train_models(train_ses = 'all',dataset = "MDTB")
   # avrg_model(train_data = "MDTB",train_ses= "all")
   # train_models(train_ses = 'all',dataset = "MDTB")
   # avrg_model(train_data = "MDTB",train_ses= "all")
   