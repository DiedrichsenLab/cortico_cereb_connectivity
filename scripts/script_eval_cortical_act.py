"""
Script for evaluating group models using either the average cortical activity or individual cortical activity 
Also calcualtes the leave-one-out (loo) reliability of cerebellar activity for each subject
"""
import os
import pandas as pd
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.reliability as frel # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import numpy as np 



def eval_models_script(ext_list = [0,1,2,3,4,6,8,10],
                train_dataset = "MDTB",
                train_ses = "all",
                method = "L2reg",
                parcellation = "Icosahedron162",
                cerebellum='MNISymC3',
                model = 'group',
                eval_dataset = ["MDTB"],
                eval_type = ["CondHalf"],
                eval_ses  = "all",
                eval_id = 'MDTBgrp',
                subj_list = "all",
                crossed = 'half',
                add_rest = False,
                cortical_act = 'avg',  # 'ind','avg','loo'                                
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
       eval_dataset (list): List of evaluation datasets. Defaults to ["Demand"].
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
      econfig = rm.get_eval_config(eval_dataset = ed,
                                 eval_ses = eval_ses,
                                 parcellation = parcellation,
                                 crossed = crossed, # "half", # or None
                                 type = eval_type[i],
                                 cerebellum=cerebellum,
                                 splitby = None,
                                 add_rest = add_rest,
                                 subj_list = subj_list,
                                 cortical_act = cortical_act)

      # Get the config for the evaluation
      mconfig = rm.get_model_config(model=model)
      dirname,mname = rm.get_model_names(train_dataset,train_ses,parcellation,method,ext_list)

      # Evaluate them
      df, df_voxels = rm.eval_model(dirname,mname,econfig,mconfig)
      save_path = gl.conn_dir+ f"/{cerebellum}/eval"

      if not os.path.isdir(save_path):
         os.mkdir(save_path)
      else:
         pass
      ename = econfig['eval_dataset']
      if econfig['eval_ses'] != 'all':
         ses_code = econfig['eval_ses'][i].split('-')[1]
         ename = econfig['eval_dataset'] + ses_code
      file_name = save_path + f"/{ename}_{method}_{eval_id}.tsv"
      if os.path.isfile(file_name) & append:
         dd = pd.read_csv(file_name, sep='\t')
         df = pd.concat([dd, df], ignore_index=True)
         # df = df.append(dd,ignore_index=True) # pd.append is deprecated
      df.to_csv(file_name, index = False, sep='\t')
   return df,df_voxels


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
   # eval_models_script(eval_id = 'MDTB_Cavg',cortical_act = 'avg')
   # eval_models_script(eval_id = 'MDTB_Cind',cortical_act = 'ind')
   for m in ['L2reg','NNLS']:
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['MDTB'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['WMFS'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['IBC'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['Demand'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['HCPur100'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['Nishimoto'],add_rest=False)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['Somatotopic'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['Social'],eval_ses=['ses-social'],add_rest=True)
      eval_models_script(eval_id = 'MDTBgrp',cortical_act = 'avg',method=m,eval_dataset=['Language'],eval_ses=['ses-localizer'],add_rest=True)
   # train_all()
   # avrg_all()
   # eval_mdtb(method='NNLS',ext_list=[-4,-2,0,2,4,6,8,10])
   # eval_mdtb(method='L2regression',ext_list=[0,2,4,6,8,10,12])
   # train_all_nnls(logalpha_list=[6],subj_list=np.arange(5,24),parcellation='Icosahedron1002')

   # train_all_l2(logalpha_list=[6],parcellation='Icosahedron1002')
   # train_all_nnls(logalpha_list=[-2,0,2],parcellation='Icosahedron1002')

   # avrg_all_wta()