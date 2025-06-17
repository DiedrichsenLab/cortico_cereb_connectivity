"""
script for training models (Better to use run_model.py directly - 
these are just examples of how to use the functions in run_model.py)

@ Ladan Shahshahani, Joern Diedrichsen Jan 30 2023 12:57
"""
import os
import pandas as pd
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
      
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

      dirname,mname = rm.get_model_name(train_dataset,train_ses,parcellation,method)
      # Evaluate them
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

def eval_mdtb(method = 'NNLS',ext_list=[-4,-2,0,2]):
   # Get the names of models to evaluate
   dirname,mname = rm.get_model_names(train_dataset = "MDTB",
               train_ses = "ses-s1",
                method = method,
                parcellation = "Icosahedron162",
                ext_list=ext_list)
   # Evaluation config
   config = rm.get_eval_config(eval_dataset = "MDTB",  
                eval_ses  = "ses-s2",
                type = "CondHalf",
                parcellation = "Icosahedron162",
                crossed = 'half',
                add_rest= False,
                std_cerebellum='global',
                std_cortex='parcel',
                model = 'ind')
   df,vox=rm.eval_model(dirname,mname,config)
   df.to_csv(gl.conn_dir + f"/SUIT3/eval/MDTBs2_{method}_MDs1-ind.tsv", index = False, sep='\t')
   pass


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
   # eval_mdtb(method='NNLS',ext_list=[-4,-2,0,2,4,6,8,10])
   # eval_mdtb(method='L2regression',ext_list=[0,2,4,6,8,10,12])
   # train_all_nnls(logalpha_list=[6],subj_list=np.arange(5,24),parcellation='Icosahedron1002')

   # train_all_l2(logalpha_list=[6],parcellation='Icosahedron1002')
   # train_all_nnls(logalpha_list=[-2,0,2],parcellation='Icosahedron1002')
   avrg_model(train_data = 'MDTB',
              train_ses= "ses-s1",
              parcellation = 'Icosahedron162',
              method='NNLS',
              parameters=['coef_'],
              cerebellum='SUIT3',
              logalpha_list = [6])
   # avrg_all_wta()