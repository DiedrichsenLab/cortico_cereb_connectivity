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
   config, conn_list, df_tmp =rm.train_model(config)
   return df_tmp

def avrg_model(logalpha_list = [0, 2, 4, 6, 8, 10, 12],
               train_data = "MDTB",
               train_ses= "ses-s1",
               parcellation = 'Icosahedron1002',
               method='L2Regression',
               cerebellum='SUIT3',
               parameters=['scale_','coef_'],
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
      fname = model_path + f"/{mname_base}{mname_ext}_{avg_id}"
      cio.save_model(avrg_model,info,fname)
      
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
                add_rest= False,
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
       eval_dataset (list): List of evaluation datasets. Defaults to ["Demand"].
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
                                 crossed = crossed, # "half", # or None
                                 type = eval_type[i],
                                 cerebellum=cerebellum,
                                 splitby = None,
                                 add_rest = add_rest,
                                 model=model)

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
         df = df.append(dd,ignore_index=True)
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

def train_all_nnls(dataset = "MDTB",
                 logalpha_list = [-2],
                 subj_list = "all",
                 parcellation="Icosahedron162"):

   config = rm.get_train_config(train_dataset=dataset,
                                train_ses='ses-s1',
                                subj_list=subj_list,
                                log_alpha = logalpha_list,
                                crossed = 'half',
                                type = 'CondHalf',
                                cerebellum='SUIT3',
                                parcellation=parcellation,
                                method = 'NNLS',
                                add_rest=False,
                                std_cortex='parcel',
                                std_cerebellum='global',
                                validate_model=False)
   config, conn_list, df_tmp =rm.train_model(config)
   return df_tmp

def train_all_l2(dataset = "MDTB",
                 logalpha_list = [0,2,4,6,8,10,12],
                 subj_list = "all",
                 parcellation="Icosahedron162"):

   config = rm.get_train_config(train_dataset=dataset,
                                train_ses='ses-s1',
                                subj_list=subj_list,
                                log_alpha = logalpha_list,
                                crossed = 'half',
                                type = 'CondHalf',
                                cerebellum='SUIT3',
                                parcellation=parcellation,
                                method = 'L2regression',
                                add_rest=False,
                                std_cortex='parcel',
                                std_cerebellum='global',
                                validate_model=False)
   config, conn_list, df_tmp =rm.train_model(config)
   return df_tmp

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
              parcellation = 'Icosahedron1002',
              method='NNLS',
              parameters=['coef_'],
              cerebellum='SUIT3',
              logalpha_list = [4])
   # avrg_all_wta()