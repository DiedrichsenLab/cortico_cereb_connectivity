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
import numpy as np

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
      fname = model_path + f"/{mname_base}{mname_ext}_{avg_id}"
      cio.save_model(avrg_model,info,fname)

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
                                train_ses='all',
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

def train_group_model(dataset = "MDTB",
                 logalpha_list = [0,1,2,3,4,6,8,10],
                 subj_list = None,
                 method = 'NNLS',
                 parcellation="Icosahedron162"):

   config = rm.get_train_config(train_dataset=dataset,
                                train_ses='all',
                                subj_list=subj_list,
                                log_alpha = logalpha_list,
                                crossed = 'half',
                                type = 'CondHalf',
                                cerebellum='MNISymC3',
                                parcellation=parcellation,
                                method = method,
                                add_rest=False,
                                std_cortex='parcel',
                                std_cerebellum='global',
                                validate_model=False,
                                cortical_cerebellar_act='avg')
   config, conn_list, df_tmp =rm.train_model(config)
   return df_tmp

def train_global_model(dataset = 'MdSoScLa',
                 logalpha_list = [0,1,2,3,4,6,8,10],
                 method = 'NNLS',
                 parcellation="Icosahedron162"):

   config = rm.get_train_config(train_dataset=dataset,
                                subj_list=None,
                                log_alpha = logalpha_list,
                                crossed = 'half',
                                type = 'CondHalf',
                                cerebellum='MNISymC3',
                                parcellation=parcellation,
                                method = method,
                                std_cerebellum='global',
                                validate_model=False)
   config, conn_list, df_tmp =rm.train_global_model(config)
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



def train_all_global_ldo(): 
      names = gl.get_ldo_names()
      for i,dsstr in enumerate(names):
         print(f"Training global model for {gl.dscode[i]}: {dsstr}")
         train_global_model(dataset=dsstr,method='L2reg',logalpha_list = [0,1,2,3,4,6,8,10])
         train_global_model(dataset=dsstr,method='NNLS',logalpha_list = [0,1,2,3,4,6,8,10])

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
   train_all_global_ldo() 
   pass
   # train_global_model(dataset='MdWfIbDeNiSoScLa',method='L2reg',logalpha_list = [0,1,2,3,4,6,8,10])
   # train_global_model(dataset='MdWfIbDeNiSoScLa',method='NNLS',logalpha_list = [0,1,2,3,4,6,8,10])
   # train_group_model(dataset='WMFS',method='L2reg',logalpha_list = [4,6,8,10])
   # train_group_model(dataset='WMFS',method='NNLS',logalpha_list = [4,6,8,10])
   # avrg_all()
   # eval_mdtb(method='NNLS',ext_list=[-4,-2,0,2,4,6,8,10])
   # eval_mdtb(method='L2regression',ext_list=[0,2,4,6,8,10,12])
   # train_all_nnls(logalpha_list=[6],subj_list=np.arange(5,24),parcellation='Icosahedron1002')

   # train_all_l2(logalpha_list=[6],parcellation='Icosahedron1002')
   # train_all_nnls(logalpha_list=[-2,0,2],parcellation='Icosahedron1002')
   # avrg_model(train_data = 'MDTB',
   #            train_ses= "ses-s1",
   #            parcellation = 'Icosahedron162',
   #            method='NNLS',
   #            parameters=['coef_'],
   #            cerebellum='SUIT3',
   #            logalpha_list = [6])
   # avrg_all_wta()