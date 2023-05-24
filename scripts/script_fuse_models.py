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
from cortico_cereb_connectivity.scripts.script_train_eval_models import eval_models
import json 
from copy import deepcopy, copy

def fuse_models(logalpha = [6, -2, 6, 8, 6, 6, 10],
               train_data = ['Demand','HCP','IBC','MDTB','Somatotopic','WMFS','Nishimoto'],
               weight = [1,1,1,1,1,1,1], 
               train_ses= "all",
               parcellation = 'Icosahedron1002',
               method='L2Regression',
               cerebellum='SUIT3',
               fuse_id = '01',
               save = True):

    coef_list = []
    scale_list = []
    models = []
    train_info = [] 
    for i,(la,tdata) in enumerate(zip(logalpha,train_data)):
        mname = f"{tdata}_{train_ses}_{parcellation}_{method}"
        model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
        m = mname + f"_A{la}_avg"
        fname = model_path + f"/{m}.h5"
        json_name = model_path + f"/{m}.json"
        m = dd.io.load(fname)
        models.append(m)
        coef_list.append(m.coef_/m.scale_) # Adjust for scaling
        scale_list.append(m.scale_)
         # Load json file
        with open(json_name) as json_file:
            train_info.append(json.load(json_file))

    Coef = np.stack(coef_list,axis=0)
    Scale = np.stack(scale_list,axis=0)
    weight_norm = np.sqrt(np.nansum((Coef**2),axis=(1,2),keepdims=True))
    wCoef = Coef/weight_norm
    wCoef = wCoef * np.array(weight).reshape(-1,1,1)
    fused_model = deepcopy(models[0])
    setattr(fused_model,'coef_',np.nansum(wCoef,axis=0))
    setattr(fused_model,'scale_',np.ones((Scale.shape[1],)))
    fused_info = train_info[0].copy()
    fused_info['train_dataset'] = 'Fusion' 
    fused_info['extension'] = fuse_id
    fused_info['logalpha'] = logalpha
    fused_info['weight'] = weight
    if save:
        mname = f"Fusion_{train_ses}_{parcellation}_{method}"
        model_path = os.path.join(gl.conn_dir,cerebellum,'train',mname)
        if os.path.exists(model_path) == False:
            os.makedirs(model_path)

        dd.io.save(model_path + f"/{mname}_{fuse_id}_avg.h5",
                   fused_model, compression=None)
        with open(model_path + f"/{mname}_{fuse_id}_avg.json", 'w') as fp:
            json.dump(fused_info, fp, indent=4)

def eval_fusion(): 
    ED=["MDTB","WMFS", "Nishimoto", "IBC","Demand","Somatotopic"]
    ET=["CondHalf","CondHalf", "CondHalf", "CondHalf",'CondHalf','CondHalf']
    eval_models(eval_dataset = ED, eval_type = ET,
                  crossed='half',
                  train_dataset='Fusion', 
                  train_ses="all",
                  ext_list = ['01','02','03','04','05','06','07'],
                  eval_id = 'Fu',
                  model='avg')



if __name__ == "__main__":
    # fuse_models(weight=[1,0,0,0,0,0,0],fuse_id='01')
    # fuse_models(weight=[0,1,0,0,0,0,0],fuse_id='02')
    # fuse_models(weight=[0,0,0,1,0,0,0],fuse_id='03')
    # fuse_models(weight=[1,1,0,1,0,0,0],fuse_id='04')
    # fuse_models(weight=[1,1,1,1,1,1,1],fuse_id='05')
    # fuse_models(weight=[1,0,1,1,1,1,1],fuse_id='06')
    # fuse_models(weight=[1,0,1,1,0,1,1],fuse_id='07')
    eval_fusion()
    dff=rm.comb_eval(models=['Fu'])
    pass
