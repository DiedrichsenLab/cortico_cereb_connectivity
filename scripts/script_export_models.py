"""
script for exporting models as pdconn.nii files
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import nibabel as nb
import Functional_Fusion.atlas_map as am # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.model as cm
import json

def export_model(model_name,model_ext,file_name,type='pdconn'):
    model_path = os.path.join(gl.conn_dir,'SUIT3','train',model_name)
    fname = model_path + f"/{model_name}_{model_ext}"
    M,info = cm.load_model(fname)
    adir = am.default_atlas_dir
    src_roi = [f"{adir}/tpl-{info['cortex']}/Icosahedron1002.L.label.gii",
               f"{adir}/tpl-{info['cortex']}/Icosahedron1002.R.label.gii"]
    C = M.to_cifti(src_atlas=info['cortex'],
                    trg_atlas=info['cerebellum'],
                    src_roi=src_roi,
                    fname=f'data/{file_name}',
                    dtype = 'float32')


if __name__ == "__main__":
    export_model('MDTB_all_Icosahedron1002_L2regression','A8_avg','Nettekoven_2024_MDTB_L2.pdconn.nii')
    export_model('Demand_all_Icosahedron1002_L2regression','A8_avg','Nettekoven_2024_Demand_L2.pdconn.nii')
    export_model('HCP_all_Icosahedron1002_L2regression','A-2_avg','Nettekoven_2024_HCP_L2.pdconn.nii')
    export_model('IBC_all_Icosahedron1002_L2regression','A6_avg','Nettekoven_2024_IBC_L2.pdconn.nii')
    export_model('Nishimoto_all_Icosahedron1002_L2regression','A10_avg','Nettekoven_2024_Nishimoto_L2.pdconn.nii')
    export_model('Somatotopic_all_Icosahedron1002_L2regression','A8_avg','Nettekoven_2024_Somatotopic_L2.pdconn.nii')
    export_model('WMFS_all_Icosahedron1002_L2regression','A8_avg','Nettekoven_2024_WMFS_L2.pdconn.nii')
    export_model('Fusion_all_Icosahedron1002_L2regression','06_avg','Nettekoven_2024_Fusion_L2.pdconn.nii')
    export_model('MDTB_ses-s1_Icosahedron1002_L2Regression','A8_avg','Shahshahani_2024_MDTB_L2.pdconn.nii')
    export_model('MDTB_ses-s1_Icosahedron1002_L1Regression','A-5_avg','Shahshahani_2024_MDTB_L1.pdconn.nii')
    export_model('Fusion_all_Icosahedron1002_L2Regression','09_avg', 'Shahshahani_2024_Fusion_L2.pdconn.nii')
