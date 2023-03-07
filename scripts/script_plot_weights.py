"""
script for getting group average weights
@ Ladan Shahshahani Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pandas as pd
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.data as cdata
import Functional_Fusion.atlas_map as am
import nitools as nt
from pathlib import Path

def plot_weights(method = "L2Regression", 
                 cortex = "Icosahedron-1002_Sym.32k", 
                 cerebellum = "NettekovenSym34", 
                 log_alpha = 8, 
                 dataset_name = "MDTB", 
                 ses_id = "ses-s1", 
                 ):
    
    # get the config 
    config = rm.get_train_config()
    # get the group weights
    weights_group = rm.get_group_weights(config, fcn = np.nanmean, fold = "train")

    # get atlases and create parcels/parcel labels
    atlas_cereb, _ = am.get_atlas('SUIT3',gl.atlas_dir)
    atlas_cortex, _ = am.get_atlas('fs32k', gl.atlas_dir)

    # get label files for cerebellum and cortex
    label_cereb = gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(gl.atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')

    # get lut file for cerebellar parcel names
    index,_,labels_names = nt.read_lut(gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}.lut')

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct = False)

    # get the average weights for each parcel
    weights_parcel, labels = fdata.agg_parcels(weights_group.T, atlas_cereb.label_vector, fcn=np.nanmean)
    
    # create cifti columns for each average cortical weight
    ## get the labels that are not absent
    labels_names_arr = np.array(labels_names)
    col_names = labels_names_arr[np.isin(index, labels)]

    # create cifti file 
    cifti_img = cdata.cortex_parcel_to_cifti(weights_parcel.T, atlas_cortex, parcel_axis_names= col_names)
    # save weight map
    nb.save(cifti_img,os.path.join(gl.conn_dir, dataset_name, 'train', f'{cortex}_{cerebellum}_{method}_{log_alpha}.dscalar.nii'))
    return


if __name__ == "__main__":
    # plot_weights(cerebellum='MDTB10')
    plot_weights(cerebellum="NettekovenSym68c32")
    # plot_weights(cerebellum="Verbal2Back")
    # plot_weights(cerebellum="Buckner7")