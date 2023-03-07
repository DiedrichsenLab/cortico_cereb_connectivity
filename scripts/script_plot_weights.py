"""
script for getting group average weights
@ Ladan Shahshahani Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pandas as pd
import re
import sys
from collections import defaultdict
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import Functional_Fusion.atlas_map as am
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
    W = rm.get_group_weights(config, fcn = np.nanmean, fold = "train")
    # get the file containing best weights
    filename = os.path.join(gl.conn_dir, dataset_name, 'train', f'{cortex}_{ses_id}_{method}_logalpha_{log_alpha}_best_weights.npy')
    weights = np.load(filename)

    # get atlases and create parcels/parcel labels
    atlas_cereb, _ = am.get_atlas('SUIT3',gl.atlas_dir)
    atlas_cortex, _ = am.get_atlas('fs32k', gl.atlas_dir)

    # get label files for cerebellum and cortex
    label_cereb = gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(gl.atlas_dir + '/tpl-fs32k' + f'/{cortex}.{hemi}.label.gii')

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct = False)
    
    # get the maps
    cifti_img = cdata.convert_cortex_to_cifti(weights, atlas_cereb, atlas_cortex)
    # save weight map
    nb.save(cifti_img,os.path.join(cdata.conn_dir, dataset_name, 'train', f'{cortex}_{cerebellum}_{method}_{log_alpha}.dscalar.nii'))
    return


if __name__ == "__main__":
    plot_weights(cerebellum='MDTB10')
    # plot_weights(cerebellum="NettekovenSym34")
    # plot_weights(cerebellum="Verbal2Back")
    # plot_weights(cerebellum="Buckner7")