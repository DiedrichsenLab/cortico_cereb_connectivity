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
import json
from pathlib import Path
import warnings

def sort_roi_rows(cifti_img):
    row_axis = cifti_img.header.get_axis(0)
    p_axis = cifti_img.header.get_axis(1)
    indx = row_axis.name.argsort()
    data = cifti_img.get_fdata()[indx,:]
    row_axis = nb.cifti2.ScalarAxis(row_axis.name[indx])
    header = nb.Cifti2Header.from_axes((row_axis, p_axis)) 
    cifti_img_new = nb.Cifti2Image(data, header=header)
    return cifti_img_new

def get_weight_map(method = "L2Regression", 
                    cortex_roi = "Icosahedron1002", 
                    cerebellum_roi = "NettekovenSym32",
                    cerebellum_atlas = "SUIT3", 
                    extension = 'A8', 
                    dataset_name = "MDTB", 
                    ses_id = "ses-s1",
                    train_t = 'train'
                    ):

    """ make cifti image for the connectivity weights
    Uses the avg model to get the weights, average the weights for voxels within a cerebellar parcel
    creates cortical maps of average connectivity weights.
    Args: 
        method (str) - connectivity method used to estimate weights
        cortex_roi (str) - cortical tessellation/roi used when training connectivity weights
        cerebellum_roi (str) - name of the cerebellar roi file you want to get the connectivity weights for
        cerebellum_atlas (str) - cerebellar atlas used in training connectivity model. "SUIT3" or "MNISym2"
        extension (str) - String indicating regularization parameter ('A0')
        dataset_name (str) - name of the dataset as in functional_fusion framework
        ses_id (str) - session id used when training the model. "all" for aggregated model over sessions
    Returns:
        cifti_img (nibabel.Cifti2Image) pscalar cifti image for the cortical maps. ready to be saved!
    """
    # make model name
    m_basename = f"{dataset_name}_{ses_id}_{cortex_roi}_{method}"
    # load in the connectivity average connectivity model
    fpath = gl.conn_dir + f"/{cerebellum_atlas}/train/{m_basename}"

    # load the avg model
    model = dd.io.load(fpath + f"/{m_basename}_{extension}_avg.h5")

    # get the weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        weights = model.coef_/model.scale_

    # prepping the parcel axis file
    ## make atlas object first
    atlas_fs, _ = am.get_atlas("fs32k", gl.atlas_dir)

    # load the label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # get parcels for the neocortex
    _, label_fs = atlas_fs.get_parcel(label_fs, unite_struct = False)

    # getting parcel info for the cerebellum 
    atlas_suit, _ = am.get_atlas(cerebellum_atlas, gl.atlas_dir)

    # load the label file for the cerebellum
    label_suit = gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}_space-SUIT_dseg.nii"

    # getting parcels for the cerebellum
    atlas_suit.get_parcel(label_suit)

    # get the average cortical weights for each cerebellar parcel
    weights_parcel, labels = fdata.agg_parcels(weights.T, atlas_suit.label_vector, fcn=np.nanmean)

    # preping the parcel axis
    ## load the lookup table for the cerebellar parcellation to get the names of the parcels
    index,colors,labels = nt.read_lut(gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}.lut")

    # create parcel axis for the cortex (will be used as column axis in pscalar file)
    p_axis = atlas_fs.get_parcel_axis()

    # generate row axis with the last rowi being the scale
    row_axis = nb.cifti2.ScalarAxis(labels[1:]) 
    data = weights_parcel.T
    
    # make header
    ## rows are maps corresponding to cerebellar parcels
    ## columns are cortical tessels
    header = nb.Cifti2Header.from_axes((row_axis, p_axis)) 
    cifti_img = nb.Cifti2Image(data, header=header)
    
    return cifti_img

def get_scale_map(method = "L2Regression", 
                    cortex_roi = "Icosahedron1002", 
                    cerebellum_roi = "NettekovenSym68c32",
                    cerebellum_atlas = "SUIT3", 
                    extension = 'A4', 
                    dataset_names = ["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC","HCP"], 
                    ses_id = "all",
                    type = "pscalar"
                    ):

    """ make cifti image for the scale factor across datasets
    Args: 
        method (str) - connectivity method used to estimate weights
        cortex_roi (str) - cortical tessellation/roi used when training connectivity weights
        cerebellum_atlas (str) - cerebellar atlas used in training connectivity model. "SUIT3" or "MNISym2"
        log_alpha (float) - log of the regularization parameter used in estimating weights (doesn't matter)
        dataset_name (str) - name of the dataset as in functional_fusion framework
        ses_id (str) - session id used when training the model. "all" for aggregated model over sessions
        type(str) - type of the cifti you want to create ("pscalar" or "dscalar")
    Returns:
        cifti_img (nibabel.Cifti2Image) pscalar cifti image for the cortical maps. ready to be saved!
    """
    scale_maps = []
    for dn in dataset_names:
        # make model name
        m_basename = f"{dn}_{ses_id}_{cortex_roi}_{method}"
        # load in the connectivity average connectivity model
        fpath = gl.conn_dir + f"/{cerebellum_atlas}/train/{m_basename}"

        # load the avg model
        model = dd.io.load(fpath + f"/{m_basename}_{extension}_avg.h5")

        # get the weights
        scale_maps.append(model.scale_/model.scale_.max())

    # prepping the parcel axis file
    atlas_fs, _ = am.get_atlas("fs32k", gl.atlas_dir)

    # load the label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # get parcels for the neocortex
    _, label_fs = atlas_fs.get_parcel(label_fs, unite_struct = False)

    # create parcel axis for the cortex (will be used as column axis in pscalar file)
    p_axis = atlas_fs.get_parcel_axis()

    # generate row axis with the last rowi being the scale
    row_axis = nb.cifti2.ScalarAxis(dataset_names) 
    data = np.r_[scale_maps]
    
    # make header
    ## rows are maps corresponding to cerebellar parcels
    ## columns are cortical tessels
    header = nb.Cifti2Header.from_axes((row_axis, p_axis)) 
    cifti_img = nb.Cifti2Image(data, header=header)
    return cifti_img

def make_weight_map(dataset= "HCP",extension = 'A0',ext=""):
    cifti_img = get_weight_map(method = "L2Regression", 
                                cortex_roi = "Icosahedron1002", 
                                cerebellum_roi = "NettekovenSym32",
                                cerebellum_atlas = "SUIT3", 
                                extension = extension, 
                                dataset_name = dataset, 
                                ses_id = "all",
                                train_t = "train"+ext
                                )
    fname = gl.conn_dir + f'/{"maps"+ext}/{dataset}_L2_{extension}.pscalar.nii'
    # cifti_img = sort_roi_rows(cifti_img)
    nb.save(cifti_img,fname)


if __name__ == "__main__":
    # make_weight_map('Demand','A8')
    # make_weight_map('HCP','A-2')
    # make_weight_map('IBC','A6')
    # make_weight_map('MDTB','A8')
    # make_weight_map('Somatotopic','A8')
    # make_weight_map('WMFS','A8')
    # make_weight_map('Nishimoto','A10')
    # make_weight_map('Fusion','05')
    # make_weight_map('Fusion','06')

    make_weight_map('Fusion','06')
    pass
