"""
Script to summarize the group average weights based on
ROIs
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
import cortico_cereb_connectivity.cio as cio
import Functional_Fusion.atlas_map as am
import nitools as nt
import json
from pathlib import Path
import warnings

def sort_roi_rows(cifti_img):
    """ sort the rows of a cifti image alphabetically by the name of the cerebellar parcel"""
    row_axis = cifti_img.header.get_axis(0)
    p_axis = cifti_img.header.get_axis(1)
    indx = row_axis.name.argsort()
    data = cifti_img.get_fdata()[indx,:]
    row_axis = nb.cifti2.ScalarAxis(row_axis.name[indx])
    header = nb.Cifti2Header.from_axes((row_axis, p_axis))
    cifti_img_new = nb.Cifti2Image(data, header=header)
    return cifti_img_new

def avrg_weight_map(method = "L2Regression",
                    cortex_roi = "Icosahedron1002",
                    cerebellum_roi = "NettekovenSym32",
                    cerebellum_atlas = "SUIT3",
                    extension = 'A8',
                    dataset_name = "MDTB",
                    ses_id = "all",
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
    if (len(extension) > 0) and extension[0] != "_":
        extension = "_" + extension
    model = dd.io.load(fpath + f"/{m_basename}{extension}_avg.h5")

    # get the weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        weights = model.coef_/model.scale_

    # label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # label file for the cerebellum
    label_suit = gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}_space-SUIT_dseg.nii"

    # get the average cortical weights for each cerebellar parcel
    atlas_suit,ainf = am.get_atlas(cerebellum_atlas)
    atlas_suit.get_parcel(label_suit)
    weights_parcel, labels = fdata.agg_parcels(weights.T, atlas_suit.label_vector, fcn=np.nanmean)

    # load the lookup table for the cerebellar parcellation to get the names of the parcels
    index,colors,labels = nt.read_lut(gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}.lut")

    cifti_img = cio.model_to_cifti(weights_parcel.T,
                                   src_atlas = "fs32k",
                                   trg_atlas = cerebellum_atlas,
                                   src_roi = label_fs,
                                   trg_roi = labels[1:],
                                   type = 'scalar')
    return cifti_img

def avrg_scale_map(method = "L2Regression",
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

def make_avrg_weight_map(dataset= "HCP",extension = 'A0',ext="",method="L2Regression"):
    """Convenience functions to generate the weight maps for the Nettekoven dataset"""
    cifti_img = avrg_weight_map(method = method,
                                cortex_roi = "Icosahedron1002",
                                cerebellum_roi = "NettekovenSym32",
                                cerebellum_atlas = "SUIT3",
                                extension = extension,
                                dataset_name = dataset,
                                ses_id = "all",
                                train_t = "train"+ext
                                )
    if (len(extension) > 0) and extension[0] != "_":
        extension = "_" + extension
    fname = gl.conn_dir + f'/{"maps"+ext}/{dataset}_{method[:2]}{extension}.pscalar.nii'
    # cifti_img = sort_roi_rows(cifti_img)
    nb.save(cifti_img,fname)

def make_weight_table(dataset="HCP",extension="A0",cortical_roi="yeo17"):
    """ Generate a from the cifti-files summarizing the input based on the Yeo parcellation"""
    fname = gl.conn_dir + f'/{"maps"}/{dataset}_L2_{extension}.pscalar.nii'
    # cifti_img = sort_roi_rows(cifti_img)
    data = nb.load(fname)
    surf_data = nt.surf_from_cifti(data)

    label = []
    for i,h in enumerate(["L","R"]):
        lname = gl.atlas_dir + f"/tpl-fs32k/{cortical_roi}.{h}.label.gii"
        gii = nb.load(lname)
        label.append(gii.agg_data())
    label_names = nt.get_gifti_labels(gii)
    clabel_names = data.header.get_axis(0).name
    T = []
    for i,h in enumerate(["L","R"]):
        A=surf_data[i].copy()
        A[A<0]=0
        for k in range(max(label[i])):
            m=np.nanmean(surf_data[i][:,label[i]==k+1],axis=1)
            t={'cereb_region':clabel_names,
               'fs_region':label_names[k+1],
               'hemisphere':h,
               'weight':m,
               'sizeR':np.sum(label[i]==k+1),
               'totalW':np.nansum(A[:,label[i]==k+1])}
            T.append(pd.DataFrame(t))
    T = pd.concat(T,ignore_index=True)
    return T

def get_weight_by_cortex(method = "L2Regression",
                    cortex_roi = "Icosahedron1002",
                    cerebellum_atlas = "SUIT3",
                    extension = 'A8',
                    dataset_name = "MDTB",
                    ses_id = "all",
                    train_t = 'train',
                    sum_cortex = 'yeo17',
                    sum_method = 'positive'
                    ):
    """ Make table of the connectivity weights for each cortical parcel,
    averaged across the entire cerebellum.
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
    label_conn_fname = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # get parcels for the neocortex
    label_conn, l_conn = atlas_fs.get_parcel(label_conn_fname, unite_struct = False)

    # load the label file for summarizing the cortex
    label_sum_fname = [gl.atlas_dir + f"/tpl-fs32k/{sum_cortex}.{hemi}.label.gii" for hemi in ["L", "R"]]
    label_sum, l_sum = atlas_fs.get_parcel(label_sum_fname, unite_struct = True)

    # Expand data, threshold, and then summarize
    ex_weights = weights[:,label_conn-1]
    if sum_method == 'positive':
        ex_weights[ex_weights<0]=0

    # Get region names and colors
    gii = nb.load(label_sum_fname[0])
    label_names = nt.get_gifti_labels(gii)
    colors,_ = nt.get_gifti_colortable(gii)

    # Summarize the data
    N = l_sum.shape[0]
    cort_size = np.zeros(N,)
    weight_sum = np.zeros((N,))
    for l in l_sum:
        cort_size[l-1] = np.sum(label_sum==l)
        weight_sum[l-1] = np.nansum(ex_weights[:,label_sum==l])
    cort_size = cort_size/cort_size.sum()*100
    weight_sum = weight_sum/weight_sum.sum()*100

    T = pd.DataFrame({'cort_size':cort_size,
                        'cereb_size':weight_sum,
                        'name':label_names[1:]})
    return T,colors[1:,:]

def export_model_as_cifti(method = "L2Regression",
                    cortex_roi = "Icosahedron1002",
                    cerebellum_atlas = "SUIT3",
                    extension = 'A8',
                    dataset_name = "MDTB",
                    ses_id = "all",
                    ):
    # make model name
    m_basename = f"{dataset_name}_{ses_id}_{cortex_roi}_{method}"
    # load in the connectivity average connectivity model
    fpath = gl.conn_dir + f"/{cerebellum_atlas}/train/{m_basename}"

    # load the avg model
    if (len(extension) > 0) and extension[0] != "_":
        extension = "_" + extension
    model = dd.io.load(fpath + f"/{m_basename}{extension}_avg.h5")

    # get the weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        weights = model.coef_/model.scale_

    # label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    cifti_img = cio.model_to_cifti(weights,
                                   src_atlas = "fs32k",
                                   trg_atlas = cerebellum_atlas,
                                   src_roi = label_fs,
                                   trg_roi = None,
                                   type = 'conn')

    fname = gl.conn_dir + f'/{"maps"}/{dataset_name}_{method[:2]}_{cerebellum_atlas}_{cortex_roi}.pdconn.nii'
    nb.save(cifti_img,fname)


    cifti_img = cio.model_to_cifti(weights.T,
                                   src_atlas = cerebellum_atlas,
                                   trg_atlas = "fs32k",
                                   src_roi = None,
                                   trg_roi = label_fs,
                                   type = 'conn')

    fname = gl.conn_dir + f'/{"maps"}/{dataset_name}_{method[:2]}_{cerebellum_atlas}_{cortex_roi}.dpconn.nii'
    nb.save(cifti_img,fname)

    return






if __name__ == "__main__":
    export_model_as_cifti(dataset_name= "Fusion",extension = '06',method="L2Regression")
    # make_avrg_weight_map(dataset= "Fusion",extension = '06',method="L2Regression")
    # make_avrg_weight_map(dataset= "MDTB",extension = '',method="WTA")
    # make_avrg_weight_map(dataset= "Demand",extension = '',method="WTA")
    # make_avrg_weight_map(dataset= "WMFS",extension = '',method="WTA")
    # make_avrg_weight_map(dataset= "Nishimoto",extension = '',method="WTA")
    # make_avrg_weight_map(dataset= "Somatotopic",extension = '',method="WTA")
    # make_avrg_weight_map(dataset= "IBC",extension = '',method="WTA")
    # make_avrg_weight_map(dataset= "HCP",extension = '',method="WTA")
    # T,colors= get_weight_by_cortex(dataset_name='Fusion',extension='06')
    pass
    # ["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC","HCP"],