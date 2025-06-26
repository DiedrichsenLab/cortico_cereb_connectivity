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
import cortico_cereb_connectivity.summarize as cs
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import nitools as nt
from pathlib import Path
import warnings


def make_avrg_weight_map(dataset= "HCP",extension = 'A0',ext="",method="L2Regression"):
    """Convenience functions to generate the weight maps for the Nettekoven dataset"""
    cifti_img = cs.avrg_weight_map_roi(method = method,
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
    # cifti_img = cs.sort_roi_rows(cifti_img)
    nb.save(cifti_img,fname)

def make_weight_table(dataset="HCP",extension="A0",cortical_roi="yeo17"):
    """ Generate a from the cifti-files summarizing the input based on the Yeo parcellation"""
    # Expand the data from the parcellation of the cortex to full surface
    fname = gl.conn_dir + f'/{"maps"}/{dataset}_L2_{extension}.pscalar.nii'
    # cifti_img = cs.sort_roi_rows(cifti_img)
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
            t={'cereb_region':clabel_names,
               'fs_region':label_names[k+1],
               'hemisphere':h,
               'sizeR':np.sum(label[i]==k+1),
               'totalW':np.nansum(A[:,label[i]==k+1],axis=1),
               'weight':np.nanmean(A[:,label[i]==k+1],axis=1)}
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
    model,info = cio.load_model(fpath + f"/{m_basename}_{extension}_avg")

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

def make_all_weight_maps_WTA():
    make_avrg_weight_map(dataset= "MDTB",extension = '',method="WTA")
    make_avrg_weight_map(dataset= "Demand",extension = '',method="WTA")
    make_avrg_weight_map(dataset= "WMFS",extension = '',method="WTA")
    make_avrg_weight_map(dataset= "Nishimoto",extension = '',method="WTA")
    make_avrg_weight_map(dataset= "Somatotopic",extension = '',method="WTA")
    make_avrg_weight_map(dataset= "IBC",extension = '',method="WTA")
    make_avrg_weight_map(dataset= "HCP",extension = '',method="WTA")

def make_all_weight_maps_L2():
    make_avrg_weight_map(dataset= "Fusion",extension = '06',method="L2Regression")
    make_avrg_weight_map(dataset= "Fusion",extension = '05',method="L2Regression")
    make_avrg_weight_map(dataset= "MDTB",extension = 'A8',method="L2Regression")
    make_avrg_weight_map(dataset= "Demand",extension = 'A8',method="L2Regression")
    make_avrg_weight_map(dataset= "WMFS",extension = 'A8',method="L2Regression")
    make_avrg_weight_map(dataset= "Nishimoto",extension = 'A10',method="L2Regression")
    make_avrg_weight_map(dataset= "Somatotopic",extension = 'A8',method="L2Regression")
    make_avrg_weight_map(dataset= "IBC",extension = 'A6',method="L2Regression")
    make_avrg_weight_map(dataset= "HCP",extension = 'A-2',method="L2Regression")

def make_avrg_weight_map_NNLS(): 
    cifti_img = cs.avrg_weight_map_roi(traindata = 'MdWfIbDeHtNiSoScLa',
                                cortex_roi = "Icosahedron1002",
                                method = 'L2reg',
                                extension='A6_group',
                                cerebellum_roi = "NettekovenSym32",
                                cerebellum_atlas = "MNISymC3",
                                )
    fname = gl.conn_dir + f'/maps/MdWfIbDeHtNiSoScLa_L2reg1002_A2.pscalar.nii'
    cifti_img = cs.sort_roi_rows(cifti_img)
    nb.save(cifti_img,fname)

def comp_weight_stats(): 
    traindata = 'MdWfIbDeHtNiSoScLa'
    cortex_roi = "Icosahedron1002"
    method = 'NNLS'
    stats = 'prob'
    cifti_img = cs.stats_weight_map_cortex(traindata = traindata,
                                cortex_roi = cortex_roi,
                                method = method,
                                extension='A2_group',
                                stats = stats)


    fname = gl.conn_dir + f'/maps/{traindata}_{cortex_roi}_{method}_{stats}.pscalar.nii'
    nb.save(cifti_img,fname)


def plot_surface_stats(traindata,cortex_roi='Icosahedron1002', method='L2reg',extension='A8_avg'):
    model, info = cs.get_model(traindata,cortex_roi, method,extension) 
    
def stats_weight_model(): 
    traindata = 'MdWfIbDeHtNiSoScLa'
    cortex_roi = "Icosahedron1002"
    method = 'NNLS'
    stats = 'prob'
    
    model = cs.get_model(traindata,cortex_roi, method,extension='A2_group')
    nifti = cs.stats_weight_map_cerebellum(traindata = traindata,
                                cortex_roi = cortex_roi,
                                method = method,
                                extension='A2_group',
                                cerebellar_space = 'MNISymC3',
                                stats = stats)
    pass

def make_stats_map(): 
    traindata = 'MdWfIbDeHtNiSoScLa'
    cortex_roi = "Icosahedron1002"
    method = 'NNLS'
    stats = 'prob'
    
    model = cs.get_model(traindata,cortex_roi, method,extension='A2_group')
    nifti = cs.stats_weight_map_cerebellum(traindata = traindata,
                                cortex_roi = cortex_roi,
                                method = method,
                                extension='A2_group',
                                cerebellar_space = 'MNISymC3',
                                stats = stats)
    pass

def do_smooth():
    name = 'MdWfIbDeHtNiSoScLa_NNLS1002_A2' 
    cs.pscalar_to_smoothed_dscalar(name + '.pscalar.nii', 
                                   name + '.dscalar.nii',
                                   sigma = 4.0)  



if __name__ == "__main__":
    # export_model_as_cifti(dataset_name= "Fusion",extension = '06',method="L2Regression")
    # Compute the average connecivity for the model for each cortical parcel
    # ["MDTB","WMFS", "Nishimoto", "Demand", "Somatotopic", "IBC","HCP"],
    do_smooth()
    pass 