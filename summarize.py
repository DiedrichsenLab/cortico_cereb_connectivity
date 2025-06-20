""" Summarize and describe the connectivity weights""" 
"""
Functions to summarize and plot the group average weights based on cerebellar ROIs. 
"""
import os
import numpy as np
import deepdish as dd
import pandas as pd
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.atlas_map as am
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.data as cdata
import cortico_cereb_connectivity.cio as cio
import matplotlib.pyplot as plt
import nitools as nt
import nilearn.plotting as npl
from pathlib import Path
import warnings

def get_model(traindata,cortex_roi,method,extension,cerebellum_atlas="MNISymC3"): 
    """ Loads a model 
    Args:
        traindata (str): name of the training data, e.g. 'MdWfIbDeHtNiSoScLa'
        cortex_roi (str, optional): name of the cortical parcellation. Defaults to "Icosahedron1002".
        method (str, optional): method used to train the model. Defaults to 'L2reg'.
        extension (str, optional): extension to the model name. Defaults to 'A8_avg'.
    """
    mroot = f"{traindata}_{cortex_roi}_{method}"
    model_name = f"{mroot}_{extension}"
    fpath = gl.conn_dir + f"/{cerebellum_atlas}/train/{mroot}"
    # load the avg model
    model,info = cio.load_model(fpath + f"/{model_name}")   
    return model, info



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

def stats_weight_map_cortex(traindata,
                    cortex_roi = "Icosahedron1002",
                    method = 'L2reg',
                    extension='A8_avg',
                    stats = np.mean):
    """ returns cifti image of average cortical input weight"""
    model,info = get_model(traindata,cortex_roi,method,extension)
    result = stats(model.coef_,axis=0,keepdims=True)
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]
    cifti_img = cio.model_to_cifti(result,src_roi = label_fs)
    return cifti_img

def avrg_weight_map_roi(traindata,
                    cortex_roi = "Icosahedron1002",
                    method = 'L2reg',
                    extension='A8_avg',
                    cerebellum_roi = "NettekovenSym32",
                    cerebellum_atlas = "MNISymC3"):
    """ Makes cifti image with the cortical maps average connectivity weights for the different cerebellar parcels - it uses the average connectivity weights (across subjects)

    Args:
        traindata (str): name of the training data, e.g. 'MdWfIbDeHtNiSoScLa'
        cortex_roi (str, optional): name of the cortical parcellation. Defaults to "
        Icosahedron1002".
        method (str, optional): method used to train the model. Defaults to 'L2reg'.
        extension (str, optional): extension to the model name. Defaults to 'A8_avg'.
        cerebellum_roi (str, optional): name of the cerebellar parcellation. Defaults to "NettekovenSym32".
        cerebellum_atlas (str, optional): name of the cerebellar atlas. Defaults to "MNISymC3".
    Returns:
        cifti_img (nibabel.Cifti2Image) pscalar cifti image for the cortical maps. ready to be saved!
    """
    # make model name
    # load in the connectivity average connectivity model

    # Load model 
    model,info = get_model(traindata,cortex_roi,method,extension,cerebellum_atlas)

    # get the weights
    if hasattr(model,'scale_'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=RuntimeWarning)
            weights = model.coef_/model.scale_
    else:
        weights = model.coef_

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





def export_model(model_dir = "MDTB_all_Icosahedron1002_L2Reg",
                    model_name = "MDTB_all_Icosahedron1002_L2Reg_A8",
                    cortex_roi = "Icosahedron1002",
                    type='pdconn'):
    model_path = os.path.join(gl.conn_dir,'SUIT3','train',model_name)
    fname = model_path + f"/{model_name}_{model_ext}"
    M,info = cio.load_model(fname)
    adir = am.default_atlas_dir
    src_roi = [f"{adir}/tpl-{info['cortex']}/Icosahedron1002.L.label.gii",
               f"{adir}/tpl-{info['cortex']}/Icosahedron1002.R.label.gii"]
    C = M.to_cifti(src_atlas=info['cortex'],
                    trg_atlas=info['cerebellum'],
                    src_roi=src_roi,
                    fname=f'data/{file_name}',
                    dtype = 'float32')
    return C 

def plot_cortical_flatmap(axes, data): 
    # Now do connectivity maps
    weights = nt.cifti.surf_from_cifti(conn_map)
    sc = conn_map.header.get_axis(0).name
    cidx = np.empty((2,), dtype=int)
    for c in range(2):
        cidx[c] = np.where(sc == labels[iidx[c]])[0][0]

    flat = []
    # Use the mirrored flatmap for the left hemisphere
    flat.append(nb.load(surf_dir + "/fs_L/fs_LR.32k.Lm.flat.surf.gii"))
    flat.append(nb.load(surf_dir + "/fs_R/fs_LR.32k.R.flat.surf.gii"))
    border = []
    border.append(surf_dir + "/fs_L/fs_LR.32k.L.border")
    border.append(surf_dir + "/fs_R/fs_LR.32k.R.border")

    axH = np.empty((2, 2), dtype=object)
    axH[0, 0] = fig.add_subplot(spec[0, 0:2])
    axH[1, 0] = fig.add_subplot(spec[1, 0:2])
    axH[0, 1] = fig.add_subplot(spec[0, 4:])
    axH[1, 1] = fig.add_subplot(spec[1, 4:])

    for h in range(2):
        for c in range(2):
            plt.axes(axH[h, c])
            surf.plot.plotmap(
                weights[h][cidx[c], :],
                flat[h],
                underlay=None,
                overlay_type="func",
                cmap="bwr",
                cscale=[-0.002, 0.002],
                borders=border[h],
            )

def plot_cortical_inflated(axes,data):
    adir = am.default_atlas_dir 
    vinf = [f"{adir}/tpl-fs32k/tpl-fs32k_hemi-L_veryinflated.surf.gii",
            f"{adir}/tpl-fs32k/tpl-fs32k_hemi-R_veryinflated.surf.gii"] 
    depth = f"{adir}/tpl-fs32k_sulc.dscalar.nii"
    
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    
    npl.plot_surf(
        vinf[0],
        data[0],
        depth,
        hemi = 'left',
        view = 'lateral',
        cmap="hot",
        vmin=0,
        vmax=0.03) 
    pass
