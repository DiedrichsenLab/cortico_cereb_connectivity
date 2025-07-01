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
import surfAnalysisPy as surf
import SUITPy as suit 

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
                    stats = 'mean'):
    """ returns cifti image of average cortical input weight"""
    model,info = get_model(traindata,cortex_roi,method,extension)
    if stats == 'mean':
        result = np.mean(model.coef_,axis=0,keepdims=True)
    if stats == 'prob':
        result = np.mean(model.coef_>0,axis=0,keepdims=True)
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]
    cifti_img = cio.model_to_cifti(result,src_roi = label_fs)
    return cifti_img

def stats_weight_map_cerebellum(traindata,
                    cortex_roi = "Icosahedron1002",
                    method = 'L2reg',
                    extension='A8_avg',
                    cerebellar_space = 'MNISymC3',
                    stats = 'mean'):
    """ returns cifti image of average cortical input weight"""
    model,info = get_model(traindata,cortex_roi,method,extension)
    myatlas,_ = am.get_atlas(cerebellar_space)
    if stats == 'mean':
        result = np.mean(model.coef_,axis=1)
    if stats == 'prob':
        result = np.mean(model.coef_>0,axis=1)
    if stats == 'quant':
        result = np.mean(model.coef_>0.05,axis=1)
    nifti_img = myatlas.data_to_nifti(result) 
    return nifti_img

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
    label_suit = gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}_space-{cerebellum_atlas}_dseg.nii"

    # get the average cortical weights for each cerebellar parcel
    atlas_cereb,ainf = am.get_atlas(cerebellum_atlas)
    atlas_cereb.get_parcel(label_suit)
    weights_parcel, labels = fdata.agg_parcels(weights.T, atlas_cereb.label_vector, fcn=np.nanmean)

    # load the lookup table for the cerebellar parcellation to get the names of the parcels
    index,colors,labels = nt.read_lut(gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}.lut")

    cifti_img = cio.model_to_cifti(weights_parcel.T,
                                   src_atlas = "fs32k",
                                   trg_atlas = cerebellum_atlas,
                                   src_roi = label_fs,
                                   trg_roi = labels[1:],
                                   type = 'scalar')
    return cifti_img

def calc_relative_input_size(traindata,
                    cortex_roi = "Icosahedron1002",
                    cerebellum_roi = "NettekovenSym32",
                    cerebellum_atlas = "MNISymC3"):
    """ Returns a dataframe with the relative size and input size of each cerebellar parcel - averaed across hemisphere. 
    This is based on the average group data- assigning each cortical parcel to the cerebellar parcel with the highest correlation. 

    Args:
        traindata (str): name of the training data, e.g. 'MdWfIbDeHtNiSoScLa'
        cortex_roi (str, optional): name of the cortical parcellation. Defaults to "
        Icosahedron1002".
        method (str, optional): method used to train the model. Defaults to 'L2reg'.
        extension (str, optional): extension to the model name. Defaults to 'A8_avg'.
        cerebellum_roi (str, optional): name of the cerebellar parcellation. Defaults to "NettekovenSym32".
        cerebellum_atlas (str, optional): name of the cerebellar atlas. Defaults to "MNISymC3".
    Returns:
        result (pd.DataFrame) 
    """
    # make model name
    # load in the connectivity average connectivity model

    # Load model 
    Cereb = nb.load(f'{gl.conn_dir}/maps/MdWfIbDeHtNiSoScLa_data_cerebellum.dscalar.nii')
    Cort = nb.load(f'{gl.conn_dir}/maps/MdWfIbDeHtNiSoScLa_data_cortex.pscalar.nii')
    
    # get the average cortical weights for each cerebellar parcel
    atlas_cereb,ainf = am.get_atlas(cerebellum_atlas)
    label_cereb = gl.atlas_dir + f"/{ainf['dir']}/atl-{cerebellum_roi}_space-{ainf['space']}_dseg.nii"
    atlas_cereb.get_parcel(label_cereb)
    Cereb_V, labels = fdata.agg_parcels(Cereb.get_fdata(), atlas_cereb.label_vector, fcn=np.nanmean)
    Cereb_V = Cereb_V/np.std(Cereb_V,axis=0,keepdims=True)    
    
    Cort_data = Cort.get_fdata()
    Cort_data = Cort_data/np.std(Cort_data,axis=0,keepdims=True)
    Corr = Cort_data.T @ Cereb_V/Cort_data.shape[0]
    winner = np.argmax(Corr,axis=1)+1 
    size_parcel,_ = fdata.agg_parcels(np.ones(atlas_cereb.P), atlas_cereb.label_vector, fcn=np.size)
    # load the lookup table for the cerebellar parcellation to get the names of the parcels
    index,colors,label_names = nt.read_lut(gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}.lut")

    T = pd.DataFrame()
    for i in range(max(labels)):
        t = {'cereb_region': [label_names[i+1]],
             'size_cereb': size_parcel[i],
             'size_cort': np.sum(winner==i+1),
             'color_r': [colors[i+1,0]],
             'color_g': [colors[i+1,1]],
             'color_b': [colors[i+1,2]]}
        T = pd.concat([T,pd.DataFrame(t)],ignore_index=True)
    return T

def get_weight_by_cortex(traindata = 'MdWfIbDeHtNiSoScLa',
                    cortex_roi = "Icosahedron1002",
                    method = "NNLS",
                    extension = 'A2_group',
                    cerebellum_atlas = "MNISymC3",
                    sum_cortex = 'yeo17',
                    sum_method = 'positive'
                    ):
    """ Make table of the connectivity weights for each cortical parcel,
    averaged across the entire cerebellum.
    """
    model,info  = get_model(traindata,cortex_roi,method,extension,cerebellum_atlas)

    weights = model.coef_

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
                        'name':label_names[1:],
                        'color_r': colors[1:,0],
                        'color_g': colors[1:,1],
                        'color_b': colors[1:,2]})
    
    return T




def pscalar_to_smoothed_dscalar(infile,outfilename=None,sigma=4.0,wdir=f'{gl.conn_dir}/maps'):
    """ Takes a pscalar file, projects it to the full surface and smoothes"""
    hem_name_bm = ['cortex_left','cortex_right']
    if isinstance(infile,str): 
        infile= nb.load(wdir + '/' + infile)
    if not isinstance(infile,nb.Cifti2Image): 
        raise(NameError('Input needs to be cifti image or cifti filename'))
    full_data = nt.surf_from_cifti(infile)
    bm=[]
    for h,hem_name in enumerate(hem_name_bm): 
        bm.append(nb.cifti2.BrainModelAxis.from_mask(np.ones((full_data[h].shape[1],)),hem_name_bm[h]))
    row_axis = infile.header.get_axis(0)
    header = nb.Cifti2Header.from_axes((row_axis,bm[0]+bm[1]))
    data = np.concatenate(full_data,axis=1)
    cifti_img = nb.Cifti2Image(dataobj=data,header=header)
    temp_name =f'{wdir}/temp.dscalar.nii'
    nb.save(cifti_img,temp_name)
    hem_surf = [f'{gl.atlas_dir}/tpl-fs32k/tpl-fs32k_hemi-{h}_inflated.surf.gii' for h in ['L','R']]
    nt.smooth_cifti(temp_name,f'{wdir}/{outfilename}',hem_surf[0],hem_surf[1],surface_sigma=sigma,ignore_zeros=False)
    os.remove(temp_name)

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

def plot_cortical_flatmap(data,axes = None, 
                          cscale=None,
                          cmap='bwr',
                          overlay_type='func'): 
    # determine scaling 
    if cscale is None:
        data_min = np.nanmin(data[0])
        data_max = np.nanmax(data[0])
        cscale = [data_min, data_max]
    
    if axes is None:
        fig, axes = plt.subplots(1, 2,figsize=(10, 4))

    flat = []
    # Use the mirrored flatmap for the left hemisphere
    surf_dir = surf.plot._surf_dir
    flat.append(nb.load(surf_dir + "/fs_L/fs_LR.32k.Lm.flat.surf.gii"))
    flat.append(nb.load(surf_dir + "/fs_R/fs_LR.32k.R.flat.surf.gii"))
    border = []
    border.append(surf_dir + "/fs_L/fs_LR.32k.L.border")
    border.append(surf_dir + "/fs_R/fs_LR.32k.R.border")

    for h in range(2):
        plt.sca(axes[h])
        surf.plot.plotmap(
            data[h].flatten(),
            flat[h],
            underlay=None,
            overlay_type=overlay_type,
            cmap=cmap,
            cscale=cscale,
            borders=border[h],
        )

def plot_cortical_inflated(data,axes = None, cscale=None):
    adir = am.default_atlas_dir 
    vinf = [f"{adir}/tpl-fs32k/tpl-fs32k_hemi-L_veryinflated.surf.gii",
            f"{adir}/tpl-fs32k/tpl-fs32k_hemi-R_veryinflated.surf.gii"] 
    depth = f"{adir}/tpl-fs32k/tpl-fs32k_sulc.dscalar.nii"

    surf_data = []
    for hem in range(2):
        gifti = nb.load(vinf[hem])
        surf_data.append([gifti.darrays[0].data,gifti.darrays[1].data])
    Depth = nb.load(depth)
    depth_data = nt.surf_from_cifti(Depth)
    if axes is None:
        fig, axes = plt.subplots(1, 4, subplot_kw={'projection': '3d'},figsize=(8, 4))
    
    hemi = ['left', 'right']
    view = ['lateral', 'medial']

    if cscale is None:
        data_min = np.nanmin(data[0])
        data_max = np.nanmax(data[0])
        cscale = [data_min, data_max]
        
    for hem in range(2):
        for row in range(2):
            npl.plot_surf(
                surf_data[hem],
                data[hem],
                depth_data[hem],
                hemi=hemi[hem],
                view=view[row],
                cmap="hot",
                vmin=cscale[0],
                vmax=cscale[1],
                axes=axes[row*2 + hem] 
            )

