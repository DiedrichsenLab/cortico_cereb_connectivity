"""
This script is to determine if we can replicate the convergence results from the King et al. 2023 paper, using the new datasets, and the new models. 
"""
import os
import pandas as pd
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am  
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import numpy as np
import SUITPy as suit
import nibabel as nb
import matplotlib.pyplot as plt
import nitools as nt
import seaborn as sb

def calc_area(weights,threshold=0):  
    """ Calculate the area of the weights above a threshold for each target voxel
    Args:
        weights (ndarray): weights
        threshold (float): threshold
    Returns:
        area (float): area of the weights above the threshold
    """
    area = np.mean((weights>threshold),axis=weights.ndim-1)
    area[area==0] = np.nan
    return area

def summarize_measure(data,
                      cerebellum_roi='NettekovenSym32',
                      cerebellum_atlas='SUIT3',
                      rois=['0','A..','M..','D..','S..']):
    """ Averages the data over the cerebellar regions and returns as a dataframe. Uses hierarchical region index."""
    label_suit = gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}_space-SUIT_dseg.nii"

    # get the average cortical weights for each cerebellar parcel
    catlas,ainf = am.get_atlas(cerebellum_atlas)
    catlas.get_parcel(label_suit)

    # load the lookup table for the cerebellar parcellation to get the names of the parcels
    index,colors,labels = nt.read_lut(gl.atlas_dir + f"/tpl-SUIT/atl-{cerebellum_roi}.lut")

    if rois is not None: 
        lv = fdata.combine_parcel_labels(labels,catlas.label_vector,rois)
    else:
        lv = catlas.label_vector
        rois = labels

    data_p, labels = fdata.agg_parcels(data, lv, fcn=np.nanmean)

    # create a dataframe
    T = pd.DataFrame(data_p,columns=rois[1:])
    T = T.melt(value_vars=rois[1:])
    T = T.rename(columns={"variable": "roi"})
    # nii = catlas.data_to_nifti(catlas.label_vector.astype(np.uint16))
    # flat_data = suit.flatmap.vol_to_surf(nii,stats='mode',ignore_zeros=True)
    # suit.flatmap.plot(flat_data,overlay_type='label',colormap=colors)
    # nii = catlas.data_to_nifti(lv.astype(np.uint16))
    # flat_data = suit.flatmap.vol_to_surf(nii,stats='mode',ignore_zeros=True)
    # suit.flatmap.plot(flat_data,overlay_type='label',colormap='tab20')
    # plt.show()
    return T


def load_model_weights(dataset='MDTB',
                       train_ses='ses-s1',
                       method='NNLS',
                       cerebellum='SUIT3',
                       parcellation='Icosahedron162',
                       ext='A6'):
    mdtb=fdata.get_dataset_class(gl.base_dir,'MDTB')
    sinfo = mdtb.get_participants()
    dirname,mname = rm.get_model_names(train_dataset = dataset,
            train_ses = train_ses,
            method = method,
            parcellation = parcellation,
            ext_list=[ext])
    models,info=rm.get_ind_models(dirname[0],mname[0],
                                  sinfo.participant_id,
                                  cerebellum)
    weights = [m.coef_ for m in models]
    weights = np.stack(weights,axis=0)
    return weights


if __name__ == "__main__":
    weights = load_model_weights('MDTB','all','NNLS','SUIT3','Icosahedron362','A4')
    area = calc_area(weights)
    T = summarize_measure(area,rois=None)
    plt.figure()
    sb.barplot(T,x='roi',y='value')
    plt.show()

    cerebellum,ainf = am.get_atlas('SUIT3')
    # cortex = am.get_atlas('fs32k')
    nii = cerebellum.data_to_nifti(np.mean(area,axis=0))
    flat_data = suit.flatmap.vol_to_surf(nii)
    suit.flatmap.plot(flat_data,colorbar=True)
    plt.show()

    pass