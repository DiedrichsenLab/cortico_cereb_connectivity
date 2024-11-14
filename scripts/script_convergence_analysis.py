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


def calc_area(weights,threshold=0):  
    """ Calculate the area of the weights above a threshold for each target voxel
    Args:
        weights (ndarray): weights
        threshold (float): threshold
    Returns:
        area (float): area of the weights above the threshold
    """
    area = np.mean((weights>threshold),axis=weights.ndim-1)

    return area


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
    weights = load_model_weights()
    area = calc_area(weights)
    cerebellum,ainf = am.get_atlas('SUIT3')
    # cortex = am.get_atlas('fs32k')
    nii = cerebellum.data_to_nifti(area.mean(axis=0))
    flat_data = suit.flatmap.vol_to_surf(nii)
    suit.flatmap.plot(flat_data,colorbar=True)
    plt.show()
    pass