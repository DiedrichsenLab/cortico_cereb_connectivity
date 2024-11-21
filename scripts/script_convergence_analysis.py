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
import cortico_cereb_connectivity.evaluation as ev
import cortico_cereb_connectivity.weights as cw
import numpy as np
import SUITPy as suit
import nibabel as nb
import matplotlib
# matplotlib.use('MacOSX')  
import matplotlib.pyplot as plt
import nitools as nt
import seaborn as sb





if __name__ == "__main__":
    weights = cw.load_model_weights('MDTB','all','NNLS','SUIT3','Icosahedron362','A4')
    cerebellum,ainf = am.get_atlas('SUIT3')
    area = cw.calc_area(weights)
    var,std = cw.calc_dispersion(weights,'Icosahedron362')
    T = summarize_measures([area, std],rois=None)
    # plt.figure()
    # plt.subplot(2,2,1)
    plt.ion()
    nii = cerebellum.data_to_nifti(np.mean(area,axis=0))
    flat_data = suit.flatmap.vol_to_surf(nii)
    suit.flatmap.plot(flat_data,colorbar=True)
    plt.title('area')

    # plt.subplot(2,2,2)
    nii = cerebellum.data_to_nifti(np.mean(area,axis=0))
    flat_data = suit.flatmap.vol_to_surf(nii)
    suit.flatmap.plot(flat_data,colorbar=True)
    plt.title('std')







    pass