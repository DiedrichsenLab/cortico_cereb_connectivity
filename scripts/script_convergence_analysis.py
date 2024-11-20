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
import matplotlib
matplotlib.use('MacOSX')  
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import nitools as nt
import seaborn as sb

def calc_area(weights,threshold=0):  
    """ Calculate the area of the weights above a threshold for each target voxel
    the area is expressed as proportion of the entire cortical surface

    Args:
        weights (ndarray): weights
        threshold (float): threshold
    Returns:
        area (float): area of the weights above the threshold
    """
    area = np.mean((weights>threshold),axis=weights.ndim-1)
    area[area==0] = np.nan
    return area

def calc_dispersion(weights,parcel,threshold=0): 
    """Caluclate spherical dispersion for the connectivity weights

    Args:
        weights (np array): N x P or nsubj x N x P array of data to claculate dispersion on  
        cortex (str): cortical 

    Returns:
        dataframe (pd dataframe)
    """
    # get data
    num_parcel = weights.shape[-1]
    weights = np.nan_to_num(weights)

    # Read the parcellation and coordinate on spherical surface into atlas space
    hem_names = ['L', 'R']
    atlas_dir = gl.atlas_dir + '/tpl-fs32k'
    fs32, ainf = am.get_atlas('fs32k')
    parcel = [atlas_dir + f"/{parcel}.{h}.label.gii" for h in hem_names]
    sphere = [atlas_dir + f"/tpl-fs32k_hemi-{h}_sphere.surf.gii" for h in hem_names]

    lable_vec = fs32.get_parcel(parcel)
    coords = fs32.read_data(sphere).T
    parcel_coords,_ = fdata.agg_parcels(coords, fs32.label_vector)
    parcel_hem,_ = fdata.agg_parcels(fs32.structure_index, fs32.label_vector)

    # Calcualte the dispersion per hemisphere
    for h,hem in enumerate(hem_names):
        indx =  parcel_hem == h
        # Calculate spherical STD as measure
        # Get coordinates and move back to 0,0,0 center
        coord_hem = parcel_coords[:,indx].copy()
        coord_hem[0,:]=coord_hem[0,:]-(h*2-1)*500

        # Now compute a weoghted spherical mean, variance, and STD
        # For each tessel, the weigth w_i is the connectivity weights with negative weights set to zero
        # also set the sum of weights to 1
        w = weights[...,indx].copy()
        w[w<0]=0
        sum_w = w.sum(axis=-1,keepdims=True)
        w = w /sum_w
        # print(sum_w.shape)

        # We then define a unit vector for each tessel, v_i:
        v = coord_hem.copy().T
        v=v / np.sqrt(np.sum(v**2,axis=1,keepdims=1))

        # Weighted average vector mv_i = sum(w_ij * v_ij)
        # R is the length of this average vector
        vw = v*w[...,np.newaxis]
        mv = np.nansum(vw ,axis=-2)
        R = np.sqrt(np.sum(mv**2,axis=-1))
        V = 1-R # This is the Spherical variance
        Std = np.sqrt(-2*np.log(R)) # This is the spherical standard deviation

        # Check with plot
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(v[0,:],v[1,:],v[2,:])
        # ax.scatter(mean_v[0],mean_v[1],mean_v[2])
            # pass
        pass
        # df1 = pd.DataFrame({'Variance':V,'Std':Std,'hem':h*np.ones((num_roi,)),'roi':np.arange(num_roi)+1,'sum_w':sum_w})
        # df = pd.concat([df,df1])
    return df




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
        map,lv = fdata.combine_parcel_labels(labels,rois,catlas.label_vector)
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
    dispersion = calc_dispersion(weights,'Icosahedron362')
    T = summarize_measure([area, dispersion],rois=None)
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