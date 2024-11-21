import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import Functional_Fusion.dataset as fdata 
import Functional_Fusion.atlas_map as am  
import numpy as np 
import nitools as nt
import pandas as pd
import warnings


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
    """Caluclate spherical variance, and STD for the connectivity weights
    assuming the weights come from a specific parcellation of the fs32K atlas
    Args:
        weights (np array): N x P or nsubj x N x P array of data to claculate dispersion on  
        parcel (str): parcellation map 'Icosahedron162'  

    Returns:
        variance (ndarray): spherical variance of the weights
        std (ndarray): spherical standard deviation of the weights (N or nsubj x N)
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

    # Calculate the length of average directional vector for each hemisphere
    Rshape = (2,)+weights.shape[:-1]
    R  = np.zeros(Rshape)
    Sum_w  = np.zeros(Rshape)
    for h,hem in enumerate(hem_names):
        indx =  parcel_hem == h

        # Calculate spherical STD as measure
        # Get coordinates and define a unit vector for each tessel, v_i:
        v = parcel_coords[:,indx].T.copy()
        v=v / np.sqrt(np.sum(v**2,axis=1,keepdims=1))

        # For each tessel, the weigth w_i is the connectivity weights with negative weights set to zero
        # also set the sum of weights to 1
        w = weights[...,indx].copy()
        w[w<0]=0
        sum_w = w.sum(axis=-1,keepdims=True)
        with warnings.catch_warnings(action="ignore"):
            w = w /sum_w
        Sum_w[h] = sum_w.squeeze()

        # Weighted average vector mv_i = sum(w_ij * v_ij)
        # R is the length of this average vector
        vw = v*w[...,np.newaxis]
        mv = np.sum(vw ,axis=-2)
        R[h,:] = np.sqrt(np.sum(mv**2,axis=-1))

        # Check with plot
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(v[:,0],v[:,1],v[:,2])
        # ax.scatter(mean_v[0],mean_v[1],mean_v[2])
            # pass
        # df1 = pd.DataFrame({'Variance':V,'Std':Std,'hem':h*np.ones((num_roi,)),'roi':np.arange(num_roi)+1,'sum_w':sum_w})
        # df = pd.concat([df,df1])


    # Weighting factor for each hemisphere
    with warnings.catch_warnings(action="ignore"):
        Sum_w = Sum_w/np.sum(Sum_w,axis=0)
    R=np.sum(Sum_w*R,axis=0)
    V = 1-R # This is the Spherical variance    
    Std = np.sqrt(-2*np.log(R)) # This is the spherical standard deviation

    return V,Std

def load_model_weights(dataset='MDTB',
                       train_ses='ses-s1',
                       method='NNLS',
                       cerebellum='SUIT3',
                       parcellation='Icosahedron162',
                       ext='A6'):
    """ Load the weights of the model for a specific dataset, method, parcellation and extension"""

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


def summarize_measures(data,
                      dataname = ['area','std'],
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
        colors=[colors[np.where(map==i)[0][0],:] for i in range(len(rois))]
    else:
        lv = catlas.label_vector
        rois = labels
        


    df_list = []
    for i in range(len(data)):
        data_p, labels = fdata.agg_parcels(data[i], lv, fcn=np.nanmean)

        T = pd.DataFrame(data_p,columns=rois[1:])
        T = T.melt(value_vars=rois[1:])
        T = T.rename(columns={"variable": "roi","value":dataname[i]})
        df_list.append(T)

    df_list[0][dataname[1]] = df_list[1][dataname[1]]   

    return df_list[0],colors
