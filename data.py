import numpy as np
import deepdish as dd
import pathlib as Path
import sys
import os
sys.path.append('..')
import Functional_Fusion as ff
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.matrix as fm

# set base directory of the functional fusion 
base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'
conn_dir = '/srv/diedrichsen/data/Cerebellum/connectivity/'

# make sure you have extrated data in functional fusion framework before running these function
def extract_group_data(dataset = "MDTB", ses_id = 'ses-s1'):
    """
    Extract group data for the dataset
    """
    # get the Dataset class
    Data = fdata.get_dataset_class(base_dir, dataset=dataset)
    
    # get group average. will be saved under <dataset_name>/derivatives/group
    Data.group_average_data(ses_id=ses_id,
                                 type="CondAll",
                                 atlas='SUIT3')

    Data.group_average_data(ses_id=ses_id,
                                 type="CondAll",
                                 atlas='fs32k')
    return

# use this to extract data if it's not already extracted
def extract_data(base_dir, dataset_name, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = fdata.get_dataset_class(base_dir, dataset = dataset_name)

    # extract data for suit atlas
    dataset.extract_all(ses_id,type,atlas)

    return

# get data tensor and save it
def save_data_tensor(dataset = "WMFS",
                    atlas='SUIT3',
                    ses_id='ses-s1',
                    type="CondHalf", 
                    outpath = conn_dir):
    """
    create a data tensor (n_subj, n_contrast, n_voxel) and saves it

    """
    # using get_dataset from functional fusion
    data_tensor, info, Data = fdata.get_dataset(base_dir,
                                                dataset,
                                                atlas=atlas,
                                                sess=ses_id,
                                                type="CondHalf", 
                                                info_only=False)
    # check if directory exists
    is_dir = os.path.exists(outpath + dataset)
    if not is_dir:
        # Create a new directory because it does not exist
        os.makedirs(outpath + dataset)
    filename = outpath + dataset+ f'/{dataset}_{atlas}_{ses_id}_{type}.npy'
    np.save(filename,data_tensor)

    return data_tensor

# getting cortical maps (from cortical roi to cortical surface)
def convert_cortex_to_cifti(data, atlas_cereb, atlas_cortex, fcn = np.nanmean):
    """
    calculating a certain measure over voxels within a cerebellar parcel
    Args: 
        data (np.ndarray) - data matrix you want to map (#cerebellar voxel-by-#cortical regions)
        atlas_cereb (atlasVolumetric) - atlas object after .get_parcel is done
        atlas_cortex (atlasSurface) - atlas object after .get_parcel is done
        fcn (function object) - function to be applied 
    Returns:
        cifti_img (nb.cifti2) - dscalar image file with maps for each cerebellar parcel
    """
    pa_cereb = atlas_cereb.get_parcel_axis() # will be used to get the names of the parcels

    # making sure parcels are created in the atlases
    try: # only runs if you have done .get_parcel
        # get the maps for left and right hemispheres
        surf_map = []
        for label in atlas_cortex.label_list:
            # loop over regions within the hemisphere
            label_arr = np.zeros([atlas_cereb.n_labels, label.shape[0]])
            for p in np.arange(1, atlas_cereb.n_labels+1):
                vox = atlas_cereb.label_vector == p
                # get average connectivity weights
                data_region = np.fcn(data[vox, :], axis = 0)
                for i in np.unique(label):            
                    np.put_along_axis(label_arr[p-1, :], np.where(label==i)[0], data_region[i-1], axis=0)
            surf_map.append(label_arr)

        cifti_img = atlas_cortex.data_to_cifti(surf_map, row_axis=pa_cereb.name)

    except AttributeError: # you haven't done .get_parcel
        print("RUN <atlas>.get_parcel(label_file) first")
    return cifti_img