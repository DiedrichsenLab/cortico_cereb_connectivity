import numpy as np
import deepdish as dd
import pathlib as Path
import os

import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module
import Functional_Fusion.matrix as fm
import ProbabilisticParcellation.util as ut

import cortico_cereb_connectivity.globals as gl


# make sure you have extrated data in functional fusion framework before running these function


def extract_group_data(dataset="MDTB", ses_id='ses-s1'):
    """
    Extract group data for the dataset
    """
    # get the Dataset class

    Data = fdata.get_dataset_class(gl.base_dir, dataset=dataset)
    
    # get group average. will be saved under <dataset_name>/derivatives/group
    Data.group_average_data(ses_id=ses_id,
                            type="CondAll",
                            atlas='SUIT3')

    Data.group_average_data(ses_id=ses_id,
                            type="CondAll",
                            atlas='fs32k')
    return

# use this to extract data if it's not already extracted


def extract_data(dataset_name, ses_id, type, atlas):
    # create an instance of the dataset class
    dataset = fdata.get_dataset_class(ut.base_dir, dataset=dataset_name)

    # extract data for suit atlas
    dataset.extract_all(ses_id, type, atlas)

    return

# getting cortical maps (from cortical roi to cortical surface)
def cortex_parcel_to_cifti(data, atlas_cortex, parcel_axis_names):
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

    # get the maps for left and right hemispheres
    surf_map = []
    for label in atlas_cortex.label_list:
        # loop over regions within the hemisphere
        label_arr = np.zeros([data.shape[0], label.shape[0]])
        for p in np.arange(1, data.shape[0]):
            for i in np.unique(label):            
                np.put_along_axis(label_arr[p-1, :], np.where(label==i)[0], data[p-1,i-1], axis=0)
        surf_map.append(label_arr)

    cifti_img = atlas_cortex.data_to_cifti(surf_map, row_axis=parcel_axis_names)
    return cifti_img
