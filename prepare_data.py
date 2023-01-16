import numpy as np
import deepdish as dd
import pathlib as Path
import sys
sys.path.append('..')
import Functional_Fusion as ff
import Functional_Fusion.atlas_map as at # from functional fusion module
import Functional_Fusion.dataset as fdata # from functional fusion module

# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'

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
                    type="CondHalf"):
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
    filename = Data.base_dir + f'/{dataset}_{atlas}_{ses_id}_{type}.npy'
    np.save(filename,data_tensor)
    return 
