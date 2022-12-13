# import libraries and packages
import os
import pandas as pd
import pathlib as Path
import numpy as np
import deepdish as dd
import scipy
import h5py
from SUITPy import flatmap
import nibabel as nb

# Import module as such - no need to make them a class


"""Main module for getting data to be used for running connectivity models.

   @authors: Maedbh King, Ladan Shahshahani, Jörn Diedrichsen

  Typical usage example:
  data = Dataset('sc1','glm7','cerebellum_suit','s02')
  data.load_mat() # Load from Matlab 
  X, INFO = data.get_data(averaging="sess") # Get numpy 

  Group averaging: 
  data = Dataset(subj_id = const.return_subjs) # Any list of subjects will do 
  data.load_mat()                             # Load from Matlab
  data.average_subj()                         # Average 

  Saving and loading as h5: 
  data.save(dataname="group")     # Save under new data name (default = subj_id)
  data = Dataset('sc1','glm7','cerebellum_suit','group')
  data.load()

"""

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'

class ConnData:
    """Dataset class, holds betas for one region, one experiment, one subject for connectivity modelling.

    Attributes:
        exp: A string indicating experiment.
        glm: A string indicating glm.
        roi: A string indicating region-of-interest.
        subj_id: A string for subject id - if the subj_id is a list of strings, the data will be averaged across these subjects. Thus, to get group-averaged data, set subj_id = const.return_subj
        data: None
    """

    def __init__(self, 
                 experiment="Demand", 
                 ses_id = 1, 
                 cerebellum="SUIT3", 
                 cortex = "Icosahedron-42_Sym", 
                 type = "CondHalf"):
        """Inits Dataset."""
        self.exp = experiment
        self.ses_id = f"ses-{ses_id:02d}"
        self.cerebellum = cerebellum
        self.cortex = cortex
        self.type = type
        self.data = None

        # create an instance of functional fusion class DataSet
        self.ds = experiment(os.path.join(base_dir, self.exp))

        # get the participants list 
        self.participant_id = self.ds.get_participants()

    def load_Y(self, subj_id):
        """
        Loads in the cifti file
        """
        self.subj_id = subj_id
        # load the curresponding info structure
        self.info = pd.read_csv(os.path.join(self.ds.data_dir, f"{self.subj_id}_{self.ses_id}_info-{self.type}.tsv"))

        # load the extracted data
        fname = f"{self.subj_id}_space-{self.cerebellum}_{self.ses_id}_{self.type}.dscalar.nii"
        fdir = os.path.join(self.ds.data_dir)

        self.data_cifti = nb.cifti2.load(os.path.join(fdir, fname))
        self.data = self.data_cifti.get_fdata()

    def load_X(self, subj_id, parcel):
        self.subj_id = subj_id 

        # load the curresponding info structure
        self.info = pd.read_csv(os.path.join(self.ds.data_dir, f"{self.subj_id}_{self.ses_id}_info-{self.type}.tsv"))

        fname = f"{self.subj_id}_space-fs32k_{self.ses_id}_{self.type}.dscalar.nii"
        fdir = os.path.join(self.ds.data_dir)

        self.data_cifti = nb.cifti2.load(os.path.join(fdir, fname))
        self.data = self.data_cifti.get_fdata()

        # get brain models
        self.bmf = self.data_cifti.header.get_axis(1)

        data_list = []
        hemi = ['L', 'R']
        for idx, (nam,slc,bm) in enumerate(self.bmf.iter_structures()):
            # get the data corresponding to the brain structure
            data_hemi = self.data[:, slc]

            # get name to be passed on to the AtlasSurfaceParcel object
            name = nam[16:].lower()

            # get the label file for the parcellation
            label_img = os.path.join(base_dir, 'Atlases', 'tpl-fs32k', f'{parcel}.32k.{hemi[idx]}.label.gii')
            
            # get the mask file for the hemisphere
            mask_img = os.path.join(base_dir, 'Atlases', 'tpl-fs32k', f'tpl-fs32k_hemi-{hemi[idx]}_mask.label.gii')

            Parcel = AtlasSurfaceParcel(name,label_img,mask_img)
            data_list.append(Parcel.agg_data(data_hemi))

        # concatenate into a single array
        self.data = np.concatenate(data_list, axis = 1)
        
    def save(self, dataname = None, filename=None):
        """Save the content of the data set in a dict as a hpf5 file.

        Args:
            dataname (str): default is subj_id - but can be set for group data
            filename (str): by default will be set to something automatic 
        Returns:
            saves dict to disk
        """
        if filename is None:
            if dataname is None: 
                if type(self.subj_id) is list: 
                    raise(NameError('For group data need to set data name'))
                else: 
                    dataname = self.subj_id
            dirs = const.Dirs(exp_name=self.exp, glm=self.glm)
            fname = "Y_" + self.glm + "_" + self.roi + ".h5"
            fdir = dirs.beta_reg_dir / dataname
        dd.io.save(fdir / fname, vars(self), compression=None)

    def weight(self, subset = None):

        # load the filtered design matrix
        self.XX = np.load(os.path.join(self.ds.estimates_dir, f'/{self.subj_id}_{self.ses_id}_designmatrix.npy'))
        # Now weight the different betas by the variance that they predict for the time series.
        # This also removes the mean of the time series implictly.
        # Note that weighting is done always on the average regressor structure, so that regressors still remain exchangeable across sessions
        XXm = np.mean(self.XX, 0)
        ind = np.where((self.info.run==1) & subset)[0]
        XXm = XXm[ind, :][:, ind]  # Get the desired subset only
        XXs = scipy.linalg.sqrtm(XXm)  # Note that XXm = XXs @ XXs.T
        for r in np.unique(self.info["run"]):  # WEight each run/session seperately
            idx = self.info.run == r
            data[idx, :] = XXs @ data[idx, :]

        # Data should be imputed if there are nan values
        data = np.nan_to_num(data)
        return data
    
    def average_subj(self): 
        """
            Averages data across subjects if data is 3-dimensional
        """
        if self.data.ndim == 2: 
            raise NameError('data is already 2-dimensional')
        self.data = np.nanmean(self.data, axis = 0)

    def get_data():
        return

def get_distance_matrix(roi):
    """
    Args:
        roi (string)
            Region of interest ('cerebellum_suit','tessels0042','yeo7')
    Returns
        distance (numpy.ndarray)
            PxP array of distance between different ROIs / voxels
    """
    dirs = const.Dirs(exp_name="sc1")
    group_dir = os.path.join(dirs.reg_dir, 'data','group')
    if (roi=='cerebellum_suit'):
        reg_file = os.path.join(group_dir,'regions_cerebellum_suit.mat')
        region = cio.read_mat_as_hdf5(fpath=reg_file)["R"]
        coord = region.data
    else:
        coordHem = []
        parcels = []
        for h,hem in enumerate(['L','R']):
            # Load the corresponding label file
            label_file = os.path.join(group_dir,roi + '.' + hem + '.label.gii')
            labels = nib.load(label_file)
            roi_label = labels.darrays[0].data

            # Load the spherical gifti
            sphere_file = os.path.join(group_dir,'fs_LR.32k.' + hem + '.sphere.surf.gii')
            sphere = nib.load(sphere_file)
            vertex = sphere.darrays[0].data

            # To achieve a large seperation between the hemispheres, just move the hemispheres apart 50 cm in the x-coordinate
            vertex[:,0] = vertex[:,0]+(h*2-1)*500

            # Loop over the regions > 0 and find the average coordinate
            parcels.append(np.unique(roi_label[roi_label>0]))
            num_parcels = parcels[h].shape[0]
            coordHem.append(np.zeros((num_parcels,3)))
            for i,par in enumerate(parcels[h]):
                coordHem[h][i,:] = vertex[roi_label==par,:].mean(axis=0)

        # Concatinate these to a full matrix
        num_regions = max(map(np.max,parcels))
        coord = np.zeros((num_regions,3))
        # Assign the coordinates - note that the
        # Indices in the label files are 1-based [Matlab-style]
        # 0-label is the medial wall and ignored!
        coord[parcels[0]-1,:]=coordHem[0]
        coord[parcels[1]-1,:]=coordHem[1]

    # Now get the distances from the coordinates and return
    Dist = eucl_distance(coord)
    return Dist, coord

def eucl_distance(coord):
    """
    Calculates euclediand distances over some cooordinates
    Args:
        coord (ndarray)
            Nx3 array of x,y,z coordinates
    Returns:
        dist (ndarray)
            NxN array pf distances
    """
    num_points = coord.shape[0]
    D = np.zeros((num_points,num_points))
    for i in range(3):
        D = D + (coord[:,i].reshape(-1,1)-coord[:,i])**2
    return np.sqrt(D)
