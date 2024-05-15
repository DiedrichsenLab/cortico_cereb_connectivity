# cio.py
# Input-output functions for connectivity model

import Functional_Fusion.atlas_map as am
import nibabel as nb
import numpy as np
import warnings 

def model_to_cifti(weight_matrix, 
                   src_atlas, 
                   trg_atlas,
                   src_roi = "Icosahedron1002",
                   trg_roi = None,
                   type = 'conn'
                ):
    """ Make a cifti image for the connectivity weights
    Src is saved in the axis[1] (columns)
    Trg is saved in the axis[0] (rows)
    If ROI is given it uses a parcel axis
    If ROI is not given it uses a dense axis

    Args:
        weight_matrix (ndarray): connectivity weights (trg x src) 
        src_atlas (am.atlas or str): Atlas for the source space
        trg_atlas (am.atlas or str): Atlas for the target space
        src_roi (str): ROI for the source space (list of gifti-names)
        trg_roi (str): ROI for the target space
        type (str): type of cifti-file. 'conn' or 'scale'
    Returns:
        cifti_img (nibabel.Cifti2Image) cifti image 
    """
   
    # Getting the atlases 
    if isinstance(src_atlas, str):
        src_atlas, _ = am.get_atlas(src_atlas)
    if isinstance(trg_atlas, str):
        trg_atlas, _ = am.get_atlas(trg_atlas)

    # Getting ROIs src_atlas
    if src_roi is not None:
        src_atlas.get_parcel(src_roi)
        src_axis = src_atlas.get_parcel_axis()
    else:
        src_axis = src_atlas.get_brain_model_axis()

    # Getting ROIs for trg_atlas
    if type == 'conn':
        if trg_roi is not None:
            trg_atlas.get_parcel(trg_roi)
            trg_axis = trg_atlas.get_parcel_axis()
        else:
            trg_axis = trg_atlas.get_brain_model_axis()
    elif type == 'scalar':
        if isinstance(trg_roi,list):
            trg_axis = nb.cifti2.ScalarAxis(trg_roi)
        elif trg_roi is not None:
            _,labels = trg_atlas.get_parcel(trg_roi)
            trg_axis = nb.cifti2.ScalarAxis(labels[labels!=0])
        else:
            trg_axis = nb.cifti2.ScalarAxis(np.arange(weight_matrix.shape[0]))

    # Creating the cifti image
    header = nb.Cifti2Header.from_axes((trg_axis, src_axis))
    cifti_img = nb.Cifti2Image(weight_matrix, header=header)

    return cifti_img


def export_model_as_cifti(method = "L2Regression",
                    cortex_roi = "Icosahedron1002",
                    cerebellum_atlas = "SUIT3",
                    extension = 'A8',
                    dataset_name = "MDTB",
                    ses_id = "all",
                    ):
    """ Loads a connectivty model and saves it as a dpconn and pdconn cifti file. The dense-axis refers to the cerebellar voxels, the parcel axis refers to the cortical parcels
    
    Args:
        method (str) - connectivity method used to estimate weights
        cortex_roi (str) - cortical tessellation/roi 
        cerebellum_atlas (str) - cerebellar atlas "SUIT3" or "MNISym2
        dataset_name (str) - name of the dataset or fusion
        ses_id (str) - "all" for aggregated model over sessions
        """
    # make model name
    m_basename = f"{dataset_name}_{ses_id}_{cortex_roi}_{method}"
    # load in the connectivity average connectivity model
    fpath = gl.conn_dir + f"/{cerebellum_atlas}/train/{m_basename}"

    # load the avg model
    if (len(extension) > 0) and extension[0] != "_":
        extension = "_" + extension
    model = dd.io.load(fpath + f"/{m_basename}{extension}_avg.h5")

    # get the weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        weights = model.coef_/model.scale_

    # label file for the cortex
    label_fs = [gl.atlas_dir + f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    cifti_img = model_to_cifti(weights,
                                   src_atlas = "fs32k",
                                   trg_atlas = cerebellum_atlas,
                                   src_roi = label_fs,
                                   trg_roi = None,
                                   type = 'conn')

    fname = gl.conn_dir + f'/{"maps"}/{dataset_name}_{method[:2]}_{cerebellum_atlas}_{cortex_roi}.pdconn.nii'
    nb.save(cifti_img,fname)


    cifti_img = model_to_cifti(weights.T,
                                   src_atlas = cerebellum_atlas,
                                   trg_atlas = "fs32k",
                                   src_roi = None,
                                   trg_roi = label_fs,
                                   type = 'conn')

    fname = gl.conn_dir + f'/{"maps"}/{dataset_name}_{method[:2]}_{cerebellum_atlas}_{cortex_roi}.dpconn.nii'
    nb.save(cifti_img,fname)

    return

