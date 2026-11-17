# cio.py
# Input-output functions for connectivity model

import Functional_Fusion.atlas_map as am
import cortico_cereb_connectivity.globals as gl
import nibabel as nb
import numpy as np
import warnings
import json
import pickle
import importlib
import h5py

H5_FORMAT = 'cortico_cereb_connectivity.model/1'

def load_model(fname):
    """ Loads model from a combination of a .h5 file and a .json file.
    Reads both the new h5 format (see save_h5_model) and old files written by deepdish.
    Args:
        fname (str): File name without .h5 or .json extension
    Returns:
        model (Model): Model object
        info (dict): Info dictionary
    """
    model = load_h5_model(fname + ".h5")
    with open(fname + ".json") as json_file:
        info = json.load(json_file)
    return model, info

def save_model(model,info,fname):
    """ Saves model to a combinattion of a .h5 file and a .json file.

    Args:
        model (Model): Model object
        info (dict): Info dictionary
        fname (str): File name without .h5 or .json extension
    """
    save_h5_model(model, fname + ".h5")
    with open(fname + ".json", 'w') as fp:
        json.dump(info, fp, indent=4)

def save_h5_model(model, fname, compression=None):
    """ Saves a model object to a plain h5 file (no pickle).
    Every numpy-array attribute becomes a dataset, all other attributes
    (numbers, strings, bools, None, lists) are stored as a JSON string in the
    'params' attribute. The class is stored by module and name.

    Args:
        model (Model): Model object
        fname (str): File name including .h5 extension
        compression (str): h5py compression for the arrays, e.g. 'gzip' (default: None)
    """
    arrays = {}
    params = {}
    for key, value in vars(model).items():
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise TypeError(f"Cannot save object array attribute '{key}' to h5")
            arrays[key] = value
        elif isinstance(value, np.generic):
            params[key] = value.item()
        else:
            params[key] = value
    # Serialize before opening the file, so a bad attribute does not leave a broken file
    params = json.dumps(params)
    with h5py.File(fname, 'w') as f:
        f.attrs['format'] = H5_FORMAT
        f.attrs['module'] = type(model).__module__
        f.attrs['class'] = type(model).__qualname__
        f.attrs['params'] = params
        for key, value in arrays.items():
            f.create_dataset(key, data=value, compression=compression)

def load_h5_model(fname):
    """ Loads a model object from an h5 file.
    Reads files written by save_h5_model, and old files written by deepdish,
    which hold a single pickled model object.

    Args:
        fname (str): File name including .h5 extension
    Returns:
        model (Model): Model object
    """
    with h5py.File(fname, 'r') as f:
        if 'DEEPDISH_IO_VERSION' in f.attrs:
            return _load_deepdish_pickle(f, fname)
        if f.attrs.get('format') != H5_FORMAT:
            raise ValueError(f"{fname} is not a model file")
        cls = getattr(importlib.import_module(f.attrs['module']), f.attrs['class'])
        model = cls.__new__(cls)
        model.__dict__.update(json.loads(f.attrs['params']))
        for key in f.keys():
            model.__dict__[key] = f[key][()]
    return model

def _load_deepdish_pickle(f, fname):
    """ Old deepdish files store the model as a pickled object in the
    PyTables node 'data'. Only use this on files you trust: unpickling can run code.
    """
    if not ('data' in f and f['data'].attrs.get('PSEUDOATOM') == b'object'):
        raise ValueError(f"{fname}: unsupported deepdish file layout")
    return pickle.loads(f['data'][0].tobytes())

def model_to_cifti(weight_matrix, 
                   src_atlas = "fs32k", 
                   trg_atlas = None,
                   src_roi = "Icosahedron1002",
                   trg_roi = None,
                   type = 'scalar'
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
        if (isinstance(trg_roi,list) or isinstance(trg_roi,np.ndarray)):
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
    model = load_h5_model(fpath + f"/{m_basename}{extension}_avg.h5")

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

