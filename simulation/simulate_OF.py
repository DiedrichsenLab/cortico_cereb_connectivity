import numpy as np
import pandas as pd
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import Functional_Fusion.dataset as fdata
import nibabel as nb
import os


def load_X_data(traindata=gl.traindata_string()):
    cifti_img = nb.load(gl.conn_dir + f'/maps/{traindata}_data_cortex.pscalar.nii')
    avrg_data = cifti_img.get_fdata().squeeze()
    return avrg_data

def get_dims():

    return 

def generate_Y_data(N, P):
    # Simulate generating Y data
    Y = np.zeros((sum(N), P))
    start = 0
    for n in N:
        Y_n = np.random.rand(n, P)
        Y_n = rm.std_data(Y_n, 'global')
        Y[start:start+n] = Y_n
        start += n
    return Y

def initiate_model(method='NNLS', la=0):
    alpha = np.exp(la)
    conn_model = getattr(model, method)(alpha)
    return conn_model

def get_num_tasks():
    N = []
    for ds in gl.datasets:
        X = load_X_data(ds)
        N.append(X.shape[0])

    return N

def save_model(conn_model, name):
    save_path = os.path.join(gl.conn_dir, 'MNISymC3', 'train', 'sim')
    model_info = {"subj_id": [],
                "mname": [],
                "R_train": [],
                "R2_train": [],
                "num_regions": [],
                "logalpha": []
                }
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cio.save_model(conn_model, model_info, save_path + f"/{name}")


def get_fdata(dataset, sessions):
    subj = rm.get_subj_list('all', dataset)
    _, info, _ = fdata.get_dataset(gl.base_dir,
                                   dataset,
                                   sess=sessions,
                                   subj=subj,
                                   atlas='fs32k',
                                   type='CondAll')
    return info

if __name__ == "__main__":
    X = load_X_data(gl.traindata_string())
    X = rm.std_data(X, 'parcel')

    N = get_num_tasks()
    Y = generate_Y_data(N, 5445)

    conn_model = initiate_model('NNLS', 0)
    conn_model.fit(X, Y)
    save_model(conn_model, 'nnls_sim_parcel')
    
