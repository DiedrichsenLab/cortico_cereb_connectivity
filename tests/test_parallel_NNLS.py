import numpy as np
import scipy.optimize as so
from joblib import Parallel, delayed
import Functional_Fusion.dataset as fdata
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as model
import time


def load_subject():
    config = rm.get_train_config(train_dataset = "MDTB",
                     train_ses = "all",
                     subj_list = 'sub-02',
                     cerebellum = "MNISymC3",
                     parcellation= "Icosahedron162",
                     validate_model = False,
                     )
    Y, info = rm.get_cerebellar_data('MDTB','all','sub-02',config)
    X, _ = rm.get_cortical_data('MDTB','all','sub-02',config)

    return X[0], Y[0], info


if __name__ == "__main__":
    alpha = np.exp(2)
    X, Y, info = load_subject()

    print(f"Data shapes: X = {X.shape}, Y = {Y.shape}")

    # Original L2reg
    conn_model_1 = getattr(model, 'L2regression')(alpha)
    print("\nFitting with L2regression...")
    t0 = time.time()
    conn_model_1.fit(X, Y)
    t1 = time.time()
    print(f"L2regression fit time: {t1 - t0:.2f} seconds")

    # Original NNLS
    conn_model_2 = getattr(model, 'NNLS')(alpha)
    print("\nFitting with original NNLS...")
    t0 = time.time()
    conn_model_2.fit(X, Y)
    t1 = time.time()
    print(f"Original NNLS fit time: {t1 - t0:.2f} seconds")

    # Parallel NNLS
    conn_model_3 = getattr(model, 'NNLS_parallel')(alpha)
    print("\nFitting with parallel NNLS...")
    t2 = time.time()
    conn_model_3.fit(X, Y)
    t3 = time.time()
    print(f"Parallel NNLS fit time: {t3 - t2:.2f} seconds")

    # Sanity check: compare outputs
    diff = np.linalg.norm(conn_model_2.coef_ - conn_model_3.coef_)
    print(f"\nL2 norm difference between models: {diff:.4f}")