from cortico_cereb_connectivity.model import Model
import numpy as np
import scipy.optimize as so
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import time


class old_nnls(Model):

    def __init__(self, alpha=0, n_jobs=-1):
        self.alpha = alpha
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        Q = X.shape[1]
        P = Y.shape[1]
        self.coef_ = np.zeros((P,Q))

        def solve_nnls_single(X, y, alpha):
            Q = X.shape[1]
            if alpha > 0:
                A = np.vstack((X, np.sqrt(alpha) * np.eye(Q)))
                b = np.concatenate([y, np.zeros(Q)])
            else:
                A = X
                b = y
            return so.nnls(A, b)[0]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(solve_nnls_single)(X, Y[:, i], self.alpha) for i in range(P)
        )
        self.coef_ = np.array(results)
        return self


class new_nnls(Model):

    def __init__(self, alpha=0, n_jobs=-1):
        self.alpha = alpha
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        Q = X.shape[1]
        P = Y.shape[1]
        if self.alpha > 0:
            A = np.vstack((X, np.sqrt(self.alpha) * np.eye(Q)))
            zero = np.zeros(Q)
            def solve_nnls_single(y):
                b = np.concatenate([y, zero])
                return so.nnls(A, b)[0]
        else:
            A = X
            def solve_nnls_single(y):
                b = y
                return so.nnls(A, b)[0]

        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(solve_nnls_single)(Y[:, i]) for i in range(P)
            )
        self.coef_ = np.array(results)
        return self
    

def generate_data(N,Q,P,sig_e=0.1):
    """
    Generate random data for testing nnls
    Args:
        N (int): Number of observations 
        Q (int): Number of cortical features 
        P (int): Number of cerebellar voxels 
    Returns:
        X (nd-array): N*Q Cortical activation matrix
        W (nd-array): Q * P True weight matrix (>0)
        Y (nd-array): N X P 
    """
    rng = np.random.default_rng(seed=None)
    X = rng.normal(0,1,(N,Q))
    W = rng.uniform(-3,1,(Q,P))
    W[W<0]= 0 # Make sparse
    Y = X @ W + rng.normal(0,sig_e,(N,P))
    return X, W, Y


def test_nnls_speed(alpha=0):
    """
    Test nnls function speed for different implementations
    """
    N = 110
    Q = 180
    P = 590
    X, W, Y = generate_data(N,Q,P)

    t1 = time.perf_counter()
    conn_model = old_nnls(alpha)
    W_est1 = conn_model.fit(X,Y).coef_
    t2 = time.perf_counter()
    print(f"Time taken by old nnls: {t2-t1}")

    t1 = time.perf_counter()
    conn_model = new_nnls(alpha)
    W_est2 = conn_model.fit(X,Y).coef_
    t2 = time.perf_counter()
    print(f"Time taken by new nnls: {t2-t1}")

    print(f"Difference between old and new nnls: {np.linalg.norm(W_est1-W_est2)}")
    pass
    

if __name__ == "__main__":
    test_nnls_speed(alpha=1)