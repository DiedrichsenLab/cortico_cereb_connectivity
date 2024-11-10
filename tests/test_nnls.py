import numpy as np
import scipy.optimize as so
from scipy.linalg import LinAlgWarning, solve
import matplotlib.pyplot as plt
import time
import fnnls
import sklearn.linear_model as slm
import warnings

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

def scipy_nnls(X,Y):
    """
    Use scipy nnls function directly
    """
    [N,Q]=X.shape
    [N1,P]=Y.shape
    W_est = np.zeros((Q,P))

    for i in range(P):
        W_est[:,i] = so.nnls(X,Y[:,i])[0]
    return W_est

def scipy_nnls_l2(X,Y,alpha=0.1):
    """
        Implements a L2-regularized version of nnls appending penality term
        Xw = y  with ||y - Xw||^2 + alpha * ||w||^2 can be solved as
        A = [X; sqrt(alpha) * I] and b = [Y; 0]
    """
    [N,Q]=X.shape
    [N1,P]=Y.shape
    W_est = np.zeros((Q,P))
    A = np.vstack((X,np.sqrt(alpha)*np.eye(Q)))
    B = np.vstack((Y,np.zeros((Q,P))))
    for i in range(P):
        W_est[:,i] = so.nnls(A,B[:,i])[0]
    return W_est


def fast_nnls(X,Y):
    """
        Fast nnls algorithm - does not seem to be fast than even the
        source code version of scipy nnls
    """
    [N,Q]=X.shape
    [N1,P]=Y.shape
    W_est = np.zeros((Q,P))

    for i in range(P):
        W_est[:,i] = fnnls.fnnls(X,Y[:,i])[0]
    return W_est

def sklearn_nnls(X,Y):
    from sklearn.linear_model import LinearRegression
    [N,Q]=X.shape
    [N1,P]=Y.shape
    W_est = np.zeros((Q,P))
    reg_nnls = LinearRegression(positive=True)
    for i in range(P):
        W_est[:,i] = reg_nnls.fit(X,Y[:,i]).coef_
    return W_est

def scipy_source_nnls(X,Y,maxiter=None,tol=None):
    """
    This is a wrapper for the source code of scipy nnls
    Slower than calling nnls directly - my guess is that scipy.nnls is compiled?
    """
    N,Q = X.shape
    P = Y.shape[1]

    AtA = X.T @ X
    Atb = X.T @ Y  # Result is 1D - let NumPy figure it out
    W_est = np.zeros((Q,P))

    if not maxiter:
        maxiter = 3*Q
    if tol is None:
        tol = 10 * max(N,Q) * np.spacing(1.)

    for i in range(P):
        W_est[:,i]= _nnls(AtA,Atb[:,i],maxiter,tol)[0]
        # Initialize vars
    return W_est

def _nnls(AtA,Atb,maxiter,tol):
    """ core NNLS implementation from Scipy (source code)"""
    n = AtA.shape[0]
    x = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    # Inactive constraint switches
    P = np.zeros(n, dtype=bool)

    # Projected residual
    w = Atb.copy().astype(np.float64)  # x=0. Skip (-AtA @ x) term

    # Overall iteration counter
    # Outer loop is not counted, inner iter is counted across outer spins
    iter = 0

    while (not P.all()) and (w[~P] > tol).any():  # B
        # Get the "most" active coeff index and move to inactive set
        k = np.argmax(w * (~P))  # B.2
        P[k] = True  # B.3

        # Iteration solution
        s[:] = 0.
        # B.4
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Ill-conditioned matrix',
                                    category=LinAlgWarning)
            s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym', check_finite=False)

        # Inner loop
        while (iter < maxiter) and (s[P].min() < 0):  # C.1
            iter += 1
            inds = P * (s < 0)
            alpha = (x[inds] / (x[inds] - s[inds])).min()  # C.2
            x *= (1 - alpha)
            x += alpha*s
            P[x <= tol] = False
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Ill-conditioned matrix',
                                        category=LinAlgWarning)
                s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym',
                            check_finite=False)
            s[~P] = 0  # C.6

        x[:] = s[:]
        w[:] = Atb - AtA @ x

        if iter == maxiter:
            # however at the top level, -1 raises an exception wasting norm
            # Instead return dummy number 0.
            return x, 0
    return x, 1

def test_nnls_speed():
    """
    Test nnls function speed for different implementations
    """
    N = 100
    Q = 40
    P = 100
    X, W, Y = generate_data(N,Q,P)

    t1 = time.perf_counter()
    W_est1 = fast_nnls(X,Y)
    t2 = time.perf_counter()
    print(f"Time taken by scipy nnls: {t2-t1}")

    t1 = time.perf_counter()
    W_est2 = scipy_source_nnls(X,Y)
    t2 = time.perf_counter()
    print(f"Time taken by scipy s nnls: {t2-t1}")

    plt.subplot(3,1,1)
    plt.imshow(W)
    plt.subplot(3,1,2)
    plt.imshow(W_est1)
    plt.subplot(3,1,3)
    plt.imshow(W_est2)
    pass

def test_nnls_reg():
    """
    Test L2 regression on NNL
    """
    N = 30
    Q = 50
    P = 100
    X, W, Y = generate_data(N,Q,P)

    t1 = time.perf_counter()
    W_est1 = scipy_nnls_l2(X,Y,alpha=0.1)
    t2 = time.perf_counter()
    print(f"Time taken by scipy nnls reg: {t2-t1}")

    t1 = time.perf_counter()
    W_est2 = scipy_nnls(X,Y)
    t2 = time.perf_counter()
    print(f"Time taken by scipy nnls: {t2-t1}")

    plt.subplot(3,1,1)
    plt.scatter(W.flatten(),W_est1.flatten())
    plt.subplot(3,1,2)
    plt.scatter(W.flatten(),W_est2.flatten())
    plt.subplot(3,1,3)
    plt.scatter(W_est2.flatten(),W_est1.flatten())
    pass



if __name__ == "__main__":
    test_nnls_reg()