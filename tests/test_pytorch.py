import torch as pt
import numpy as np
import scipy.optimize as so
from scipy.linalg import LinAlgWarning, solve
import time

from cortico_cereb_connectivity.tests.test_nnls import generate_data

def np_mult(X,Y,alpha=0.1):
    [N,P]=Y.shape
    [N,Q]=X.shape
    W_est = np.zeros((Q,P))
    for i in range(P):
        AtA = X.T @ X + alpha * np.eye(Q)
        Atb = X.T @ Y  # Result is 1D - let NumPy figure it out
        W_est[:,i]= AtA @ Atb[:,i] # solve(AtA,Atb[:,i],assume_a='sym', check_finite=False)
    return W_est

def pt_mult(X,Y,alpha=0.1,dtype=pt.float32,device='mps'):
    [N,P]=Y.shape
    [N,Q]=X.shape
    X = pt.tensor(X,dtype=dtype,device=device)
    Y = pt.tensor(Y,dtype=dtype,device=device)

    W_est = pt.zeros((Q,P),dtype=dtype,device=device)
    for i in range(P):
        AtA = X.T @ X + alpha * pt.eye(Q,dtype=dtype,device=device)
        Atb = X.T @ Y  # Result is 1D - let NumPy figure it out
        W_est[:,i]= AtA @ Atb[:,i] # solve(AtA,Atb[:,i],assume_a='sym', check_finite=False)
    return W_est.cpu().numpy()

def np_solve(X,Y,alpha=0.1):
    [N,P]=Y.shape
    [N,Q]=X.shape
    W_est = np.zeros((Q,P))
    AtA = X.T @ X + alpha * np.eye(Q)
    Atb = X.T @ Y  # Result is 1D - let NumPy figure it out
    for i in range(P):
        W_est[:,i]= solve(AtA,Atb[:,i],assume_a='sym', check_finite=False)
    return W_est

def pt_solve(X,Y,alpha=0.1,dtype=pt.float32,device='mps'):
    [N,P]=Y.shape
    [N,Q]=X.shape
    X = pt.tensor(X,dtype=dtype,device=device)
    Y = pt.tensor(Y,dtype=dtype,device=device)
    W_est = pt.zeros((Q,P),dtype=dtype,device=device)
    AtA = X.T @ X + alpha * pt.eye(Q,dtype=dtype,device=device)
    Atb = X.T @ Y  # Result is 1D - let NumPy figure it out
    for i in range(P):
        W_est[:,i]= pt.linalg.solve(AtA,Atb[:,i]) # solve(AtA,Atb[:,i],assume_a='sym', check_finite=False)
    return W_est.cpu().numpy()


def test_torch_mult():
    N = 200
    Q = 1800
    P = 800
    X, W, Y = generate_data(N,Q,P)

    alpha = 0.1

    t1 = time.perf_counter()
    a= np_mult(X,Y,alpha)
    t2 = time.perf_counter()
    print(f"Time taken by 1: {t2-t1}")

    t1 = time.perf_counter()
    b= pt_mult(X,Y,alpha,device='mps')
    t2 = time.perf_counter()
    print(f"Time taken by 2: {t2-t1}")
    pass

def test_torch_solve():
    N = 50
    Q = 1000
    P = 400
    X, W, Y = generate_data(N,Q,P)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    alpha = 0.1

    t1 = time.perf_counter()
    a= np_solve(X,Y,alpha)
    t2 = time.perf_counter()
    print(f"Time taken by 1: {t2-t1}")

    t1 = time.perf_counter()
    b= pt_solve(X,Y,alpha,device='mps')
    t2 = time.perf_counter()
    print(f"Time taken by 2: {t2-t1}")

    pass



if __name__ == "__main__":
    print(pt.get_default_dtype())
    #print(pt.get_default_device())
    # test_torch_mult()
    test_torch_solve()
    pass
