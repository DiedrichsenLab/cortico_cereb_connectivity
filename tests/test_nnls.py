import numpy as np 
import scipy.optimize as so
import matplotlib.pyplot as plt

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


def test_nnls():
    """
    Test nnls function
    """
    N = 100
    Q = 10
    P = 40
    X, W, Y = generate_data(N,Q,P)



    plt.subplot(2,1,1)
    plt.imshow(W)
    plt.subplot(2,1,2)
    plt.imshow(W_est)
    plt.show()

def scipy_nnls(X,Y):
    """
    Use scipy nnls function
    """
    [N,Q]=X.shape
    [N1,P]=Y.shape
    W_est = np.zeros((Q,P))

    for i in range(P):
        W_est[:,i] = so.nnls(X,Y[:,i])[0]
    return W_est    

if __name__ == "__main__":
    test_nnls()