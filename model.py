from operator import index
import numpy as np
import pandas as pd
from scipy import sparse
import scipy.optimize as so
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import cortico_cereb_connectivity.evaluation as ev
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.run_model as rm
import warnings
import nibabel as nb
"""
connectivity models
A connectivity model is inherited from the sklearn class BaseEstimator
such that Ridge, Lasso, ElasticNet and other models can
be easily used.

@authors: Jörn Diedrichsen, Maedbh King, Ladan Shahshahani,
"""



class Model:
    def __init__(self, name = None):
        self.name = name

    def fit(self, X, Y):
        """ Fitting function needs to be implemented for each model.
        Needs to return self.
        """
        return self

    def predict(self, X):
        Xs = np.nan_to_num(X) # there are Nan values
        return Xs @ self.coef_.T

    def to_dict(self):
        data = {"coef_": self.coef_}
        return data

    def to_cifti(self,
                 src_atlas,
                 trg_atlas,
                 src_roi = None,
                 trg_roi = None,
                 fname = None,
                 dtype = 'float32'):
        """ Convert the weights to a cifti conn-image. """

        # Integrate the scaling factor (if present) to the weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=RuntimeWarning)
            weights = self.coef_/self.scale_

        # Convert the weights to a cifti image
        cifti_img = cio.model_to_cifti(weights.astype(dtype),
                                   src_atlas,
                                   trg_atlas,
                                   src_roi,
                                   trg_roi,
                                   type = 'conn')

        if fname is not None:
            nb.save(cifti_img,fname)
        return cifti_img

    def from_cifti(self, fname = None):
        """ Load the model weights from a cifti conn-image.

        Args:
            fname (str) - filename of the cifti image
        Returns:
            self (Model) - the model with the loaded weights
        """
        C = nb.load(fname)
        self.coef_ = C.get_fdata()
        self.scale_ = np.ones(self.coef_.shape[1])
        return self

class L2regression(Ridge, Model):
    """
    L2 regularized connectivity model
    simple wrapper for Ridge. It performs scaling by stdev, but not by mean before fitting and prediction
    """

    def __init__(self, alpha=1):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__(alpha=alpha, fit_intercept=False)

    def fit(self, X, Y):
        self.scale_ = np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return super().fit(Xs, Y)

    def predict(self, X):
        Xs = X / self.scale_
        Xs = X
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T # weights need to be transposed (throws error otherwise)
    
class L2reg(Model):
    """
    New model for L2regression. This model assumes the data is already normalized.
    The attribute 'sigma2eps' is an estimation of variance of the measurement noise on Y for each voxel.
    The attribute 'coef_var' is an estimation of uncertainty of connectivity weights that can be used in Bayes optimal integration.
    """

    def __init__(self, alpha=1):
        self.alpha = alpha

    def estimate_sigma2eps(self, Y, dataframe=None):
        if dataframe is None:
            self.sigma2eps =  np.ones(Y.shape[1])
            return self.sigma2eps
        elif isinstance(dataframe, str) and dataframe == 'half':
            dataframe = pd.DataFrame([1]*(Y.shape[0]//2) + [2]*(Y.shape[0]-Y.shape[0]//2), columns=["half"])

        if len(dataframe[dataframe['half']==1]) != len(dataframe[dataframe['half']==2]):
            print('sigma2_eps estimation cannot be done. Data has inconsistent length of halves.')
            self.sigma2eps =  np.ones(Y.shape[1])
            return self.sigma2eps

        Y_list = []
        for half in np.unique(dataframe["half"]):
            Y_list.append(Y[dataframe["half"] == half, :])

        Y_mean = np.nanmean(Y_list, axis=0)

        sigma2eps = np.zeros(Y_mean.shape[1])
        for i, half in enumerate(np.unique(dataframe["half"])):
            sigma2eps += np.nansum((Y_list[i]-Y_mean)**2, axis=0) / (Y_mean.shape[0])
        self.sigma2eps = sigma2eps / i
        return self.sigma2eps

    def fit(self, X, Y, dataframe=None):
        Xs = np.nan_to_num(X)
        Xs_T = Xs.T
        self.estimate_sigma2eps(Y, dataframe)
        pseudoinverse = np.linalg.inv(Xs_T @ Xs + self.alpha * np.identity(Xs.shape[1])) @ Xs_T
        self.coef_ = (pseudoinverse @ Y).T
        self.coef_var = self.sigma2eps * np.nansum(pseudoinverse**2)
        return self.coef_.T
    
    def predict(self, X):
        Xs = np.nan_to_num(X)
        return Xs @ self.coef_.T
    

class L2reg2(Model):
    """
    Model for L2 (Ridge) regression, assuming normalized data.
    Estimates measurement noise variance for Y ('sigma2eps_Y') and X ('sigma2eps_X') per feature.
    Computes uncertainty of connectivity weights ('coef_var') accounting for noise in both X and Y.
    """

    def __init__(self, alpha=1):
        self.alpha = alpha

    def estimate_sigma2eps(self, data, dataframe=None):
        """
        Estimate noise variance for each feature in the input data (X or Y) using two halves.
        Returns a vector of variances, one per feature (column).
        """
        if dataframe is None:
            return np.ones(data.shape[1])
        elif isinstance(dataframe, str) and dataframe == 'half':
            dataframe = pd.DataFrame([1]*(data.shape[0]//2) + [2]*(data.shape[0]-data.shape[0]//2), columns=["half"])

        if len(dataframe[dataframe['half']==1]) != len(dataframe[dataframe['half']==2]):
            print('sigma2eps estimation cannot be done. Data has inconsistent length of halves.')
            return np.ones(data.shape[1])

        data_list = []
        for half in np.unique(dataframe["half"]):
            data_list.append(data[dataframe["half"] == half, :])

        data_mean = np.nanmean(data_list, axis=0)
        sigma2eps = np.zeros(data_mean.shape[1])
        for i, half in enumerate(np.unique(dataframe["half"])):
            sigma2eps += np.nansum((data_list[i] - data_mean)**2, axis=0) / (data_mean.shape[0])
        return sigma2eps / i

    def fit(self, X, Y, dataframe=None):
        """
        Fit the Ridge regression model and compute coefficient variance.
        Accounts for noise in both X and Y.
        """
        Xs = np.nan_to_num(X)
        Xs_T = Xs.T
        # Estimate noise variances (vectors of variances per feature)
        self.sigma2eps_Y = self.estimate_sigma2eps(Y, dataframe)  # Shape: (n_voxels,)
        self.sigma2eps_X = self.estimate_sigma2eps(Xs, dataframe)  # Shape: (n_regions,)
        
        # Compute Ridge regression coefficients
        A_inv = np.linalg.inv(Xs_T @ Xs + self.alpha * np.identity(Xs.shape[1]))
        pseudoinverse = A_inv @ Xs_T
        self.coef_ = (pseudoinverse @ Y).T  # Shape: (n_voxels, n_regions)
        
        # Compute coefficient variance for each voxel
        # Term 1: Y noise contribution
        y_noise_var = np.trace(pseudoinverse @ pseudoinverse.T)  # Scalar, trace of pseudoinverse product
        
        # Term 2: X noise contribution
        # Compute W_j^T Sigma_X W_j = sum(W_j,i^2 * sigma2_X,i) for all voxels
        x_noise_var = np.sum(self.coef_**2 * self.sigma2eps_X, axis=1)  # Shape: (n_voxels,)
        term2 = np.trace(A_inv)  # Scalar, trace of A_inv
        
        # Total variance per voxel
        self.coef_var = self.sigma2eps_Y * y_noise_var + x_noise_var * term2
        
        return self.coef_.T

    def predict(self, X):
        """
        Predict Y using the fitted model.
        """
        Xs = np.nan_to_num(X)
        return Xs @ self.coef_.T
    

class L2reghalf(Model):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, Y, config, info):
        Xs = np.nan_to_num(X)

        if len(info['half']==1) != len(info['half']==2):
            print('Splitting data cannot be done. Data has inconsistent length of halves. Trimming to the smallest length...')
            min_len = min(len(info['half']==1), len(info['half']==2))
        else:
            min_len = len(info['half']==1)
        Xs_1 = Xs[info['half']==1, :]
        Xs_1 = Xs_1[:min_len, :]
        Xs_2 = Xs[info['half']==2, :]
        Xs_2 = Xs_2[:min_len, :]

        Y_1 = Y[info['half']==1, :] # if config['crossed']=='half', then info['half']==1 is Y of half 2
        Y_1 = Y_1[:min_len, :]
        Y_2 = Y[info['half']==2, :]
        Y_2 = Y_2[:min_len, :]

        # Definitely subtract intercept across all conditions
        Xs_1 = (Xs_1 - Xs_1.mean(axis=0))
        Xs_2 = (Xs_2 - Xs_2.mean(axis=0))
        Y_1 = (Y_1 - Y_1.mean(axis=0))
        Y_2 = (Y_2 - Y_2.mean(axis=0))

        if 'std_cortex' in config.keys():
            Xs_1 = rm.std_data(Xs_1,config['std_cortex'])
            Xs_2 = rm.std_data(Xs_2,config['std_cortex'])
        if 'std_cerebellum' in config.keys():
            Y_1 = rm.std_data(Y_1,config['std_cerebellum'])
            Y_2 = rm.std_data(Y_2,config['std_cerebellum'])

        Xs_1_T = Xs_1.T
        Xs_2_T = Xs_2.T

        self.coef_1 = np.linalg.solve(Xs_1_T @ Xs_1 + self.alpha * np.identity(Xs_1.shape[1]), Xs_1_T @ Y_1).T
        self.coef_2 = np.linalg.solve(Xs_2_T @ Xs_2 + self.alpha * np.identity(Xs_2.shape[1]), Xs_2_T @ Y_2).T
        self.coef_ = (self.coef_1 + self.coef_2) / 2
        return self.coef_1, self.coef_2
    
    def predict(self, X):
        Xs = np.nan_to_num(X)
        return Xs @ self.coef_.T


class L1regression(Lasso, Model):
    """
    L2 regularized connectivity model
    simple wrapper for Ridge. It performs scaling by stdev, but not by mean before fitting and prediction
    """

    def __init__(self, alpha=1):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__(alpha=alpha, fit_intercept=False)

    def fit(self, X, Y):
        self.scale_ = np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return super().fit(Xs, Y)

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class WTA(BaseEstimator, Model):
    """
    WTA model
    Assumes data is already properly scaled. 
    """

    def __init__(self):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__()

    def fit(self, X, Y):
        """ Coefficients are the Y'T X between cortical and cerebellar data
        self.scale_: standard deviation of cortical data
        self.coef_: regression coefficient between cortical and cerebellar data for best cortical parcel
        self.labels: 1-based index for the best cortical parcel
        """
        scale = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        Xs = X / scale # Scaling here is only done for comparision purposes... 
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        self.coef_ = Y.T @ Xs  # This is the correlation (non-standardized)
        self.labels = np.argmax(self.coef_, axis=1)
        wta_coef_ = np.amax(self.coef_, axis=1)
        self.coef_ = np.zeros((self.coef_.shape))
        num_vox = self.coef_.shape[0]
        self.coef_[np.arange(num_vox), self.labels] = wta_coef_
        self.labels = self.labels + 1 # we don't want zero-indexed label
        return self.coef_, self.labels

    def predict(self, X):
        Xs = np.nan_to_num(X) # there are 0 values after scaling
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class NNLS(Model):
    """
        NNLS model with L2 regularization - no internal scaling of the data.
        Xw = y  with ||y - Xw||^2 + alpha * ||w||^2 can be solved as
        A = [X; sqrt(alpha) * I] and b = [Y; 0]
    """

    def __init__(self,alpha=0):
        self.alpha = alpha

    def fit(self, X, Y):
        [N,Q]=X.shape
        [N1,P]=Y.shape
        self.coef_ = np.zeros((P,Q))
        # With L2 regularization - appen
        if self.alpha > 0:
            A = np.vstack((X,np.sqrt(self.alpha)*np.eye(Q)))
            for i in range(P):
                if (i % 100) == 0:
                    print('.')
                v= np.concatenate([Y[:,i],np.zeros(Q)])
                self.coef_[i,:] = so.nnls(A,v)[0]
        else:
            for i in range(P):
                if (i % 100) == 0:
                    print('.')
                self.coef_[i,:] = so.nnls(X,Y[:,i])[0]
        return self
