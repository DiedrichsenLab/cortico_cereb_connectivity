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
        # Xs = X / self.scale_
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

        Y_list = []
        for i in np.unique(dataframe["half"]):
            Y_list.append(Y[dataframe["half"] == i, :])

        Y_mean = np.nanmean(Y_list, axis=0)

        sigma2eps = np.zeros(Y_mean.shape[1])
        for i in np.unique(dataframe["half"]):
            Y_i = Y[dataframe["half"] == i, :]
            sigma2eps += np.diag((Y_i-Y_mean).T @ (Y_i-Y_mean)) / (Y_mean.shape[0])
        self.sigma2eps = sigma2eps
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

class L2reg(Model):
    """
    L2 regularized connectivity model
    simple wrapper for Ridge. It performs scaling by stdev, but not by mean before fitting and prediction
    """

    def __init__(self, alpha=1):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        self.alpha = alpha
        self.fit_intercept=False

    def fit(self, X, Y):
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        # Compute Psuedu-inverse using solve

        self.coef_ = A @ Y 
        self.var_coef = ...
        return self

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T # weights need to be transposed (throws error otherwise)



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
    It performs scaling by stdev, but not by mean before fitting and prediction
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
        self.scale_ = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
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
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
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
