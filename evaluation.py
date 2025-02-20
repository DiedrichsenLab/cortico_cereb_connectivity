import numpy as np
import warnings

"""Main module for evaluation metrics for connectivity models.

   @authors: Maedbh King, Ladan Shahshahani, Jörn Diedrichsen  
"""


def calculate_R(Y, Y_pred):
    """Calculates correlation between Y and Y_pred without subtracting the mean.

    Args:
        Y (nd-array):
        Y_pred (nd-array):
    Returns:
        R (scalar): Correlation between Y and Y_pred
        R_vox (1d-array): Correlation per voxel between Y and Y_pred
    """
    SYP = np.nansum(Y * Y_pred, axis=0)
    SPP = np.nansum(Y_pred * Y_pred, axis=0)
    SST = np.nansum(Y ** 2, axis=0) 

    R = np.nansum(SYP) / np.sqrt(np.nansum(SST) * np.nansum(SPP))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        R_vox = SYP / np.sqrt(SST * SPP)  # per voxel

    return R, R_vox


def calculate_R2(Y, Y_pred):
    """Calculates squared correlation between Y and Y_pred without subtracting the mean.

    Args:
        Y (nd-array):
        Y_pred (nd-array):
    Returns:
        R2 (scalar): Squared Correlation between Y and Y_pred
        R2_vox (1d-array): Squared Correlation per voxel between Y and Y_pred
    """
    res = Y - Y_pred

    SSR = np.nansum(res ** 2, axis=0)  
    SST = np.nansum(Y ** 2, axis=0)  

    R2 = 1 - (np.nansum(SSR) / np.nansum(SST))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        R2_vox = 1 - (SSR / SST)

    return R2, R2_vox

def calculate_reliability(Y, dataframe):
    """Calculates reliability of Y data across sessions.

    Data for session need to have same structure and length.
    Args:
        Y (nd-array)
        dataframe (pandas dataframe): dataframe with session info
    Returns:
        R (scalar): Correlation value
        R_vox (1d-array): Correlation value per voxel
        R2 (scalar): Squared correlation
        R2_vox (1d-array): Squared correlation per voxel
    """
    Y_flip = np.r_[Y[dataframe["half"] == 2, :], Y[dataframe["half"] == 1, :]]

    R, R_vox = calculate_R(Y, Y_flip)
    R2, R2_vox = calculate_R2(Y, Y_flip)
    return R, R_vox, R2, R2_vox
