import numpy as np
import pandas as pd
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.evaluation as ev
import cortico_cereb_connectivity.run_model as rm
from cortico_cereb_connectivity.bayes_simulation.simulation_functions import *


def create_dataset(data_mean, data_var, data_ind_var, shape, same_half=False):
    """ Creates data based on the mean and variance provided. Then makes realizitions for each subject.
    Args:
        data_mean (scalar): mean of the true dataset distribution
        data_var (scalar): variance of the true dataset distribution
        data_ind_var (scalar): variance of the distribution of the realizations around the true dataset
        shape ((S, A, B)): shape of the dataset
    Returns:
        data_subjects (ndarray (SxAxB)): realizations of the true dataset
        dataset_true (ndarray (AxB)): true dataset generated
    """

    S, A, B = shape
    dataset_true = np.random.normal(data_mean, np.sqrt(data_var), (A, B))
    data_subjects = create_subject_realization(dataset_true, data_ind_var, shape)
    if same_half:
        dataset_true = np.concatenate((dataset_true, dataset_true), axis=0)
        data_subjects = np.concatenate((data_subjects, data_subjects), axis=1)
    return data_subjects, dataset_true


def create_subject_realization(dataset_true, data_ind_var, shape):
    S, A, B = shape
    data_subjects = np.empty(shape)
    for s in range(S):
        data_subjects[s, :, :] = np.random.normal(dataset_true, np.sqrt(data_ind_var), (A, B))
    return data_subjects


def normalize_dataset(dataset, std_method='global'):
    """ This function will normalize the data
    Args:
        dataset (ndarray (AxB) or (SxAxB)): dataset generated
        std_method (string): how to standardize data
    Returns:
        dataset_normalized (ndarray (AxB) or (SxAxB)): normalized dataset generated
    """
    
    if dataset.ndim == 2:
        dataset = rm.std_data(dataset, std_method)
    elif dataset.ndim == 3:
        S = dataset.shape[0]
        for s in range(S):
            X = dataset[s, :, :]
            dataset[s, :, :] = rm.std_data(X, std_method)
    return dataset


def generate_sigma2eps(shape_params, scale_params, shape):
    """ Generates sigma2_eps (variance of the measurement noise) based on the shape and scale parameters of gamma distribution.
    Args:
        shape_params (ndarray (S)): shape parameter of gamma distribution
        scale_params (ndarray (S)): scale parameter of gamma distribution
    Returns:
        sigma2_epss (ndarray (SxP)): sigma2_eps (1 scalar for each voxel) for each subject
    """

    S, P = shape
    sigma2_epss = np.empty(shape)
    for s in range(S):
        sigma2_epss[s, :] = np.random.gamma(shape=shape_params[s], scale=scale_params[s], size=P)
    return sigma2_epss


def generate_Y(X_subjects, W_subjects, sigma2_epss, shape):
    """ Generates Y* based on the X and W. Also generates a measurement of Y* (Y_i).
    Args:
        X_subjects (ndarray (SxNxQ)): cortical data for subjects
        W_subjects (ndarray (SxQxP)): true connectivity weights for subjects
        sigma2_epss (ndarray (SxP)): sigma2_eps (1 scalar for each voxel) for each subject
    Returns:
        Y_subjects (ndarray (SxNxP)): measurement (Y_i = Y* + eps) for subjects
        Y_star_subjects (ndarray (SxNxP)): true cerebellar activity for each subject (Y*)
    """

    S, N, P = shape
    Y_star_subjects = np.zeros((S, N, P))
    Y_subjects = np.zeros((S, N, P))
    for s in range(S):
        X = X_subjects[s, :, :]
        W = W_subjects[s, :, :]
        sigma2_eps = sigma2_epss[s, :]
        eps = np.zeros((N, P))
        for v in range(P):
            eps[:, v] = np.random.normal(0, np.sqrt(sigma2_eps[v]), size=N)

        Y_star = X @ W
        Y_star_subjects[s, :, :] = Y_star
        Y_subjects[s, :, :] = Y_star + eps
    return Y_subjects, Y_star_subjects


def estimate_sigma2eps(Y):
    N = Y.shape[0]
    Y_1 = Y[:N//2, :]
    Y_2 = Y[N//2:, :]
    Y_mean = np.nanmean([Y_1, Y_2], axis=0)
    sigma2_eps = np.zeros(Y_mean.shape[1])
    sigma2_eps += np.diag((Y_1-Y_mean).T @ (Y_1-Y_mean)) / (Y_mean.shape[0])
    sigma2_eps += np.diag((Y_2-Y_mean).T @ (Y_2-Y_mean)) / (Y_mean.shape[0])
    return sigma2_eps


def estimate_W(X_subjects, Y_subjects, alpha, shape, sigma2_epss=None):
    """ Estimates W from the observed X and Y with Ridge regression.
    Args:
        X_subjects (ndarray (SxNxQ)): cortical data for subjects
        Y_subjects (ndarray (SxNxP)): measurement (Y_i = Y* + eps) for subjects
        alpha (scalar): alpha in Ridge regression
        sigma2_epss (ndarray (SxP)): sigma2_eps (1 scalar for each voxel) for each subject
    Returns:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
        Var_W_hat_subjects (ndarray (SxP)): estimated variance (uncertainity) of weights for subjects
        true_Var_W_hat_subjects (ndarray (SxP)): true variance (uncertainity) of weights for subjects
    """

    S, Q, P = shape
    W_hat_subjects = np.zeros((S, Q, P))
    Var_W_hat_subjects = np.zeros((S, P))
    true_Var_W_hat_subjects = np.zeros((S, P))
    for s in range(S):
        X = X_subjects[s, :, :] #- np.nanmean(X_subjects[s, :, :], axis=0)
        Y = Y_subjects[s, :, :] #- np.nanmean(Y_subjects[s, :, :], axis=0)
        conn_model = getattr(model, 'L2reg')(alpha)
        conn_model.fit(X, Y, dataframe='half')
        A = np.linalg.inv(X.T @ X + alpha * np.eye(Q)) @ X.T
        W_hat_subjects[s, :, :] = conn_model.coef_.T
        Var_W_hat_subjects[s, :] = conn_model.coef_var

        if sigma2_epss is not None:
            sigma2_eps = sigma2_epss[s, :]
            true_Var_W_hat_subjects[s, :] = sigma2_eps * np.nansum(A**2)
    if sigma2_epss is None:
        return W_hat_subjects, Var_W_hat_subjects
    else:
        return W_hat_subjects, Var_W_hat_subjects, true_Var_W_hat_subjects
    

def decompose_variance(data):
    """ Decomposes variance of group, subject, and measurement noise. This is an upgraded version to handle subject-specific scaling.
    Args:
        data (ndarray (n_sub, n_rep, n_A, n_B)): the data to decompose, at least 2 for each dimension
    Returns:
        vg (ndarray (n_sub,)): group variance scaled for each subject
        vs (ndarray (n_sub,)): subject variance scaled for each subject
        vm (ndarray (n_sun,)): measurement noise variance scaled for each subject
    """

    n_sub, n_rep, n_A, n_B = data.shape
    n_features = n_A * n_B
    data = data.reshape((n_sub, n_rep, n_features))    # Shape: (n_sub, n_rep, n_features)

    product_matrices = np.einsum('srf,tkf->stkr', data, data) / n_features  # Shape: (n_sub, n_sub, n_rep, n_rep)

    # Masks
    mask_self_sub = np.eye(n_sub, dtype=bool)[:, :, None, None] # Shape: (n_sub, n_sub, 1, 1)
    mask_self_rep = np.eye(n_rep, dtype=bool)[None, None, :, :] # Shape: (1, 1, n_rep, n_rep)
    
    # Cross-subject (type 1)
    # Remove self-pairs by masking
    type_1 = np.where(mask_self_sub, 0, product_matrices)   # Set self-pairs to 0
    # Mean over repetitions
    SS_1 = np.nansum(type_1, axis=(2, 3)) / (n_rep**2)  # Shape: (n_sub, n_sub)

    # Within-subject, diff reps (type 2)
    # Remove other-pairs and self-reps by masking
    type_2 = np.where(mask_self_sub, product_matrices, 0)   # Set other-pairs to 0
    type_2 = np.where(mask_self_rep, 0, type_2) # Set self-reps to 0
    # Mean over repetitions
    SS_2 = np.diagonal(np.nansum(type_2, axis=(2,3)) / (n_rep**2-n_rep), axis1=0, axis2=1)    # Shape: (n_sub)

    # Within-subject, same reps (type 3)
    type_3 = np.where(mask_self_sub, product_matrices, 0)   # Set other-pairs to 0
    type_3 = np.where(mask_self_rep, type_3, 0) # Set other-reps to 0
    # Mean over repetitions
    SS_3 = np.diagonal(np.nansum(type_3, axis=(2,3)) / (n_rep), axis1=0, axis2=1)   # Shape: (n_sub)

    vm = SS_3 - SS_2
    vg = np.nansum(np.sqrt(SS_2[:, None] / SS_2) * SS_1, axis=1) / (n_sub-1)    # Shape: (n_sub)
    vs = SS_2 - vg

    return vg, vs, vm


def calc_SNR(data, signal):
    S, _, _ = data.shape
    SNR_db = np.zeros((S,))
    for s in range(S):
        signal_power = np.sum(signal[s]**2)
        noise_power = np.sum((data[s]-signal[s])**2)
        SNR_db[s] = 10 * np.log10(signal_power / noise_power)
    return SNR_db


def weighted_avg(data, weights):
    weights /= np.sum(weights)
    if data.ndim == 3:
        avg_model = np.nansum(data * weights[:, np.newaxis, np.newaxis], axis=0)
    elif data.ndim == 2:
        avg_model = np.nansum(data * weights[:, np.newaxis], axis=0)
    return avg_model


def calc_model(model, W_hat_subjects, sigma2_w_hat, sigma2_s_hat):
    """ Combines the connectivity weights of subjects in a leave-one-out manner using the model specified
    Args:
        model (string):
            'loo': is simple average of leave-one-out
            'bayes': is one factor weighted average
            'bayes vox': is same as above, but for each voxel
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
        Var_W_hat_subjects (ndarray (SxP)): estimated variance (uncertainity) of weights for subjects
    Returns:
        W_group_model (ndarray (SxQxP)): model prediction of group connectiviy weights
    """

    S, Q, P = W_hat_subjects.shape
    Var_W_hat_subjects = (sigma2_w_hat + sigma2_s_hat)
    if model == 'loo':
        W_group_model = calc_model_loo(W_hat_subjects)
    elif model == 'bayes':
        W_group_model = calc_model_bayes(W_hat_subjects, np.nanmean(sigma2_w_hat, axis=1))
    elif model == 'bayes vox':
        W_group_model = calc_model_bayes_vox(W_hat_subjects, sigma2_w_hat)
    elif model == 'bayes new':
        Var_W_hat_subjects = np.nanmean(Var_W_hat_subjects, axis=1)
        signal_norm2_hat = np.linalg.norm(W_hat_subjects, axis=(1,2))**2 - Q*P*(Var_W_hat_subjects)
        signal_norm2_hat = np.maximum(1e-40, signal_norm2_hat)
        print(signal_norm2_hat)
        Var_W_hat_subjects = Var_W_hat_subjects / signal_norm2_hat
        W_hat_subjects_normalized = W_hat_subjects / np.sqrt(signal_norm2_hat)[:, np.newaxis, np.newaxis]
        W_group_model = calc_model_bayes(W_hat_subjects_normalized, Var_W_hat_subjects)
    elif model == 'bayes vox new':
        signal_norm2_hat = np.linalg.norm(W_hat_subjects, axis=1)**2 - Q*P*(Var_W_hat_subjects)
        Var_W_hat_subjects = Var_W_hat_subjects / signal_norm2_hat
        W_hat_subjects_normalized = W_hat_subjects / np.sqrt(signal_norm2_hat)[:, np.newaxis, :]
        W_group_model = calc_model_bayes_vox(W_hat_subjects_normalized, Var_W_hat_subjects)
    return W_group_model


def calc_model_loo(W_hat_subjects):
    """ Averages the connectivity weights of subjects in a leave-one-out manner.
    Args:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
    Returns:
        W_hat_loo (ndarray (SxQxP)): average of all other subject connectivity weights
    """

    S = W_hat_subjects.shape[0]
    W_hat_loo = np.empty_like(W_hat_subjects)
    for s in range(S):
        W_loo = np.delete(W_hat_subjects, s, axis=0)
        W_hat_loo[s, :, :] = np.nanmean(W_loo, axis=0)
    return W_hat_loo


def calc_model_bayes(W_hat_subjects, Var_W_hat_subjects):
    """ Combines the connectivity weights of subjects in a leave-one-out manner using Bayes integration.
    Args:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
        Var_W_hat_subjects (ndarray (S,)): estimated variance (uncertainity) of weights for subjects
    Returns:
        W_hat_opt (ndarray (SxQxP)): optimally combined connectivity weights
    """

    S = W_hat_subjects.shape[0]
    W_hat_opt = np.empty_like(W_hat_subjects)
    for s in range(S):
        W_hat_loo = np.delete(W_hat_subjects, s, axis=0)
        var_loo = np.delete(Var_W_hat_subjects, s, axis=0)

        weights = (1 / var_loo) / np.nansum(1 / var_loo, axis=0)
        W_hat_opt[s, :, :] = np.nansum(W_hat_loo * weights[:, np.newaxis, np.newaxis], axis=0)
    return W_hat_opt


def calc_model_bayes_vox(W_hat_subjects, Var_W_hat_subjects):
    """ Combines the connectivity weights of subjects in a leave-one-out manner using Bayes integration for each voxel.
    Args:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
        Var_W_hat_subjects (ndarray (SxP)): estimated variance (uncertainity) of weights for subjects
    Returns:
        W_hat_opt_vox (ndarray (SxQxP)): optimally combined connectivity weights voxel-wise
    """

    S = W_hat_subjects.shape[0]
    W_hat_opt_vox = np.empty_like(W_hat_subjects)
    for s in range(S):
        W_hat_loo = np.delete(W_hat_subjects, s, axis=0)
        var_loo = np.delete(Var_W_hat_subjects, s, axis=0)

        weights = (1 / var_loo) / np.nansum(1 / var_loo, axis=0)
        W_hat_opt_vox[s, :, :] = np.nansum(W_hat_loo * weights[:, np.newaxis, :], axis=0)
    return W_hat_opt_vox


def predict_Y_hat(X_subjects, W_subjects):
    """ Predicts the cerebellar activity using the cortical activity and group connectivity weights
    Args:
        X_subjects (ndarray (SxNxQ)): cortical data for subjects
        W_subjects (ndarray (SxQxP)): group connectivity weights for subjects
    Returns:
        Y_hat_group_model (ndarray (SxNxP)): predicted cerebellar activity
    """

    S, N, _ = X_subjects.shape
    _, _, P = W_subjects.shape
    conn_model = getattr(model, 'L2regression')()
    Y_hat_group_model = np.empty((S, N, P))
    for s in range(S):
        X = X_subjects[s, :, :]
        W = W_subjects[s, :, :]
        setattr(conn_model, 'coef_', W.T)
        Y_hat_group_model[s, :, :] = conn_model.predict(X)
    return Y_hat_group_model


def make_dataframe(method_dic, method_name, alpha, S, nc=None):
    """ Creates a dataframe of methods and their evaluation metrics for visualization.
    Args:
        method_dic (dictionary): keys are evaluation metrics ('RMSE'/'R') and values are evaluation values
        method_name (strings): name of the method ('loo'/'bayes'/'bayes vox')
        alpha (double): alpha of Ridge regression
    Returns:
        the dataframe
    """

    subject_list = [f'sub {i}' for i in range(S)]
    n_sub = len(subject_list)
    columns = ['subject', 'method', 'alpha', 'noise_ceiling'] + list(method_dic.keys())
    data = {col: [] for col in columns}
    data['subject'].extend(subject_list)
    data['method'].extend([method_name] * n_sub)
    data['alpha'].extend([alpha] * n_sub)
    if nc is None:
        data['noise_ceiling'].extend([1.0] * n_sub)
    else:
        data['noise_ceiling'].extend(nc)
    for eval_metric, value in method_dic.items():
        data[eval_metric].extend(value)
    return pd.DataFrame(data)


def evaluate_model_with_params(method, eval_metric, eval_param, eval_with, cv, eval_data, W_group_model, Y_hat_group_model):
    """ Calls the eval_model() with appropriate inputs
    """
    if eval_param == 'W':
        if eval_with == 'group':
            performance = eval_model(metric=eval_metric,
                                     data=eval_data['W_group'],
                                     data_pred=W_group_model[method])
        elif eval_with == 'ind':
            performance = eval_model(metric=eval_metric,
                                     data=eval_data['W_i'],
                                     data_pred=W_group_model[method])
    elif eval_param == 'Y':
        if eval_with == 'group':
            performance = eval_model(metric=eval_metric,
                                     data=eval_data['Y_group'],
                                     data_pred=Y_hat_group_model[method])
        elif eval_with == 'star':
            performance = eval_model(metric=eval_metric,
                                     data=eval_data['Y_star'],
                                     data_pred=Y_hat_group_model[method])
        elif eval_with == 'ind':
            if cv:
                performance = eval_model(metric=eval_metric,
                                         data=eval_data['Y_2'],
                                         data_pred=Y_hat_group_model[method])
            else:
                performance = eval_model(metric=eval_metric,
                                         data=eval_data['Y_1'],
                                         data_pred=Y_hat_group_model[method])
    return performance


def eval_model(data, data_pred, metric='R'):
    """ Calculates the performance of models based on the evaluation metric
    Args:
        data (ndarray (QxP)/(SxAxB)): true data
        data_pred (ndarray (SxAxB)): estimated data
    Returns:
        performance (list of scalars): root mean squared error / correlation coef
    """

    S = data_pred.shape[0]
    performance = []
    if metric == 'RMSE':
        for s in range(S):
            if data.ndim == 3:
                data_s = data[s, :, :]
            else:
                data_s = data
            performance.append(calc_rmse(data_s, data_pred[s, :, :]))
    elif metric == 'R':
        for s in range(S):
            if data.ndim == 3:
                data_s = data[s, :, :]
            else:
                data_s = data
            R, _ = ev.calculate_R(data_s, data_pred[s, :, :])
            performance.append(R)
    elif metric == 'R2':
        for s in range(S):
            if data.ndim == 3:
                data_s = data[s, :, :]
            else:
                data_s = data
            R2, _ = ev.calculate_R2(data_s, data_pred[s, :, :])
            performance.append(R2)
    else:
        raise ValueError("Undefined metric. Should be either 'RMSE' or 'R' or 'R2'.")
    return performance


def calc_noise_ceiling(X_subjects, W_model, Y1_subjects, Y2_subjects):
    """ Is not complete yet
    """
    nc = []
    for s in range(S):
        R, _ = ev.calculate_R(Y1_subjects[s,:,:], Y2_subjects[s,:,:])
        nc.append(R)
    return nc


def calc_rmse(data, data_pred):
    """ Calculates the root mean squared error between two matrices.
    Args:
        data (ndarray (AxB)): true weights generated / true cerebellar activity
        data_pred (ndarray (SxAxB)/(SxAxB)): estimated combined weights / estimated cerebellar activity
    Returns:
        rsme_error (scalar): root mean squared error
    """
    rsme_error = np.sqrt(np.nanmean((data - data_pred)**2))
    return rsme_error