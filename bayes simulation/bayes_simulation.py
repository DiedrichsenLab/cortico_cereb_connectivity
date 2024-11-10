import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cortico_cereb_connectivity.model as model


# Set global parameters
N = 58  # Number of tasks
Q = 100  # Number of cortical regions
P = 400  # Number of cerebellar voxels
S = 24  # Number of subjects
n_simulation = 10 # Number of simulations


def create_dataset(data_mean, data_var, data_ind_var, shape):
    """ Creates data based on the mean and variance provided. Then makes realizitions for each subject.
    Args:
        data_mean (scalar): mean of the true dataset distribution
        data_var (scalar): variance of the true dataset distribution
        data_ind_var (scalar): variance of the distribution of the realizations around the true dataset
        shape ((A, B)): shape of the dataset
    Returns:
        data_subjects (ndarray (SxAxB)): realizations of the true dataset
        dataset_true (ndarray (AxB)): true dataset generated
    """

    dataset_true = np.random.normal(data_mean, np.sqrt(data_var), shape)
    data_subjects = np.empty((S, *shape))
    for s in range(S):
        data_subjects[s, :, :] = np.random.normal(dataset_true, np.sqrt(data_ind_var), shape)
    return data_subjects, dataset_true


def generate_sigma2eps(shape_params, scale_params):
    """ Generates sigma2_eps (variance of the measurement noise) based on the shape and scale parameters of gamma distribution.
    Args:
        shape_params (ndarray (S)): shape parameter of gamma distribution
        scale_params (ndarray (S)): scale parameter of gamma distribution
    Returns:
        sigma2_epss (ndarray (SxP)): sigma2_eps (1 scalar for each voxel) for each subject
    """

    sigma2_epss = np.empty((S, P))
    for s in range(S):
        sigma2_epss[s, :] = np.random.gamma(shape=shape_params[s], scale=scale_params[s], size=P)
    return sigma2_epss


def generate_Y(X_subjects, W_subjects, sigma2_epss):
    """ Generates Y* based on the X and W. Also generates a measurement of Y* (Y_i).
    Args:
        X_subjects (ndarray (SxNxQ)): cortical data for subjects
        W_subjects (ndarray (SxQxP)): true connectivity weights for subjects
        sigma2_epss (ndarray (SxP)): sigma2_eps (1 scalar for each voxel) for each subject
    Returns:
        Y_subjects (ndarray (SxNxP)): measurement (Y_i = Y* + eps) for subjects
        Y_star_subjects (ndarray (SxNxP)): true cerebellar activity for each subject (Y*)
    """

    Y_star_subjects = np.empty((S, N, P))
    Y_subjects = np.empty((S, N, P))
    for s in range(S):
        X = X_subjects[s, :, :]
        W = W_subjects[s, :, :]
        sigma2_eps = sigma2_epss[s, :]
        eps = np.empty((N, P))
        for v in range(P):
            eps[:, v] = np.random.normal(0, np.sqrt(sigma2_eps[v]), size=N)

        Y_star = X @ W
        Y_star_subjects[s, :, :] = Y_star
        Y_subjects[s, :, :] = Y_star + eps
    return Y_subjects, Y_star_subjects


def estimate_W(X_subjects, Y_subjects, alpha, sigma2_epss):
    """ Estimates W from the observed X and Y with Ridge regression.
    Args:
        X_subjects (ndarray (SxNxQ)): cortical data for subjects
        Y_subjects (ndarray (SxNxP)): measurement (Y_i = Y* + eps) for subjects
        alpha (scalar): alpha in Ridge regression
        sigma2_epss (ndarray (SxP)): sigma2_eps (1 scalar for each voxel) for each subject
    Returns:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
        Var_W_hat_subjects (ndarray (SxP)): estimated variance (uncertainity) of weights for subjects
    """

    W_hat_subjects = np.empty((S, Q, P))
    Var_W_hat_subjects = np.empty((S, P))
    for s in range(S):
        X = X_subjects[s, :, :] - np.nanmean(X_subjects[s, :, :])
        Y = Y_subjects[s, :, :] #- np.nanmean(Y_subjects[s, :, :])
        sigma2_eps = sigma2_epss[s, :]
        A = np.linalg.inv(X.T @ X + alpha * np.eye(Q)) @ X.T

        # using model code
        conn_model = getattr(model, 'L2regression')(alpha)
        conn_model.fit(X, Y)
        W_hat_subjects[s, :, :] = conn_model.coef_.T

        Var_W_hat_subjects[s, :] = sigma2_eps * np.trace(A @ A.T)
        # print(f'Trace(A A.T): {np.trace(A @ A.T)}')
    return W_hat_subjects, Var_W_hat_subjects


def calc_model(model, W_hat_subjects, Var_W_hat_subjects):
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

    if model == 'loo':
        W_group_model = calc_model_loo(W_hat_subjects)
    elif model == 'bayes':
        W_group_model = calc_model_bayes(W_hat_subjects, Var_W_hat_subjects)
    elif model == 'bayes vox':
        W_group_model = calc_model_bayes_vox(W_hat_subjects, Var_W_hat_subjects)
    return W_group_model


def calc_model_loo(W_hat_subjects):
    """ Averages the connectivity weights of subjects in a leave-one-out manner.
    Args:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
    Returns:
        W_hat_loo (ndarray (SxQxP)): average of all other subject connectivity weights
    """

    W_hat_loo = np.empty((S, Q, P))
    for s in range(S):
        W_loo = np.delete(W_hat_subjects, s, axis=0)
        W_hat_loo[s, :, :] = np.nanmean(W_loo, axis=0)
    return W_hat_loo


def calc_model_bayes(W_hat_subjects, Var_W_hat_subjects):
    """ Combines the connectivity weights of subjects in a leave-one-out manner using Bayes integration.
    Args:
        W_hat_subjects (ndarray (SxQxP)): estimated connectivity weights for subjects
        Var_W_hat_subjects (ndarray (SxP)): estimated variance (uncertainity) of weights for subjects
    Returns:
        W_hat_opt (ndarray (SxQxP)): optimally combined connectivity weights
    """

    W_hat_opt = np.empty((S, Q, P))
    Var_W_hat_mean = np.nanmean(Var_W_hat_subjects, axis=1)
    for s in range(S):
        W_hat_loo = np.delete(W_hat_subjects, s, axis=0)
        var_loo = np.delete(Var_W_hat_mean, s, axis=0)

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

    W_hat_opt_vox = np.empty((S, Q, P))
    for s in range(S):
        W_hat_loo = np.delete(W_hat_subjects, s, axis=0)
        var_loo = np.delete(Var_W_hat_subjects, s, axis=0)

        weights = (1 / var_loo) / np.nansum(1 / var_loo, axis=0)
        W_hat_opt_vox[s, :, :] = np.nansum(W_hat_loo * weights[:, np.newaxis, :], axis=0)
    return W_hat_opt_vox


def predict_Y_hat(X_subjects, W_subjects, alpha):
    """ Predicts the cerebellar activity using the cortical activity and group connectivity weights
    Args:
        X_subjects (ndarray (SxNxQ)): cortical data for subjects
        W_subjects (ndarray (SxQxP)): group connectivity weights for subjects
    Returns:
        Y_hat_group_model (ndarray (SxNxP)): predicted cerebellar activity
    """

    conn_model = getattr(model, 'L2regression')(alpha)
    Y_hat_group_model = np.empty((S, N, P))
    for s in range(S):
        X = X_subjects[s, :, :] - np.nanmean(X_subjects[s, :, :])
        W = W_subjects[s, :, :]
        setattr(conn_model, 'coef_', W.T)
        Y_hat_group_model[s, :, :] = conn_model.predict(X)
    return Y_hat_group_model


def make_dataframe(method_dic, method_name):
    """ Creates a dataframe of methods and their evaluation metrics for visualization.
    Args:
        method_dic (dictionary): keys are evaluation metrics ('RMSE'/'R') and values are evaluation values
        method_name (strings): name of the method ('loo'/'bayes'/'bayes vox')
    Returns:
        the dataframe
    """

    subject_list = [f'sub {i}' for i in range(S)]
    columns = ['subject', 'method'] + list(method_dic.keys())
    data = {col: [] for col in columns}
    data['subject'].extend(subject_list)
    data['method'].extend([method_name] * len(subject_list))
    for eval_metric, value in method_dic.items():
        data[eval_metric].extend(value)
    return pd.DataFrame(data)


def eval_model(data, data_pred, metric='R'):
    """ Calculates the performance of models based on the evaluation metric
    Args:
        data (ndarray (QxP)/(SxAxB)): true data
        data_pred (ndarray (SxAxB)): estimated data
    Returns:
        performance (list of scalars): root mean squared error / correlation coef
    """

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
            performance.append(calculate_R(data_s, data_pred[s, :, :]))
    else:
        raise ValueError("Undefined metric. Should be either 'RMSE' or 'R'.")
    return performance


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


def calculate_R(data, data_pred):
    """ Calculates correlation between Y and Y_pred without subtracting the mean.
    Args:
        data (nd-array (AxB)): true data
        data_pred (nd-array (AxB)): prediction data
    Returns:
        R (scalar): Correlation between Y and Y_pred
    """
    SYP = np.nansum(data * data_pred)
    SPP = np.nansum(data_pred * data_pred)
    SST = np.nansum(data ** 2)
    R = SYP / np.sqrt(SST * SPP)
    return R

if __name__ == "__main__":
    # Parameters to change:
    x_true_mean = 0
    x_true_var = 0.1
    x_ind_var = 0
    w_true_mean = 1
    w_true_var = 1
    w_ind_var = 0.1
    alpha = 1

    # eval_metrics = ['RMSE', 'R']
    eval_metrics = ['R']

    # eval_parameter, eval_with, cross_validation
    eval_types = {
        'W_mean': ('W', 'mean', False),
        'W_ind': ('W', 'ind', False),
        'Y_mean': ('Y', 'mean', False),     # X_mean @ W_mean
        'Y_star': ('Y', 'star', False),     # individual activations without measurement noise
        'Y_1': ('Y', 'ind', False),
        'Y_2': ('Y', 'ind', True),
    }

    for eval_type, (eval_param, eval_with, cross_validation) in eval_types.items():
        eval_df = pd.DataFrame()
        for i in range(n_simulation):

            # Generate data
            X_subjects, X_mean = create_dataset(data_mean=x_true_mean, data_var=x_true_var, data_ind_var=x_ind_var, shape=(N, Q))
            W_subjects, W_mean = create_dataset(data_mean=w_true_mean, data_var=w_true_var, data_ind_var=w_ind_var, shape=(Q, P))

            shape_params = np.array([1.406748684275005, 1.2770534266710496, 1.1328150830096797, 1.4097048103536656, 2.064867002624763, 1.695807134957357, 2.4181751782219525, 1.6466319246865178, 2.303449282655356, 2.196768808481912, 1.6871730244950494, 2.234433894000422, 1.8194967278142473, 1.7845065275970844, 1.9901045844731078, 1.5100204369833015, 1.6734372570281622, 2.0225755760743884, 2.4965773619608242, 1.7729638629911857, 2.5112686931467536, 1.2858801082140117, 1.0713484813115728, 1.834150830155816])
            scale_params = 1000*np.array([0.48014338742523444, 0.5029592831976545, 0.3088100038552092, 0.2830578818847568, 0.3761138670159451, 0.23180716843129726, 0.20671882896774318, 0.3653442872894432, 0.17615115692680014, 0.22338571360549525, 0.4844926697630648, 0.13288777516505962, 0.18098701041274284, 0.2080933225419945, 0.18274563579463807, 0.3306029115358266, 0.2833600862305082, 0.1473968056527721, 0.2570323605425981, 0.1415003010076477, 0.13768735651996084, 0.6540178640282134, 0.823032041952238, 0.19220675226632639])
            sigma2_epss = generate_sigma2eps(shape_params=shape_params,
                                            scale_params=scale_params)

            Y1_subjects, Y_star_subjects = generate_Y(X_subjects=X_subjects,
                                                    W_subjects=W_subjects,
                                                    sigma2_epss=sigma2_epss)
            if cross_validation:
                Y2_subjects, _ = generate_Y(X_subjects=X_subjects,
                                            W_subjects=W_subjects,
                                            sigma2_epss=sigma2_epss)

            W_hat_subjects, Var_W_hat_subjects = estimate_W(X_subjects=X_subjects,
                                                            Y_subjects=Y1_subjects,
                                                            alpha=alpha,
                                                            sigma2_epss=sigma2_epss)

            # Run methods
            method_names = ['loo', 'bayes', 'bayes vox']
            W_group_model = {}
            for method in method_names:
                W_group_model[method] = calc_model(model=method, W_hat_subjects=W_hat_subjects, Var_W_hat_subjects=Var_W_hat_subjects)

            # Calculate Y_hat
            if eval_param == 'Y':
                Y_hat_group_model = {}
                for method in method_names:
                    Y_hat_group_model[method] = predict_Y_hat(X_subjects=X_subjects, W_subjects=W_group_model[method], alpha=alpha)

            # Evaluate methods
            performance = {}
            for method in method_names:
                performance[method] = {}
                for eval_metric in eval_metrics:
                    if eval_param == 'W':
                        if eval_with == 'mean':
                            performance[method][eval_metric] = eval_model(metric=eval_metric, data=W_mean, data_pred=W_group_model[method])
                        elif eval_with == 'ind':
                            performance[method][eval_metric] = eval_model(metric=eval_metric, data=W_subjects, data_pred=W_group_model[method])
                    elif eval_param == 'Y':
                        if eval_with == 'mean':
                            performance[method][eval_metric] = eval_model(metric=eval_metric, data=X_mean@W_mean, data_pred=Y_hat_group_model[method])
                        elif eval_with == 'star':
                            performance[method][eval_metric] = eval_model(metric=eval_metric, data=Y_star_subjects, data_pred=Y_hat_group_model[method])
                        elif eval_with == 'ind':
                            if cross_validation:
                                performance[method][eval_metric] = eval_model(metric=eval_metric, data=Y2_subjects, data_pred=Y_hat_group_model[method])
                            else:
                                performance[method][eval_metric] = eval_model(metric=eval_metric, data=Y1_subjects, data_pred=Y_hat_group_model[method])

                # Make dataframe
                df = make_dataframe(performance[method], method)
                eval_df = pd.concat([eval_df, df])

        # Plot
        for eval_metric in eval_metrics:
            means = eval_df.groupby('method')[eval_metric].mean().sort_values(ascending=True)
            sns.barplot(data=eval_df, x='method', y=eval_metric, hue='method', order=means.index.to_list())
            plt.title(f'Evaluation of models for type: {eval_type}')
            print(means)
            plt.show()
