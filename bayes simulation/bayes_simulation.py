import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.evaluation as ev
from scipy import stats


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


def normalize_dataset(data_subjects, data_mean):
    """ This function will normalize each column of data
    Args:
        data_subjects (ndarray (SxAxB)): realizations of the true dataset
        dataset_mean (ndarray (AxB)): true dataset generated
    Returns:
        data_subjects (ndarray (SxAxB)): normalized realizations of the true dataset
        dataset_mean (ndarray (AxB)): normalized true dataset generated
    """
    
    data_mean /= np.sqrt(np.nansum(data_mean ** 2, 0) / data_mean.shape[0])
    for s in range(S):
        X = data_subjects[s, :, :]
        data_subjects[s, :, :] = X / np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])
    return data_subjects, data_mean


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


def make_dataframe(method_dic, method_name, alpha, nc=None):
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


def evaluate_model_with_params(method, eval_metric, eval_param, eval_with, cv):
    """ Calls the eval_model() with appropriate inputs
    """
    if eval_param == 'W':
        if eval_with == 'mean':
            performance = eval_model(metric=eval_metric,
                                     data=W_mean,
                                     data_pred=W_group_model[method])
        elif eval_with == 'ind':
            performance = eval_model(metric=eval_metric,
                                     data=W_subjects,
                                     data_pred=W_group_model[method])
    elif eval_param == 'Y':
        if eval_with == 'mean':
            performance = eval_model(metric=eval_metric,
                                     data=X_mean@W_mean,
                                     data_pred=Y_hat_group_model[method])
        elif eval_with == 'star':
            performance = eval_model(metric=eval_metric,
                                     data=Y_star_subjects,
                                     data_pred=Y_hat_group_model[method])
        elif eval_with == 'ind':
            if cv:
                performance = eval_model(metric=eval_metric,
                                         data=Y2_subjects,
                                         data_pred=Y_hat_group_model[method])
            else:
                performance = eval_model(metric=eval_metric,
                                         data=Y1_subjects,
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


if __name__ == "__main__":
    # Parameters to change:
    param_set = 2
    save_df = False
    plot_df = True

    if param_set==0:
        # MDTB estimations:
        x_true_mean = 0
        x_true_var = 1.1e-2
        x_ind_var = 0
        w_true_mean = 2.5e-6
        w_true_var = 1.3e-9
        w_ind_var = 1.1e-8
    elif param_set==1:
        # Works:
        x_true_mean = 0
        x_true_var = 1.1e-2
        x_ind_var = 0
        w_true_mean = 2.5e-4
        w_true_var = 1.3e-7
        w_ind_var = 1.1e-6
    elif param_set==2:
        # Test:
        x_true_mean = 0
        x_true_var = 1.1e-2
        x_ind_var = 0
        w_true_mean = 2.5e-3
        w_true_var = 1.3e-7
        w_ind_var = 1.1e-6
    elif param_set==3:
        # Test:
        x_true_mean = 0
        x_true_var = 1.1e-2
        x_ind_var = 0
        w_true_mean = 2.5e-2
        w_true_var = 1.3e-7
        w_ind_var = 1.1e-6

    alpha_list = [np.exp(8)]#, np.exp(12)]

    # eval_metrics = ['RMSE', 'R', 'R2']
    eval_metrics = ['R', 'R2']
    nc = False

    # eval_parameter, eval_with, cross_validation
    eval_types = {
        'W_mean':   ('W',   'mean',   False),
        'W_ind':    ('W',   'ind',    False),
        'Y_mean':   ('Y',   'mean',   False),     # X_mean @ W_mean
        'Y_star':   ('Y',   'star',   False),     # individual activations without measurement noise
        'Y_1':      ('Y',   'ind',    False),
        'Y_2':      ('Y',   'ind',    True),
    }

    save_path = '/cifs/diedrichsen/data/Cerebellum/connectivity/SUIT3/bayes_simulation/'

    for eval_type, (eval_param, eval_with, cross_validation) in eval_types.items():
        eval_df = pd.DataFrame()
        save_this_eval_path = save_path + f"{eval_type}_param{param_set}.tsv"
        for alpha in alpha_list:
            for i in range(n_simulation):

                # Generate data
                X_subjects, X_mean = create_dataset(data_mean=x_true_mean, data_var=x_true_var, data_ind_var=x_ind_var, shape=(N, Q))
                X_subjects, X_mean = normalize_dataset(data_subjects=X_subjects, data_mean=X_mean)
                W_subjects, W_mean = create_dataset(data_mean=w_true_mean, data_var=w_true_var, data_ind_var=w_ind_var, shape=(Q, P))
                
                shape_params = np.array([1.1473032982361686, 1.2951274520344758, 0.9679436505658012, 1.8452195681630008, 1.7822897660345185, 1.607088495022598, 3.0570816937067122, 1.6375526562271254, 2.196827625986511, 1.6968870328825958, 1.5361470793690848, 1.8737230591204828, 2.5429175638943415, 1.7508963622765272, 1.8814280106435932, 1.689557858582146, 1.642025348856807, 2.262145061135111, 2.0946819405313866, 1.615498520643554, 2.593505731931406, 1.3085170629883034, 0.9076669866982224, 1.7745448565227115])
                scale_params = np.array([0.0066259928636138365, 0.004441351577975239, 0.005469882539545577, 0.0014597200297646815, 0.005874380248508628, 0.002968258552110554, 0.0012413757517457087, 0.003660177050149884, 0.0024257992930657455, 0.0035196654276645075, 0.004251923597306932, 0.0019981405813116384, 0.0009927134345150818, 0.002412156771064612, 0.002684417548898932, 0.0023342178689106314, 0.0032771103172777832, 0.001363372580933022, 0.003734655471652732, 0.0018394035829100242, 0.0014278704511578569, 0.007163116054806027, 0.013264794964631333, 0.0023458007943860113])

                sigma2_epss = generate_sigma2eps(shape_params=shape_params,
                                                 scale_params=scale_params)

                Y1_subjects, Y_star_subjects = generate_Y(X_subjects=X_subjects,
                                                          W_subjects=W_subjects,
                                                          sigma2_epss=sigma2_epss)
                if cross_validation:
                    Y2_subjects, _ = generate_Y(X_subjects=X_subjects,
                                                W_subjects=W_subjects,
                                                sigma2_epss=sigma2_epss)
                                        
                # for s in range(S):
                #     mn = Y2_subjects[s,:,:]-Y_star_subjects[s,:,:]
                #     mu, std = stats.norm.fit(mn.flatten())
                #     plt.hist(mn.flatten(), bins=30, alpha=0.5, label=f'noise~N({mu:.1e},{std:.1e})', color='blue', density=True)
                #     mu, std = stats.norm.fit(Y_star_subjects[s,:,:].flatten())
                #     plt.hist(Y_star_subjects[s,:,:].flatten(), alpha=0.5, label=f'Y_star~N({mu:.1e},{std:.1e})', color='red', density=True)
                #     mu, std = stats.norm.fit(Y2_subjects[s,:,:].flatten())
                #     plt.hist(Y2_subjects[s,:,:].flatten(), bins=30, alpha=0.5, label=f'Y_2~N({mu:.1e},{std:.1e})', color='green', density=True)
                #     plt.legend()
                #     plt.show()

                W_hat_subjects, Var_W_hat_subjects = estimate_W(X_subjects=X_subjects,
                                                                Y_subjects=Y1_subjects,
                                                                alpha=alpha,
                                                                sigma2_epss=sigma2_epss)

                # Run methods
                method_names = ['loo', 'bayes', 'bayes vox']
                W_group_model = {}
                for method in method_names:
                    W_group_model[method] = calc_model(model=method,
                                                       W_hat_subjects=W_hat_subjects,
                                                       Var_W_hat_subjects=Var_W_hat_subjects)

                # Calculate Y_hat
                if eval_param == 'Y':
                    Y_hat_group_model = {}
                    for method in method_names:
                        Y_hat_group_model[method] = predict_Y_hat(X_subjects=X_subjects,
                                                                  W_subjects=W_group_model[method],
                                                                  alpha=alpha)

                # Evaluate methods
                performance = {}
                for method in method_names:
                    performance[method] = {}
                    for eval_metric in eval_metrics:
                        performance[method][eval_metric] = evaluate_model_with_params(method=method,
                                                                                      eval_metric=eval_metric,
                                                                                      eval_param=eval_param,
                                                                                      eval_with=eval_with,
                                                                                      cv=cross_validation)
                    if eval_type == 'Y_2':
                        noise_ceiling = calc_noise_ceiling(X_subjects=X_subjects, W_model=W_group_model[method], Y1_subjects=Y1_subjects, Y2_subjects=Y2_subjects)
                        # Make dataframe
                        df = make_dataframe(performance[method], method, alpha, nc=noise_ceiling)
                        eval_df = pd.concat([eval_df, df])
                    else:
                        df = make_dataframe(performance[method], method, alpha)
                        eval_df = pd.concat([eval_df, df])

        if save_df:
            # Save the dataframe for future visualizations
            eval_df.to_csv(save_this_eval_path, sep="\t")

        if plot_df:
            for eval_metric in eval_metrics:
                if eval_metric == 'R' and eval_type == 'Y_2' and nc == True:
                    eval_df['R_eval_adj'] = eval_df['R'] / eval_df['noise_ceiling']
                    means = eval_df.groupby('method')['R_eval_adj'].mean().sort_values(ascending=True)
                    sns.barplot(data=eval_df, x='method', y='R_eval_adj', hue='method', order=means.index.to_list())
                else:
                    means = eval_df.groupby('method')[eval_metric].mean().sort_values(ascending=True)
                    sns.barplot(data=eval_df, x='method', y=eval_metric, hue='method', order=means.index.to_list())
                plt.title(f'Evaluation of models for type: {eval_type}')
                print(means)
                plt.show()