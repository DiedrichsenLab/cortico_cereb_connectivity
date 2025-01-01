import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.evaluation as ev
import cortico_cereb_connectivity.run_model as rm
from scipy import stats


# Set global parameters
N = 58  # Number of tasks
Q = 200  # Number of cortical regions
P = 800  # Number of cerebellar voxels
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


def normalize_dataset(dataset, std_method='global'):
    """ This function will normalize each column of data
    Args:
        dataset (ndarray (AxB) or (SxAxB)): dataset generated
        std_method (string): how to standardize data
    Returns:
        dataset_normalized (ndarray (AxB) or (SxAxB)): normalized dataset generated
    """
    
    if dataset.ndim == 2:
        dataset = rm.std_data(dataset, std_method)
    else:
        for s in range(S):
            X = dataset[s, :, :]
            dataset[s, :, :] = rm.std_data(X, std_method)
    return dataset


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


def estimate_sigma2eps(Y):
    # Y = rm.std_data(Y, 'global')
    Y_1 = Y[:N//2, :]
    Y_2 = Y[N//2:, :]
    Y_mean = np.nanmean([Y_1, Y_2], axis=0)
    sigma2_eps = np.zeros(Y_mean.shape[1])
    sigma2_eps += np.diag((Y_1-Y_mean).T @ (Y_1-Y_mean)) / (Y_mean.shape[0])
    sigma2_eps += np.diag((Y_2-Y_mean).T @ (Y_2-Y_mean)) / (Y_mean.shape[0])
    return sigma2_eps


def estimate_W(X_subjects, Y_subjects, alpha, sigma2_epss=None):
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

    W_hat_subjects = np.empty((S, Q, P))
    Var_W_hat_subjects = np.empty((S, P))
    true_Var_W_hat_subjects = np.empty((S, P))
    for s in range(S):
        X = X_subjects[s, :, :] #- np.nanmean(X_subjects[s, :, :], axis=0)
        Y = Y_subjects[s, :, :] #- np.nanmean(Y_subjects[s, :, :], axis=0)
        A = np.linalg.inv(X.T @ X + alpha * np.eye(Q)) @ X.T
        W_hat_subjects[s, :, :] = A @ Y

        Var_W_hat_subjects[s, :] = estimate_sigma2eps(Y) * np.trace(A @ A.T)

        if sigma2_epss is not None:
            sigma2_eps = sigma2_epss[s, :]
            true_Var_W_hat_subjects[s, :] = sigma2_eps * np.trace(A @ A.T)
    if sigma2_epss is None:
        return W_hat_subjects, Var_W_hat_subjects
    else:
        return W_hat_subjects, Var_W_hat_subjects, true_Var_W_hat_subjects


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


def simulate_all():
    # Parameters to change:
    param_set = 3
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

    alpha_list = [np.exp(8)]

    # eval_metrics = ['RMSE', 'R', 'R2']
    eval_metrics = ['R']#, 'R2']
    nc = False

    # eval_parameter, eval_with, cross_validation
    eval_types = {
        'W_mean':   ('W',   'mean',   False),
        # 'W_ind':    ('W',   'ind',    False),
        # 'Y_mean':   ('Y',   'mean',   False),     # X_mean @ W_mean
        # 'Y_star':   ('Y',   'star',   False),     # individual activations without measurement noise
        # 'Y_1':      ('Y',   'ind',    False),
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
                X_subjects = normalize_dataset(dataset=X_subjects, std_method='parcel')
                X_mean = normalize_dataset(dataset=X_mean, std_method='parcel')
                W_subjects, W_mean = create_dataset(data_mean=w_true_mean, data_var=w_true_var, data_ind_var=w_ind_var, shape=(Q, P))
                
                # shape_params = np.array([1.1473032982361686, 1.2951274520344758, 0.9679436505658012, 1.8452195681630008, 1.7822897660345185, 1.607088495022598, 3.0570816937067122, 1.6375526562271254, 2.196827625986511, 1.6968870328825958, 1.5361470793690848, 1.8737230591204828, 2.5429175638943415, 1.7508963622765272, 1.8814280106435932, 1.689557858582146, 1.642025348856807, 2.262145061135111, 2.0946819405313866, 1.615498520643554, 2.593505731931406, 1.3085170629883034, 0.9076669866982224, 1.7745448565227115])
                # scale_params = np.array([0.0066259928636138365, 0.004441351577975239, 0.005469882539545577, 0.0014597200297646815, 0.005874380248508628, 0.002968258552110554, 0.0012413757517457087, 0.003660177050149884, 0.0024257992930657455, 0.0035196654276645075, 0.004251923597306932, 0.0019981405813116384, 0.0009927134345150818, 0.002412156771064612, 0.002684417548898932, 0.0023342178689106314, 0.0032771103172777832, 0.001363372580933022, 0.003734655471652732, 0.0018394035829100242, 0.0014278704511578569, 0.007163116054806027, 0.013264794964631333, 0.0023458007943860113])
                # shape_params = np.array([1.228641386995552, 1.3230495338574872, 1.1051873905660858, 1.8329214741152584, 1.9106942551865682, 1.7459275571322626, 3.1437552697356006, 1.8667236546084651, 2.209309224161418, 1.9938981362184156, 1.6521100634205743, 2.086548746382947, 2.5690246669720556, 1.8606238138570024, 1.9726085374308622, 1.888532852188271, 1.689594082264879, 2.3883105111290117, 2.257069518736845, 1.7963955469413289, 2.622036171495563, 1.3811471223405907, 0.9432647560928347, 1.9070708060824326])
                # scale_params = np.array([0.005223529748895295, 0.0040515265373286735, 0.0036867908890735575, 0.0014097026694733445, 0.0046904481341335895, 0.002374764350847169, 0.0011417108962444317, 0.0027917899289833293, 0.002111111227179373, 0.0024723872868413067, 0.003573713933312996, 0.0015346334483584533, 0.0009355787940918994, 0.002071921378518455, 0.0022712878872298563, 0.0018989230316653033, 0.0029254754540701047, 0.0011753369398176624, 0.0029858056990220844, 0.0014374006354268968, 0.0013012982620218146, 0.006096842346727135, 0.010654281862905608, 0.0019483355993102575])
                shape_params = np.array([1.147303298236169, 1.2951274520344769, 0.9679436505658018, 1.8452195681630037, 1.7822897660345203, 1.6070884950225959, 3.0570816937067327, 1.6375526562271243, 2.1968276259865145, 1.6968870328825905, 1.5361470793690861, 1.8737230591204794, 2.542917563894335, 1.7508963622765255, 1.8814280106435888, 1.6895578585821485, 1.6420253488568122, 2.262145061135117, 2.0946819405313852, 1.615498520643554, 2.5935057319314105, 1.3085170629883034, 0.9076669866982234, 1.7745448565227135])
                scale_params = np.array([0.6637798380267982, 0.46657192379123447, 0.9632069640215286, 0.22398335299700103, 0.498003478011745, 0.48046865555341634, 0.15497464833030872, 0.4236359668707756, 0.38499293370453785, 0.4722049769668939, 0.35736278023811796, 0.42706672536184603, 0.18579466478136689, 0.4060979729144744, 0.4740329939830097, 0.30344052070407035, 0.433431972919698, 0.2799293666758564, 0.38761037408371685, 0.41915654448113043, 0.27181207571727045, 0.5612981305479915, 0.986969927097492, 0.4282710771621106])


                sigma2_epss = generate_sigma2eps(shape_params=shape_params,
                                                 scale_params=scale_params)

                Y1_subjects, Y_star_subjects = generate_Y(X_subjects=X_subjects,
                                                          W_subjects=W_subjects,
                                                          sigma2_epss=sigma2_epss)
                # X_subjects = normalize_dataset(dataset=X_subjects, std_method='parcel')
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

                W_hat_subjects, Var_W_hat_subjects, true_Var_W_hat_subjects = estimate_W(X_subjects=X_subjects,
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

def simulate_variance():
    # Change global parameters
    N = 58  # Number of tasks
    Q = 200  # Number of cortical regions
    global P
    P = 50  # Number of cerebellar voxels
    global S
    S = 1  # Number of subjects
    n_simulation = 1000 # Number of simulations

    x_true_mean = 0
    x_true_var = 1.1e-2
    x_ind_var = 0
    w_true_mean = 2.5e-4
    w_true_var = 1.3e-7
    w_ind_var = 0

    alpha = np.exp(8)

    all_w_hat = np.zeros((Q, P, n_simulation))
    all_var_w_hat = np.zeros((P, n_simulation))
    shape_params = np.array([1.147303298236169, 1.2951274520344769, 0.9679436505658018, 1.8452195681630037, 1.7822897660345203, 1.6070884950225959, 3.0570816937067327, 1.6375526562271243, 2.1968276259865145, 1.6968870328825905, 1.5361470793690861, 1.8737230591204794, 2.542917563894335, 1.7508963622765255, 1.8814280106435888, 1.6895578585821485, 1.6420253488568122, 2.262145061135117, 2.0946819405313852, 1.615498520643554, 2.5935057319314105, 1.3085170629883034, 0.9076669866982234, 1.7745448565227135])
    scale_params = np.array([0.6637798380267982, 0.46657192379123447, 0.9632069640215286, 0.22398335299700103, 0.498003478011745, 0.48046865555341634, 0.15497464833030872, 0.4236359668707756, 0.38499293370453785, 0.4722049769668939, 0.35736278023811796, 0.42706672536184603, 0.18579466478136689, 0.4060979729144744, 0.4740329939830097, 0.30344052070407035, 0.433431972919698, 0.2799293666758564, 0.38761037408371685, 0.41915654448113043, 0.27181207571727045, 0.5612981305479915, 0.986969927097492, 0.4282710771621106])
    sigma2_epss = generate_sigma2eps(shape_params=shape_params,
                                     scale_params=scale_params)
    
    X_subjects, X_mean = create_dataset(data_mean=x_true_mean, data_var=x_true_var, data_ind_var=x_ind_var, shape=(N, Q))
    # X_subjects = normalize_dataset(dataset=X_subjects, std_method='parcel')
    # X_mean = normalize_dataset(dataset=X_mean, std_method='parcel')
    W_subjects, W_mean = create_dataset(data_mean=w_true_mean, data_var=w_true_var, data_ind_var=w_ind_var, shape=(Q, P))
    
    for n in range(n_simulation):
        Y1_subjects, _ = generate_Y(X_subjects=X_subjects,
                                    W_subjects=W_subjects,
                                    sigma2_epss=sigma2_epss)
        # Y1_subjects = normalize_dataset(dataset=Y1_subjects, std_method='global')
        
        W_hat_subjects, Var_W_hat_subjects = estimate_W(X_subjects=X_subjects,
                                                        Y_subjects=Y1_subjects,
                                                        alpha=alpha)
        
        all_w_hat[:, :, n] = np.squeeze(W_hat_subjects)
        all_var_w_hat[:, n] = np.squeeze(Var_W_hat_subjects)

    df = pd.DataFrame()
    df["simulation_points"] = np.repeat(np.unique(np.logspace(0, 3, 20, dtype=int)), 2*P)
    w_var = []
    label = []
    for point in np.unique(df["simulation_points"]):
        w_var.extend(np.mean(all_var_w_hat[:, :point], axis=1))
        label.extend(['estimated' for _ in range(P)])
        w_var.extend(np.sum(np.var(all_w_hat[:, :, :point], axis=2), axis=0))
        label.extend(['empirical' for _ in range(P)])
    
    df["w_var"] = w_var
    df["mode"] = label
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=df, x='simulation_points', y='w_var', hue='mode', errorbar='se')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(f'Simulation Results Over Iterations (P={P})')
    plt.xscale('log')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Var(W)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # simulate_all()
    simulate_variance()

