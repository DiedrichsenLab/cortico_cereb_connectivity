import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cortico_cereb_connectivity.model as model
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.evaluation as ev
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
from cortico_cereb_connectivity.bayes_simulation.simulation_functions import *
import Functional_Fusion.dataset as ds
from scipy import stats
from statannotations.Annotator import Annotator
from IPython.display import display


# Set global parameters
N = 58  # Number of tasks
Q = 100  # Number of cortical regions
P = 400  # Number of cerebellar voxels
S = 24  # Number of subjects
n_simulation = 50 # Number of simulations


def simulate_all():
    # Parameters to change:
    param_set = 4
    save_df = False
    plot_df = True

    if param_set==0:
        # MDTB estimations:
        x_group_mean = 0
        x_sigma2_g = 1.1e-2
        x_sigma2_s = 0
        w_group_mean = 2.5e-6
        w_sigma2_g = 1.3e-9
        w_sigma2_s = 1.1e-8
    elif param_set==1:
        # Works:
        x_group_mean = 0
        x_sigma2_g = 1.1e-2
        x_sigma2_s = 0
        w_group_mean = 2.5e-4
        w_sigma2_g = 1.3e-7
        w_sigma2_s = 1.1e-6
    elif param_set==2:
        # Test:
        x_group_mean = 0
        x_sigma2_g = 1.1e-2
        x_sigma2_s = 0
        w_group_mean = 2.5e-3
        w_sigma2_g = 1.3e-7
        w_sigma2_s = 1.1e-6
    elif param_set==3:
        # Test:
        x_group_mean = 0
        x_sigma2_g = 1.1e-2
        x_sigma2_s = 0
        w_group_mean = 2.5e-2
        w_sigma2_g = 1.3e-6
        w_sigma2_s = 1.1e-6
    elif param_set==4:
        x_group_mean = 0
        x_sigma2_g = 1
        x_sigma2_s = 0
        w_group_mean = 0
        w_sigma2_g = 8e-4
        w_sigma2_s = 8e-4

    alpha_list = [np.exp(8)]

    # eval_metrics = ['RMSE', 'R', 'R2']
    eval_metrics = ['R']#, 'R2']
    nc = False

    # eval_parameter, eval_with, cross_validation
    eval_types = {
        'W_group':   ('W',   'group',   False),
        # 'W_ind':    ('W',   'ind',    False),
        # 'Y_group':   ('Y',   'group',   False),     # X_group @ W_group
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
                X_i, X_group = create_dataset(data_mean=x_group_mean, data_var=x_sigma2_g, data_ind_var=x_sigma2_s, shape=(S, N//2, Q), same_half=True)
                # X_i = normalize_dataset(dataset=X_i, std_method='parcel')
                # X_group = normalize_dataset(dataset=X_group, std_method='parcel')
                W_i, W_group = create_dataset(data_mean=w_group_mean, data_var=w_sigma2_g, data_ind_var=w_sigma2_s, shape=(S, Q, P))
                
                # shape_params = np.array([1.1473032982361686, 1.2951274520344758, 0.9679436505658012, 1.8452195681630008, 1.7822897660345185, 1.607088495022598, 3.0570816937067122, 1.6375526562271254, 2.196827625986511, 1.6968870328825958, 1.5361470793690848, 1.8737230591204828, 2.5429175638943415, 1.7508963622765272, 1.8814280106435932, 1.689557858582146, 1.642025348856807, 2.262145061135111, 2.0946819405313866, 1.615498520643554, 2.593505731931406, 1.3085170629883034, 0.9076669866982224, 1.7745448565227115])
                # scale_params = np.array([0.0066259928636138365, 0.004441351577975239, 0.005469882539545577, 0.0014597200297646815, 0.005874380248508628, 0.002968258552110554, 0.0012413757517457087, 0.003660177050149884, 0.0024257992930657455, 0.0035196654276645075, 0.004251923597306932, 0.0019981405813116384, 0.0009927134345150818, 0.002412156771064612, 0.002684417548898932, 0.0023342178689106314, 0.0032771103172777832, 0.001363372580933022, 0.003734655471652732, 0.0018394035829100242, 0.0014278704511578569, 0.007163116054806027, 0.013264794964631333, 0.0023458007943860113])
                # shape_params = np.array([1.228641386995552, 1.3230495338574872, 1.1051873905660858, 1.8329214741152584, 1.9106942551865682, 1.7459275571322626, 3.1437552697356006, 1.8667236546084651, 2.209309224161418, 1.9938981362184156, 1.6521100634205743, 2.086548746382947, 2.5690246669720556, 1.8606238138570024, 1.9726085374308622, 1.888532852188271, 1.689594082264879, 2.3883105111290117, 2.257069518736845, 1.7963955469413289, 2.622036171495563, 1.3811471223405907, 0.9432647560928347, 1.9070708060824326])
                # scale_params = np.array([0.005223529748895295, 0.0040515265373286735, 0.0036867908890735575, 0.0014097026694733445, 0.0046904481341335895, 0.002374764350847169, 0.0011417108962444317, 0.0027917899289833293, 0.002111111227179373, 0.0024723872868413067, 0.003573713933312996, 0.0015346334483584533, 0.0009355787940918994, 0.002071921378518455, 0.0022712878872298563, 0.0018989230316653033, 0.0029254754540701047, 0.0011753369398176624, 0.0029858056990220844, 0.0014374006354268968, 0.0013012982620218146, 0.006096842346727135, 0.010654281862905608, 0.0019483355993102575])
                # shape_params = np.array([1.147303298236169, 1.2951274520344769, 0.9679436505658018, 1.8452195681630037, 1.7822897660345203, 1.6070884950225959, 3.0570816937067327, 1.6375526562271243, 2.1968276259865145, 1.6968870328825905, 1.5361470793690861, 1.8737230591204794, 2.542917563894335, 1.7508963622765255, 1.8814280106435888, 1.6895578585821485, 1.6420253488568122, 2.262145061135117, 2.0946819405313852, 1.615498520643554, 2.5935057319314105, 1.3085170629883034, 0.9076669866982234, 1.7745448565227135])
                # scale_params = np.array([0.6637798380267982, 0.46657192379123447, 0.9632069640215286, 0.22398335299700103, 0.498003478011745, 0.48046865555341634, 0.15497464833030872, 0.4236359668707756, 0.38499293370453785, 0.4722049769668939, 0.35736278023811796, 0.42706672536184603, 0.18579466478136689, 0.4060979729144744, 0.4740329939830097, 0.30344052070407035, 0.433431972919698, 0.2799293666758564, 0.38761037408371685, 0.41915654448113043, 0.27181207571727045, 0.5612981305479915, 0.986969927097492, 0.4282710771621106])

                # sigma2_epss = generate_sigma2eps(shape_params=shape_params,
                #                                  scale_params=scale_params,
                #                                  shape=(S, P))

                sigma2_epss = 50 * np.tile(np.random.normal(loc=np.sqrt(w_sigma2_s)*5, scale=3*np.sqrt(w_sigma2_s), size=(S,))**2, (P, 1)).T

                # plt.hist(sigma2_epss[0,:])
                # plt.show()

                Y1_subjects, Y_star_subjects = generate_Y(X_subjects=X_i,
                                                          W_subjects=W_i,
                                                          sigma2_epss=sigma2_epss,
                                                          shape=(S, N, P))

                Y2_subjects, _ = generate_Y(X_subjects=X_i,
                                            W_subjects=W_i,
                                            sigma2_epss=sigma2_epss,
                                            shape=(S, N, P))
                
                # SNR
                SNR_db = calc_SNR(Y1_subjects, Y_star_subjects)
                print(f'mean SNR: {np.mean(SNR_db):.2f} db')
                                        
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

                W_i_hat, w_sigma2_w_hat, w_sigma2_w = estimate_W(X_subjects=X_i,
                                                                 Y_subjects=Y1_subjects,
                                                                 alpha=alpha,
                                                                 shape=(S, Q, P),
                                                                 sigma2_epss=sigma2_epss)
                
                W_i_hat_2, w_sigma2_w_hat_2, _ = estimate_W(X_subjects=X_i,
                                                            Y_subjects=Y2_subjects,
                                                            alpha=alpha,
                                                            shape=(S, Q, P),
                                                            sigma2_epss=sigma2_epss)
                
                # estimate the sigma2_g, sigma2_s, and sigma2_m
                vg, vs, vm = decompose_variance(np.stack((W_i_hat, W_i_hat_2), axis=1))

                # control
                # print(f'diff sigma2_w: {(vm-np.mean(w_sigma2_w,axis=1)/Q) / (np.mean(w_sigma2_w,axis=1)/Q) * 100}%')
                # print(f'diff sigma2_w: {(np.mean((w_sigma2_w_hat+w_sigma2_w_hat_2)/2,axis=1)-np.mean(w_sigma2_w,axis=1)) / np.mean(w_sigma2_w,axis=1) * 100}%')

                # Run methods
                method_names = ['loo', 'bayes', 'bayes new']#, 'bayes vox', 'bayes vox new']
                W_group_model = {}
                for method in method_names:
                    W_group_model[method] = calc_model(model=method,
                                                       W_hat_subjects=W_i_hat,
                                                       sigma2_w_hat=vm[:,np.newaxis],
                                                       sigma2_s_hat=vs[:,np.newaxis])

                # Calculate Y_hat
                if eval_param == 'Y':
                    Y_hat_group_model = {}
                    for method in method_names:
                        Y_hat_group_model[method] = predict_Y_hat(X_subjects=X_i,
                                                                  W_subjects=W_group_model[method])
                else:
                    Y_hat_group_model = None

                # Evaluate methods
                eval_data = {}
                eval_data['W_group'] = W_group
                eval_data['W_ind'] = W_i
                eval_data['Y_group'] = X_group @ W_group
                eval_data['Y_star'] = Y_star_subjects
                eval_data['Y_1'] = Y1_subjects
                if cross_validation:
                    eval_data['Y_2'] = Y2_subjects
                performance = {}
                for method in method_names:
                    performance[method] = {}
                    for eval_metric in eval_metrics:
                        performance[method][eval_metric] = evaluate_model_with_params(method=method,
                                                                                      eval_metric=eval_metric,
                                                                                      eval_param=eval_param,
                                                                                      eval_with=eval_with,
                                                                                      cv=cross_validation,
                                                                                      eval_data=eval_data,
                                                                                      W_group_model=W_group_model,
                                                                                      Y_hat_group_model=Y_hat_group_model)
                    if False:#eval_type == 'Y_2':
                        noise_ceiling = calc_noise_ceiling(X_subjects=X_i, W_model=W_group_model[method], Y1_subjects=Y1_subjects, Y2_subjects=Y2_subjects)
                        # Make dataframe
                        df = make_dataframe(performance[method], method, alpha, S, nc=noise_ceiling)
                        eval_df = pd.concat([eval_df, df])
                    else:
                        df = make_dataframe(performance[method], method, alpha, S)
                        eval_df = pd.concat([eval_df, df])

        if save_df:
            # Save the dataframe for future visualizations
            eval_df.to_csv(save_this_eval_path, sep="\t")

        if plot_df:
            for eval_metric in eval_metrics:
                if eval_metric == 'R' and eval_type == 'Y_2' and nc == True:
                    eval_df['R_eval_adj'] = eval_df['R'] / eval_df['noise_ceiling']
                    means = eval_df.groupby('method')['R_eval_adj'].mean().sort_values(ascending=True)
                    ax = sns.barplot(data=eval_df, x='method', y='R_eval_adj', hue='method', order=means.index.to_list())
                else:
                    means = eval_df.groupby('method')[eval_metric].mean().sort_values(ascending=True)
                    ax = sns.barplot(data=eval_df, x='method', y=eval_metric, hue='method', order=means.index.to_list())
                for i,_ in enumerate(eval_df.method.unique()):
                    ax.bar_label(ax.containers[i], fontsize=10, fmt='%.3f')
                plt.title(f'Evaluation of models for type: {eval_type}')
                print(means)
                plt.show()

def simulate_variance():
    # Change global parameters
    global N
    N = 58  # Number of tasks
    global Q
    Q = 100  # Number of cortical regions
    global P
    P = 400  # Number of cerebellar voxels
    global S
    S = 1  # Number of subjects
    global n_simulation
    n_simulation = 100 # Number of simulations

    x_group_mean = 0
    x_sigma2_g = 1 #1.1e-2
    x_sigma2_s = 0
    w_group_mean = 0 #2.5e-4
    w_sigma2_g = 1
    w_sigma2_s = 0

    alpha = np.exp(2)

    all_w_hat = np.zeros((Q, P, n_simulation))
    all_var_w_hat = np.zeros((P, n_simulation))
    all_true_var_w_hat = np.zeros((P, n_simulation))
    shape_params = np.array([1.147303298236169, 1.2951274520344769, 0.9679436505658018, 1.8452195681630037, 1.7822897660345203, 1.6070884950225959, 3.0570816937067327, 1.6375526562271243, 2.1968276259865145, 1.6968870328825905, 1.5361470793690861, 1.8737230591204794, 2.542917563894335, 1.7508963622765255, 1.8814280106435888, 1.6895578585821485, 1.6420253488568122, 2.262145061135117, 2.0946819405313852, 1.615498520643554, 2.5935057319314105, 1.3085170629883034, 0.9076669866982234, 1.7745448565227135])
    scale_params = np.array([0.6637798380267982, 0.46657192379123447, 0.9632069640215286, 0.22398335299700103, 0.498003478011745, 0.48046865555341634, 0.15497464833030872, 0.4236359668707756, 0.38499293370453785, 0.4722049769668939, 0.35736278023811796, 0.42706672536184603, 0.18579466478136689, 0.4060979729144744, 0.4740329939830097, 0.30344052070407035, 0.433431972919698, 0.2799293666758564, 0.38761037408371685, 0.41915654448113043, 0.27181207571727045, 0.5612981305479915, 0.986969927097492, 0.4282710771621106])
    sigma2_epss = generate_sigma2eps(shape_params=shape_params,
                                     scale_params=scale_params,
                                     shape=(S, P))
    
    X_i, X_group = create_dataset(data_mean=x_group_mean, data_var=x_sigma2_g, data_ind_var=x_sigma2_s, shape=(S, N//2, Q), same_half=True)
    # X_i = normalize_dataset(dataset=X_i, std_method='parcel')
    # X_group = normalize_dataset(dataset=X_group, std_method='parcel')
    W_i, W_group = create_dataset(data_mean=w_group_mean, data_var=w_sigma2_g, data_ind_var=w_sigma2_s, shape=(S, Q, P))
    
    for n in range(n_simulation):
        Y1_subjects, Y_star_subjects = generate_Y(X_subjects=X_i,
                                                  W_subjects=W_i,
                                                  sigma2_epss=sigma2_epss,
                                                  shape=(S, N, P))
        # Y1_subjects = normalize_dataset(dataset=Y1_subjects, std_method='global')
        SNR_db = calc_SNR(Y1_subjects, Y_star_subjects)
        
        W_hat_subjects, Var_W_hat_subjects, true_Var_W_hat_subjects = estimate_W(X_subjects=X_i,
                                                        Y_subjects=Y1_subjects,
                                                        alpha=alpha,
                                                        shape=(S, Q, P),
                                                        sigma2_epss=sigma2_epss)
        
        all_w_hat[:, :, n] = np.squeeze(W_hat_subjects)
        all_var_w_hat[:, n] = np.squeeze(Var_W_hat_subjects)
        all_true_var_w_hat[:, n] = np.squeeze(true_Var_W_hat_subjects)

    df = pd.DataFrame()
    df["simulation_points"] = np.repeat(np.unique(np.logspace(0, np.ceil(np.log10(n_simulation)), 20, dtype=int)), 3*P)
    w_var = []
    label = []
    for point in np.unique(df["simulation_points"]):
        w_var.extend(np.mean(all_var_w_hat[:, :point], axis=1))
        label.extend(['estimated' for _ in range(P)])
        w_var.extend(np.sum(np.var(all_w_hat[:, :, :point], axis=2), axis=0))
        label.extend(['empirical' for _ in range(P)])
        w_var.extend(np.mean(all_true_var_w_hat[:, :point], axis=1))
        label.extend(['real' for _ in range(P)])
    
    df["w_var"] = w_var
    df["mode"] = label
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=df, x='simulation_points', y='w_var', hue='mode', errorbar='se')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(f'Simulation Results Over Iterations (P={P}, SNR={np.mean(SNR_db):.2f} db)')
    plt.xscale('log')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Var(W)')
    plt.legend()
    plt.grid()
    plt.show()


def simulate_normalized_bayes():
    proportion = 10
    global Q
    Q = 100#1876#//proportion
    global P
    P = 500#5446#//proportion
    global S
    S = 24
    global n_simulation
    n_simulation = 50
    sigma2_g = 1
    sigma2_s = sigma2_g
    sigma_w_ranges = [[0.2, 1.4], [0.6, 4.2], [1, 7], [4, 28]]

    W_i, W_group = create_dataset(data_mean=0, data_var=sigma2_g, data_ind_var=sigma2_s, shape=(S, Q, P))

    model_names = ["Simple avg", "Simple avg", "Bayes with $\sigma^2_w$", "Bayes with $\sigma^2_w$",
                   "Bayes with $\sigma^2_s+\sigma^2_w$", "Bayes with $\sigma^2_s+\sigma^2_w$",
                   "Opt Bayes", "Opt Bayes with estimation"]
    model_types = ["non-norm", "norm", "non-norm", "norm",
                   "non-norm", "norm",
                   "norm", "norm"]

    for sigma_w_range in sigma_w_ranges:
        results = pd.DataFrame(columns=['type', 'model', 'corr'])
        a = np.sqrt(sigma2_s)*sigma_w_range[0]
        b = np.sqrt(sigma2_s)*sigma_w_range[1]
        var_mu = (a + b) / 2
        var_sigma = (b - a) / 4 
        for n in range(n_simulation):
            sigma2_w = np.random.normal(var_mu, var_sigma, size=(S,))**2

            # add noise
            noise1 = np.zeros_like(W_i)
            noise2 = np.zeros_like(W_i)
            for s in range(S):
                noise1[s] = np.random.normal(loc=0, scale=np.sqrt(sigma2_w[s]), size=(Q, P))
                noise2[s] = np.random.normal(loc=0, scale=np.sqrt(sigma2_w[s]), size=(Q, P))
            W_i_hat = W_i + noise1
            W_i_hat_2 = W_i + noise2

            # scale
            scales = np.random.normal(loc=2, scale=0.5, size=(S,))**2
            # scales = np.random.normal(loc=4, scale=2, size=(S,))**2
            W_i_hat = W_i_hat * scales[:, np.newaxis, np.newaxis]
            W_i_hat_2 = W_i_hat_2 * scales[:, np.newaxis, np.newaxis]

            # adjust the variances
            sigma2_g_sc = sigma2_g * scales**2
            sigma2_s_sc = sigma2_s * scales**2
            sigma2_w_sc = sigma2_w * scales**2

            # estimate the sigma2_g, sigma2_s, and sigma2_w
            vg, vs, vm = decompose_variance(np.stack((W_i_hat, W_i_hat_2), axis=1))

            W_i_hat_norms = np.linalg.norm(W_i_hat, axis=(1,2))
            W_i_hat_normalized = W_i_hat / W_i_hat_norms[:, np.newaxis, np.newaxis]

            # control
            # print(f'coef_var, coef_norm corr: {np.corrcoef(sigma2_w, W_i_hat_norms)[0,1]}')

            # Simple average
            W_group_avg = np.mean(W_i_hat, axis=0)

            # Simple average on normalized
            W_group_avg_norm = np.mean(W_i_hat_normalized, axis=0)

            # Bayes
            weights = 1 / sigma2_w_sc
            W_group_bayes = weighted_avg(W_i_hat, weights)

            # Bayes normalized
            weights = (W_i_hat_norms**2) / sigma2_w_sc
            W_group_bayes_norm = weighted_avg(W_i_hat_normalized, weights)

            # Bayes with sigma2_s
            weights = 1 / (sigma2_s_sc + sigma2_w_sc)
            W_group_bayes_better = weighted_avg(W_i_hat, weights)

            # Bayes with sigma_s normalized
            weights = (W_i_hat_norms**2) / (sigma2_s_sc + sigma2_w_sc)
            W_group_bayes_better_norm = weighted_avg(W_i_hat_normalized, weights)

            # Bayes Opt
            signal_norm = np.linalg.norm(W_group) * scales
            weights = signal_norm**2 / (sigma2_s_sc + sigma2_w_sc)
            W_group_bayes_opt = weighted_avg(W_i_hat / signal_norm[:, np.newaxis, np.newaxis], weights)

            # Bayes Opt est
            signal_norm2_hat = W_i_hat_norms**2 - Q*P*(vs + vm)
            weights = signal_norm2_hat / (vs + vm)
            weights[weights < 0] = 0
            W_group_bayes_opt_est = weighted_avg(W_i_hat / np.sqrt(signal_norm2_hat)[:, np.newaxis, np.newaxis], weights)

            models = [W_group_avg, W_group_avg_norm, W_group_bayes, W_group_bayes_norm,
                      W_group_bayes_better, W_group_bayes_better_norm,
                      W_group_bayes_opt, W_group_bayes_opt_est]
            
            for i, (model_name, model) in enumerate(zip(model_names, models)):
                R, _ = ev.calculate_R(W_group, model)
                results.loc[len(results)] = {'type': model_types[i], 'model': model_name, 'corr': R}
    
        # display(results)
        colors = plt.get_cmap('tab10')
        palette = {mn: colors(i) for i, mn in enumerate(list(dict.fromkeys(model_names)))}
        plt.figure(figsize=(10,5))
        ax = sns.barplot(results, x='type', y='corr', hue='model', errorbar='se', palette=palette, dodge=True, gap=0.1)
        for i, model in enumerate(results.model.unique()):
            ax.bar_label(ax.containers[i], fontsize=10, fmt='%.3f')
        plt.title(r"Correlation of $W_{{group}}$ and $\hat{{W}}_{{group}}$: $\left({:.1f} < \frac{{\sigma_w}}{{\sigma_s}} < {:.1f}\right)$".format(sigma_w_range[0], sigma_w_range[1]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()


def simulate_W_normalization():
    global Q
    Q = 100#1876
    global P
    P = 500#5446
    global S
    S = 24
    global n_simulation
    n_simulation = 10

    w_group_mean = 0
    w_sigma2_g = 1e-2
    sigma2_s = 1e-2
    sigma2_w_range = [5e-2, 1]

    W_i, W_group = create_dataset(data_mean=w_group_mean, data_var=w_sigma2_g, data_ind_var=sigma2_s, shape=(S, Q, P))

    eval_df = pd.DataFrame(columns=['model', 'R_eval', 'n_simulation'])
    for n in range(n_simulation):
        W_group_hat = {}
        scales = np.random.normal(5, 1, size=(S,))
        sigma2_w = np.random.uniform(sigma2_w_range[0], sigma2_w_range[1], size=(S,))
        W_i_hat = np.zeros_like(W_i)
        for s in range(S):
            W_i_noise = np.random.normal(0, np.sqrt(sigma2_w[s]), size=(Q, P))
            W_i_hat[s] = (W_i[s] + W_i_noise) * scales[s]
        
        # The simple average of non-normalized W_i
        W_group_hat['non-normalized'] = np.mean(W_i_hat, axis=0)

        # The simple average of normalized W_i
        W_i_hat_normalized = normalize_dataset(W_i_hat, std_method='norm')
        W_group_hat['normalized'] = np.mean(W_i_hat_normalized, axis=0)

        # evaluation
        R1, _ = ev.calculate_R(W_group, W_group_hat['non-normalized'])
        new_row1 = pd.DataFrame([{'model': 'simple average non_norm', 'R_eval': R1, 'n_simulation': n}])

        R2, _ = ev.calculate_R(W_group, W_group_hat['normalized'])
        new_row2 = pd.DataFrame([{'model': 'simple average norm', 'R_eval': R2, 'n_simulation': n}])
        
        eval_df = pd.concat([eval_df, new_row1, new_row2])

    # results
    ax = sns.barplot(eval_df, x='model', y='R_eval', errorbar='se', estimator='mean', hue='model')
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.bar_label(ax.containers[1], fontsize=10)
    plt.show()


def simulate_cheat_model():
    global S
    global N
    global P
    global Q
    x_group_mean = 0
    x_sigma2_g = 1e-5
    x_s_var = 0
    w_group_mean = 0
    w_sigma2_g = 5e-7
    sigma2_s = 5e-7

    alpha = np.exp(8)

    eval_df = pd.DataFrame(columns=["model", "train_subj", "eval_subj", "R_eval"])

    X_i, X_group = create_dataset(x_group_mean, x_sigma2_g, x_s_var, (S, N, Q))
    W_i, W_group = create_dataset(w_group_mean, w_sigma2_g, sigma2_s, (S, Q, P))

    X_group = normalize_dataset(X_group, 'parcel')
    X_i = normalize_dataset(X_i, 'parcel')

    # shape_params = np.array([1.147303298236169, 1.2951274520344769, 0.9679436505658018, 1.8452195681630037, 1.7822897660345203, 1.6070884950225959, 3.0570816937067327, 1.6375526562271243, 2.1968276259865145, 1.6968870328825905, 1.5361470793690861, 1.8737230591204794, 2.542917563894335, 1.7508963622765255, 1.8814280106435888, 1.6895578585821485, 1.6420253488568122, 2.262145061135117, 2.0946819405313852, 1.615498520643554, 2.5935057319314105, 1.3085170629883034, 0.9076669866982234, 1.7745448565227135])
    # scale_params = np.array([0.6637798380267982, 0.46657192379123447, 0.9632069640215286, 0.22398335299700103, 0.498003478011745, 0.48046865555341634, 0.15497464833030872, 0.4236359668707756, 0.38499293370453785, 0.4722049769668939, 0.35736278023811796, 0.42706672536184603, 0.18579466478136689, 0.4060979729144744, 0.4740329939830097, 0.30344052070407035, 0.433431972919698, 0.2799293666758564, 0.38761037408371685, 0.41915654448113043, 0.27181207571727045, 0.5612981305479915, 0.986969927097492, 0.4282710771621106])

    # sigma2_epss = generate_sigma2eps(shape_params, scale_params, (S, P))
    sub_sigma2_eps = np.random.gamma(0.7, 1, (S,))*1e-3
    sigma2_epss = np.zeros((S, P))
    for s in range(S):
        sigma2_epss[s] = np.full((P,), sub_sigma2_eps[s])

    Y_1, _ = generate_Y(X_i, W_i, sigma2_epss, (S, N, P))
    Y_1 = normalize_dataset(Y_1, 'global')

    Y_2, Y_star = generate_Y(X_i, W_i, sigma2_epss, (S, N, P))
    Y_star = normalize_dataset(Y_2, 'global')
    Y_2 = normalize_dataset(Y_2, 'global')

    W_i_hat, _, _ = estimate_W(X_i, Y_1, alpha, (S, Q, P), sigma2_epss)

    for s1 in range(S):
        for s2 in range(S):
            R, _ = ev.calculate_R(Y_2[s2], X_i[s2] @ W_i_hat[s1])
            eval_df.loc[len(eval_df)] = {"model": "ind", "train_subj": s1, "eval_subj": s2, "R_eval": R}

    # Simple avg
    weights = np.full(S, 1/S)
    avg_model = weighted_avg(W_i_hat, weights)

    # Cheat model
    cheat_weights = eval_df[eval_df.train_subj != eval_df.eval_subj].groupby("train_subj")["R_eval"].mean().reset_index()
    cheat_weights = cheat_weights['R_eval'].values
    cheat_weights /= np.sum(cheat_weights)
    cheat_model = weighted_avg(W_i_hat, cheat_weights)

    for s in range(S):
        R, _ = ev.calculate_R(Y_2[s], X_i[s] @ avg_model)
        eval_df.loc[len(eval_df)] = {"model": "avg", "train_subj": "avg", "eval_subj": s, "R_eval": R}
        R, _ = ev.calculate_R(Y_2[s], X_i[s] @ cheat_model)
        eval_df.loc[len(eval_df)] = {"model": "cheat", "train_subj": "cheat", "eval_subj": s, "R_eval": R}
    
    pivot_df = eval_df.pivot_table(index="train_subj", columns="eval_subj", values="R_eval", aggfunc="mean")
    plt.figure(figsize=(8,7))
    sns.heatmap(pivot_df, cbar_kws={'label': 'R_eval'}, square=True)#, vmin=0.6, vmax=1)
    plt.ylabel('Train Subject')
    plt.xlabel('Eval Subject')
    plt.show()

    plt.hist(cheat_weights)
    plt.xlabel('Cheat Weights')
    plt.show()

    ax = sns.barplot(eval_df[eval_df.model != 'ind'], x='model', y='R_eval', errorbar='se', hue='model')
    # sns.stripplot(eval_df[eval_df.model != 'ind'], x='model', y='R_eval', alpha=0.5, color=(0.0, 0.0, 0.0))
    ax.bar_label(ax.containers[0], fmt='%.4f')
    ax.bar_label(ax.containers[1], fmt='%.4f')
    
    pairs = [('avg', 'cheat')]
    annotator = Annotator(ax, pairs, data=eval_df[eval_df.model != 'ind'], x='model', y='R_eval', hue='model')
    annotator.configure(test='t-test_paired', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    plt.show()


def simulate_sigma_estimation():
    # generate some group, indiv with noise and then estimate the variances using ff.dataset
    S = 50
    A = 100
    B = 200
    sigma2_g = 1
    sigma2_s = sigma2_g
    X_i, X_group = create_dataset(0, sigma2_g, sigma2_s, shape=(S, A, B))

    X_i_hat_1 = np.empty_like(X_i)
    X_i_hat_2 = np.empty_like(X_i)
    sigma2_e = np.random.uniform(5*sigma2_s, 20*sigma2_s, (S,))
    scales = np.random.uniform(2, 20, (S,))
    # scales = np.full((S,), 1)

    for s in range(S):
        noise1 = np.random.normal(0, np.sqrt(sigma2_e[s]), (A, B))
        noise2 = np.random.normal(0, np.sqrt(sigma2_e[s]), (A, B))
        X_i_hat_1[s] = (X_i[s] + noise1) * scales[s]
        X_i_hat_2[s] = (X_i[s] + noise2) * scales[s]

    est_var = ds.decompose_pattern_into_group_indiv_noise(np.stack((X_i_hat_1, X_i_hat_2), axis=1), 'subject_wise')
    vg, vs, ve = np.split(est_var, 3, axis=1)
    vg = np.squeeze(vg)
    vs = np.squeeze(vs)
    ve = np.squeeze(ve)

    print(f'Using old variance decomposition:')
    print(f'mean error for sigma2_g: {np.mean((vg - sigma2_g*(scales**2)) / (sigma2_g*(scales**2))) * 100:.2f}%')
    print(f'mean error for sigma2_s: {np.mean((vs - sigma2_s*(scales**2)) / (sigma2_s*(scales**2))) * 100:.2f}%')
    print(f'mean error for sigma2_e: {np.mean((ve - sigma2_e*(scales**2)) / (sigma2_e*(scales**2))) * 100:.2f}%')

    print('------------------------')
    print(f'Using scale-compatible variance decomposition:')
    vg, vs, ve = decompose_variance(np.stack((X_i_hat_1, X_i_hat_2), axis=1))
    print(f'mean error for sigma2_g: {np.mean((vg - sigma2_g*(scales**2)) / (sigma2_g*(scales**2))) * 100:.2f}%')
    print(f'mean error for sigma2_s: {np.mean((vs - sigma2_s*(scales**2)) / (sigma2_s*(scales**2))) * 100:.2f}%')
    print(f'mean error for sigma2_e: {np.mean((ve - sigma2_e*(scales**2)) / (sigma2_e*(scales**2))) * 100:.2f}%')



if __name__ == "__main__":
    simulate_all()
    # simulate_variance()
    # simulate_normalized_bayes()
    # simulate_W_normalization()
    # simulate_cheat_model()
    # simulate_sigma_estimation()

