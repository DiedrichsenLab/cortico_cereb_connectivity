# split-half_simulation.py
# A simple module for split-half simulation in Bayesian analysis

import numpy as np
from cortico_cereb_connectivity.bayes_simulation.simulation_functions import *
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    S = 1
    N = 29
    Q = 1500
    P = 3000
    n_simulations = 50
    ratio_list = []

    for n in range(n_simulations):
        Q = int(np.random.uniform(500, 3500))
        la_range = np.arange(5, 11, step=np.log(2)/2)

        # Generate data
        _, X = create_dataset(0, 1, 0, (S, N, Q), same_half=True)
        # X = normalize_dataset(X, 'parcel')
        X1 = X[:N, :]
        X1 = normalize_dataset(X1, 'parcel')
        X2 = X[N:, :]
        X2 = normalize_dataset(X2, 'parcel')

        _, W = create_dataset(0, 1e-5, 0, (S, Q, P))

        sigma2epss = generate_sigma2eps(np.array([0.15]), np.array([0.15]), (S, P))
        Y_train, _ = generate_Y(X.reshape((1, 2*N, Q)), W.reshape((1, Q, P)), sigma2epss, (S, 2*N, P))
        Y_train = Y_train[0]
        Y_train1 = Y_train[:N, :]
        Y_train2 = Y_train[N:, :]

        Y_eval, _ = generate_Y(X.reshape((1, 2*N, Q)), W.reshape((1, Q, P)), sigma2epss, (S, 2*N, P))
        Y_eval = Y_eval[0]
        Y_eval1 = Y_eval[:N, :]
        Y_eval2 = Y_eval[N:, :]

        # R, _ = ev.calculate_R(Y_train.flatten(), Y_eval.flatten())
        # print(f"Y_R: {R}")

        performance = pd.DataFrame()
        for i, la in enumerate(la_range):
            # whole dataset
            W_hat = np.linalg.inv(X.T @ X + np.exp(la)*np.identity(Q)) @ X.T @ Y_train
            Y_pred = X @ W_hat
            R, _ = ev.calculate_R(Y_pred.flatten(), Y_eval.flatten())
            performance = pd.concat([performance, pd.DataFrame([{'type': 'Y', 'logalpha': la, 'corr': R}])], ignore_index=True)
            # R, _ = ev.calculate_R(W_hat.flatten(), W.flatten())
            # performance = pd.concat([performance, pd.DataFrame([{'type': 'W', 'logalpha': la, 'corr': R}])], ignore_index=True)

            # split dataset
            W_hat1 = np.linalg.inv(X1.T @ X1 + np.exp(la)*np.identity(Q)) @ X1.T @ Y_train1
            # Y_pred1 = X1 @ W_hat1
            # R1, _ = ev.calculate_R(Y_pred1.flatten(), Y_eval1.flatten())
            # performance = pd.concat([performance, pd.DataFrame([{'type': 'Y1', 'logalpha': la, 'corr': R1}])], ignore_index=True)

            W_hat2 = np.linalg.inv(X2.T @ X2 + np.exp(la)*np.identity(Q)) @ X2.T @ Y_train2
            # Y_pred2 = X2 @ W_hat2
            # R2, _ = ev.calculate_R(Y_pred2.flatten(), Y_eval2.flatten())
            # performance = pd.concat([performance, pd.DataFrame([{'type': 'Y2', 'logalpha': la, 'corr': R2}])], ignore_index=True)

            W_hat_avg = (W_hat1 + W_hat2) / 2
            Y_pred_avg = X @ W_hat_avg
            R_avg, _ = ev.calculate_R(Y_pred_avg.flatten(), Y_eval.flatten())
            performance = pd.concat([performance, pd.DataFrame([{'type': 'Y_avg', 'logalpha': la, 'corr': R_avg}])], ignore_index=True)

        # Find the best logalpha where the corr mean is maximum for Y and Y_avg
        best_logalpha_Y = performance[performance['type'] == 'Y'].groupby('logalpha')['corr'].mean().idxmax()
        best_logalpha_Y_avg = performance[performance['type'] == 'Y_avg'].groupby('logalpha')['corr'].mean().idxmax()
        ratio = np.exp(best_logalpha_Y)/np.exp(best_logalpha_Y_avg)
        ratio_list.append(ratio)
        print(f"Q: {Q} -----------------:")
        print(f"Best logalpha ratio: {best_logalpha_Y} - {best_logalpha_Y_avg} : {ratio:.3f}\n")

    print(f"-----------------")
    print(f"Best logalpha ratio: {np.mean(ratio):.3f}")
    # fig, ax1 = plt.subplots(3,2, figsize=(10, 8), sharex=True)

    # plt.subplot(3,2,1)
    # sns.lineplot(data=performance[performance['type']=='W'], x='logalpha', y='corr', label='W')
    # plt.subplot(3,2,3)
    # sns.lineplot(data=performance[performance['type']=='Y'], x='logalpha', y='corr', label='Y')
    # plt.subplot(3,2,5)
    # sns.lineplot(data=performance[performance['type']=='Y_avg'], x='logalpha', y='corr', label='Y_mean')
    # plt.subplot(3,2,2)
    # sns.lineplot(data=performance[performance['type']=='Y1'], x='logalpha', y='corr', label='Y1')
    # plt.subplot(3,2,4)
    # sns.lineplot(data=performance[performance['type']=='Y2'], x='logalpha', y='corr', label='Y2')

    # plt.xlabel('alpha')
    # plt.show()
