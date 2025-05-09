# split-half_simulation.py
# A simple module for split-half simulation in Bayesian analysis

import numpy as np
from cortico_cereb_connectivity.bayes_simulation.simulation_functions import *
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_ridge_shape(correlation_strength=None):
    """
    Generate ridge for shape simulation
    """
    S = 1
    N = 29
    Q = 50
    P = 150
    alpha_1 = [4, 6]
    alpha_2 = [4, 6]

    _, X = create_dataset(0, 1, 0, (S, N, Q))

    _, W1 = create_dataset(0, 1, 0, (S, Q, P))
    if correlation_strength is not None:
        W2 = correlation_strength * W1 + np.sqrt(1 - correlation_strength**2) * np.random.normal(0, 1, W1.shape)
    else:
        _, W2 = create_dataset(0, 1, 0, (S, Q, P))

    sigma2epss = generate_sigma2eps(np.array([0]), np.array([0]), (S, P))
    Y1, _ = generate_Y(X.reshape((1, N, Q)), W1.reshape((1, Q, P)), sigma2epss, (S, N, P))
    Y1 = Y1[0]
    Y2, _ = generate_Y(X.reshape((1, N, Q)), W2.reshape((1, Q, P)), sigma2epss, (S, N, P))
    Y2 = Y2[0]

    plt.subplots(1,2, figsize=(12, 6), sharey=True)
    for i, (la1, la2) in enumerate(zip(alpha_1, alpha_2)):
        a1 = np.exp(la1)
        a2 = np.exp(la2)
        W_hat1 = np.linalg.inv(X.T @ X + a1 * np.identity(Q)) @ (X.T @ Y1)
        W_hat2 = np.linalg.inv(X.T @ X + a2 * np.identity(Q)) @ (X.T @ Y2)

        plt.subplot(1,2,i+1)
        sns.scatterplot(x=W1.flatten(), y=W2.flatten(), label='True W')
        sns.scatterplot(x=W_hat1.flatten(), y=W_hat2.flatten(), label='Estimated W', marker='x')
        plt.legend()
        plt.xlabel('W1')
        plt.ylabel('W2')
        plt.title(r'$\log(\alpha_1)$ = {}, $\quad$ $\log(\alpha_2)$ = {}'.format(la1, la2))
        plt.axis('square')
    plt.show()


if __name__ == "__main__":
    simulate_ridge_shape(correlation_strength=0.3)
