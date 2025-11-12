import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cortico_cereb_connectivity.model as cmo
import cortico_cereb_connectivity.evaluation as cev
import cortico_cereb_connectivity.run_model as crm


def generate_random_data(N, Q, P):
    X = np.random.normal(0, 1, (2*N, Q))
    Y = np.random.normal(0, 1, (2*N, P))
    X = crm.std_data(X, 'parcel')
    Y = crm.std_data(Y, 'global')
    return X, Y

def run_one_simulation(N=45, Q=180, P=550, la_list=None):
    if la_list is None:
        la_list = [0, 2, 4, 6, 8, 10]
    X, Y = generate_random_data(N, Q, P)

    results = []
    # Ridge loop
    for la in la_list:
        alpha = np.exp(la)
        ridge_model = getattr(cmo, "L2reg")(alpha)
        ridge_model.fit(X, Y, None)
        R2, _ = cev.calculate_R2(Y, ridge_model.predict(X))
        nnz = np.round(np.count_nonzero(ridge_model.coef_)/P, 1)
        results.append(dict(model='ridge', la=la, R2=R2, nonzero=nnz))
        print(f"Ridge alpha={alpha:.2e}, R2={R2:.4f}, nonzero={nnz}")

    # NNLS
    nnls_model = getattr(cmo, "NNLS")()
    nnls_model.fit(X, Y)
    R2, _ = cev.calculate_R2(Y, nnls_model.predict(X))
    nnz = np.round(np.count_nonzero(nnls_model.coef_)/P, 1)
    results.append(dict(model='nnls', la=None, R2=R2, nonzero=nnz))
    print(f"NNLS R2={R2:.4f}, nonzero={nnz}")

    return results

def run_multiple_simulations(N=45, Q=180, P=550, la_list=[0, 2, 4, 6, 7], n_runs=10):
    """Repeat the random experiment and average results."""
    all_results = []
    for i in range(n_runs):
        all_results.extend(run_one_simulation(N, Q, P, la_list))
    df = pd.DataFrame(all_results)
    return df

if __name__ == "__main__":
    n_runs = 1
    Q = 900
    P = 2750
    N = 225
    la_list = [0, 2, 4, 6, 8]
    # ext = "_nostd"
    ext = ""

    df_summary = run_multiple_simulations(N, Q, P, la_list, n_runs)
    out_file = Path(f"/home/UWO/ashahb7/Github/cortico_cereb_connectivity/results/simulation_results_Q{Q}P{P}N{N}{ext}.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_file, index=False, sep='\t')

    ridge_df = df_summary[df_summary['model']=='ridge']
    # Plot results
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=ridge_df, x='la', y='R2', marker='o', label='Ridge', errorbar='ci')
    nnls_mean = df_summary[df_summary['model']=='nnls']['R2'].mean()
    plt.axhline(nnls_mean, color='orange', linestyle='--', label='NNLS')
    nnls_vals = df_summary[df_summary['model'] == 'nnls']['R2']
    sem = nnls_vals.std(ddof=1) / np.sqrt(n_runs)
    z = 1.96  # ~95% CI
    ci_low = nnls_mean - z * sem
    ci_high = nnls_mean + z * sem
    plt.axhspan(ci_low, ci_high, color='orange', alpha=0.2)
    plt.xlabel('log(alpha)')
    plt.ylabel('R² on training data')
    plt.title(f'Q={Q}, P={P}, N={N}, runs={n_runs}')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file.parent / f"simulation_performance_Q{Q}P{P}N{N}{ext}.png")
    plt.show()