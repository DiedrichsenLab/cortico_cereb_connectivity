import numpy as np
import pandas as pd
import Functional_Fusion.reliability as frel # from functional fusion module
import Functional_Fusion.dataset as fdata
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.globals as gl


def new_var_decom(dataset_name='MDTB', logalpha=6):
    # Load the product matrix and metadata vectors 
    data = np.load("/home/UWO/ashahb7/Github/bayes_temp/L2reghalf_cov_matrix.npz", allow_pickle=True)
    indices = []
    indices.append(np.where((data['dataset_vec'] == dataset_name) & (data['logalpha_vec'] == logalpha))[0])
    indices = np.concatenate(indices)
    product_la = data['product_matrix'][np.ix_(indices, indices)]
    dataset_vec_la = data['dataset_vec'][indices]
    sub_vec_la = data['sub_vec'][indices]
    part_vec_la = data['part_vec'][indices]

    subset_indices = np.where(dataset_vec_la == dataset_name)[0]
    product_subset = product_la[np.ix_(subset_indices, subset_indices)]
    dataset_subset = dataset_vec_la[subset_indices]
    sub_subset = sub_vec_la[subset_indices]
    part_subset = part_vec_la[subset_indices]

    # Solve
    ds_var_decom_df = rm.decompose_variance_scaled_from_SS(product_subset, dataset_subset, sub_subset, part_subset, single_scaling=True)

    return ds_var_decom_df

def rel_var_decom(dataset_name='MDTB', logalpha=6):
    # Load data for reliability function
    model_base_path = gl.conn_dir + f"/MNISymC3/train/"
    data = []
    sub_list = fdata.get_dataset_class(gl.base_dir, dataset=dataset_name).get_participants().participant_id
    for sub_id in sub_list:
        print(f"load subject {sub_id} for dataset {dataset_name}")
        mname_base = f"{dataset_name}_all_Icosahedron1002_L2reghalf"
        mname = mname_base + f"/{mname_base}_A{logalpha}_{sub_id}"

        mo, _ = cio.load_model(model_base_path+mname)

        sub_data_list = []
        sub_data_list.append(mo.coef_1.T)
        sub_data_list.append(mo.coef_2.T)
        data.append(np.concatenate(sub_data_list, axis=0))

    cond_vec = np.arange(mo.coef_1.T.shape[0])
    cond_vec = np.concatenate([cond_vec, cond_vec])

    part_vec = np.array([1] * mo.coef_1.T.shape[0] + [2] * mo.coef_2.T.shape[0])

    # Run the reliability function
    vars = frel.decompose_subj_group(np.array(data), cond_vec, part_vec,
                         separate='none',
                         subtract_mean=False)

    return vars


if __name__ == "__main__":
    # Run the test function on datasets
    dataset_name = 'MDTB'
    logalpha = 6

    ds_var_decom_df = new_var_decom(dataset_name, logalpha)
    vars = rel_var_decom(dataset_name, logalpha)[0]
    new_vars = [
        (ds_var_decom_df['sc']**2 * ds_var_decom_df['v_g']).mean(),
        (ds_var_decom_df['sc']**2 * ds_var_decom_df['v_s']).mean(),
        (ds_var_decom_df['sc']**2 * ds_var_decom_df['v_m']).mean()
    ]
    
    print(f'\nNew Variance decomposition components: {[f"{v:.2e}" for v in new_vars]}')
    print(f'Reliability Variance components: {[f"{v:.2e}" for v in vars]}')

    correlation = np.corrcoef(vars, new_vars)[0, 1]
    print(f'Correlation: {correlation}')

    new_vars_percentage = [v / sum(new_vars) * 100 for v in new_vars]
    vars_percentage = [v / sum(vars) * 100 for v in vars]
    print(f'\nNew Variance decomposition percentages: {[f"{v:.2f}%" for v in new_vars_percentage]}')
    print(f'Reliability Variance percentages: {[f"{v:.2f}%" for v in vars_percentage]}')