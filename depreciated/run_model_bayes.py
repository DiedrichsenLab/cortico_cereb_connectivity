def get_fitted_models(model_dirs, model_names, config):
   elif config['model'].startswith('bayes'):
      # get list of subject for model
      model_subj = get_subj_list(config["subj_list"], config["dataset"])
      fitted_model = []
      train_info = []
      for d,m in zip(model_dirs,model_names):
         model_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',d)
         ext = '_' + m.split('_')[-1]
         fm,fi = calc_avrg_model(config['dataset'],d,ext,
                                 cerebellum=config['cerebellum'],
                                 model_subj=model_subj,
                                 avrg_mode=config['model'])
         fitted_model.append(fm)
         train_info.append(fi)

def calc_wopt_var(sub_weight_variance,
                  avrg_mode,
                  S):
   
   uncertainty = np.reciprocal(sub_weight_variance)
   wopt_variance_list = [np.nansum(uncertainty[s], axis=0) for s in range(S)]
   if 'vox' in avrg_mode:
      for wopt_var in wopt_variance_list:
         wopt_var[wopt_var == 0] = np.nan

   return wopt_variance_list


def calc_bayes_avrg(param_lists,
                    subject_list,
                    avrg_mode,
                    parameters=['coef_','coef_var']):
   # stack for numpy functions
   param_coef_ = np.stack(param_lists['coef_'], axis=0)
   
   # dimensions
   S = len(subject_list)
   n_vox, n_region = param_coef_[0].shape

   if 'loo' in avrg_mode:
      param_coef_var = np.stack(param_lists['coef_var'], axis=0)
      # calculate adjusted variance (vs: Sx(S-1))
      vg, vs = decompose_variance(param_coef_, np.nanmean(param_coef_var, axis=1)/n_region, model_type="loo")

      # reshape param_coef_var for loo
      idx = np.arange(24)[:, None]
      param_coef_var = param_coef_var[np.arange(24) != idx].reshape(S, S-1, n_vox)
      sub_var = vs[:, :, np.newaxis]*n_region + param_coef_var

      if not 'vox' in avrg_mode:
         sub_var = np.nanmean(sub_var, axis=-1)
         coef_norm = np.linalg.norm(param_coef_, axis=(1,2))[np.arange(24) != idx].reshape(S, S-1)
      else:
         coef_norm = np.linalg.norm(param_coef_, axis=2)[np.arange(24) != idx].reshape(S, S-1, n_vox)

      signal_norm2 = coef_norm**2 - n_vox*sub_var
      param_coef_ /= np.sqrt(signal_norm2).reshape(S, *([1]* (param_coef_.ndim - signal_norm2.ndim)))
      sub_var /= signal_norm2

      wopt_variance_list = calc_wopt_var(sub_weight_variance=sub_var,
                                         avrg_mode=avrg_mode,
                                         S=S)
      param_w_opt = {}
      if 'vox' in avrg_mode:
         # divide each weights by its variance
         P = [np.delete(param_coef_, s, axis=0) / sub_var[s, :, :, None] for s in range(S)]
         # sum over subjects and normalize
         param_w_opt['coef_'] = [np.nansum(P[s], axis=0) / wopt_variance_list[s][:, None] for s in range(S)]
      else:
         # divide each weights by its variance
         P = [np.delete(param_coef_, s, axis=0) / sub_var[s, :, None, None] for s in range(S)]
         # sum over subjects and normalize
         param_w_opt['coef_'] = [np.nansum(P[s], axis=0) / wopt_variance_list[s] for s in range(S)]

      param_w_opt['coef_'] = [np.nan_to_num(arr) for arr in param_w_opt['coef_']]
      param_w_opt['coef_var'] = wopt_variance_list
   elif 'half' in avrg_mode:
      param_coef_1 = np.stack(param_lists['coef_1'], axis=0)
      param_coef_2 = np.stack(param_lists['coef_2'], axis=0)
      vg, vs, vm = decompose_variance_half(np.stack((param_coef_1, param_coef_2), axis=1))
      sub_var = vs + vm/2

      coef_norm = np.linalg.norm(param_coef_, axis=(1,2))
      signal_norm2 = coef_norm**2 - n_vox*n_region*sub_var
      param_coef_ /= np.sqrt(signal_norm2).reshape(S, *([1]* (param_coef_.ndim - signal_norm2.ndim)))
      sub_var /= signal_norm2

      param_w_opt = {}
      wopt_variance = np.nan_to_num(np.nansum(1 / sub_var, axis=0))
      # divide each weights by its variance
      P = param_coef_ / sub_var[:, None, None]
      # sum over subjects and normalize
      param_w_opt['coef_'] = np.nan_to_num(np.nansum(P, axis=0) / wopt_variance)
      param_w_opt['coef_var'] = wopt_variance
   else:
      param_coef_var = np.stack(param_lists['coef_var'], axis=0)
      # calculate adjusted variance (vs: S)
      vg, vs = decompose_variance(param_coef_, np.nanmean(param_coef_var, axis=1)/n_region)
      sub_var = vs[:, np.newaxis]*n_region + param_coef_var

      if not 'vox' in avrg_mode:
         sub_var = np.nanmean(sub_var, axis=-1)
         coef_norm = np.linalg.norm(param_coef_, axis=(1,2))
      else:
         coef_norm = np.linalg.norm(param_coef_, axis=2)
   
      signal_norm2 = coef_norm**2 - n_vox*sub_var
      param_coef_ /= np.sqrt(signal_norm2).reshape(S, *([1]* (param_coef_.ndim - signal_norm2.ndim)))
      sub_var /= signal_norm2

      param_w_opt = {}
      wopt_variance = np.nan_to_num(np.nansum(1 / sub_var, axis=0))
      if 'vox' in avrg_mode:
         # divide each weights by its variance
         P = param_coef_ / sub_var[:, :, None]
         # sum over subjects and normalize
         param_w_opt['coef_'] = np.nansum(P, axis=0) / wopt_variance[:, None]
      else:
         # divide each weights by its variance
         P = param_coef_ / sub_var[:, None, None]
         # sum over subjects and normalize
         param_w_opt['coef_'] = np.nansum(P, axis=0) / wopt_variance
      param_w_opt['coef_'] = np.nan_to_num(param_w_opt['coef_'])
      param_w_opt['coef_var'] = wopt_variance

   return param_w_opt

def calc_avrg_model(avrg_mode, param_lists, subject_list, fitted_model):

   if avrg_mode.startswith('bayes') & ('half' not in avrg_mode):
      parameters = ['coef_', 'coef_var']
   elif avrg_mode.startswith('bayes') & ('half' in avrg_mode):
      parameters = ['coef_', 'coef_1', 'coef_2']
   elif avrg_mode=='avg-half':
      parameters = ['coef_', 'coef_1', 'coef_2']
      avrg_mode = 'avrg_sep'
   elif avrg_mode=='loo-half':
      parameters = ['coef_', 'coef_1', 'coef_2']
      avrg_mode = 'loo_sep'

      elif avrg_mode.startswith('bayes'):
      param_w_opt = calc_bayes_avrg(parameters=parameters,
                              param_lists=param_lists,
                              subject_list=subject_list,
                              avrg_mode=avrg_mode)
      if 'loo' in avrg_mode:
         avrg_model = []
         for s,sub in enumerate(subject_list):
            avrg_model.append(copy(fitted_model))
         for s,(coef,var) in enumerate(zip(param_w_opt['coef_'], param_w_opt['coef_var'])):
            setattr(avrg_model[s], 'coef_', coef)
            setattr(avrg_model[s], 'coef_var', var)
      else:
         avrg_model = fitted_model
         setattr(avrg_model, 'coef_', param_w_opt['coef_'])
         setattr(avrg_model, 'coef_var', param_w_opt['coef_var'])

   elif avrg_mode=='avg-half':
      for p in parameters:
         P = np.stack(param_lists[p],axis=0)
         setattr(avrg_model,p,P.mean(axis=0))
      setattr(avrg_model, 'coef_', (avrg_model.coef_1 + avrg_model.coef_2)/2)


def decompose_variance_half(data):
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


def decompose_variance_scaled_from_SS(
    covariance_matrix: np.ndarray,
    dataset_vec: np.ndarray,
    sub_vec: np.ndarray,
    part_vec: np.ndarray,
    single_scaling: bool = False
) -> pd.DataFrame:
    """
    Decomposes variance components from a covariance matrix.
    Args:
        covariance_matrix (np.ndarray): A square covariance matrix.
        dataset_vec (np.ndarray): A vector containing dataset names for each row/column of the covariance matrix.
        sub_vec (np.ndarray): A vector containing subject IDs for each row/column of the covariance matrix.
        part_vec (np.ndarray): A vector containing partition IDs for each row/column of the covariance matrix.
        single_scaling (bool): If True, assumes a single scale factor for all subjects. Defaults to False.
    Returns:
        Q_df (pandas.DataFrame): DataFrame containing variance components:
            - train_dataset: Dataset names.
            - subj_id: Subject IDs.
            - sc: Scale factors for each subject.
            - v_u: Universal variance component.
            - v_d: Dataset variance component (dataset-specific).
            - v_s: Subject variance component (dataset-specific).
            - v_m: Measurement noise variance component (subject-specific).
    """

    N_SS = covariance_matrix.shape[0]

    # Identify unique subjects, datasets, and partitions
    subjects = [(dataset_vec[i], sub_vec[i]) for i in range(N_SS)]
    unique_subjects = list(dict.fromkeys(subjects))
    N_subj = len(unique_subjects)

    unique_datasets = list(dict.fromkeys(dataset_vec))
    N_datasets = len(unique_datasets)

    N_part = len(np.unique(part_vec))

    # ------------------------------
    # ------- Ckeck inputs ---------
    # ------------------------------
    if covariance_matrix.size == 0:
        raise ValueError("covariance_matrix cannot be empty.")

    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("The covariance_matrix must be a square 2D array.")

    if len(dataset_vec) != N_SS or len(sub_vec) != N_SS or len(part_vec) != N_SS:
        raise ValueError("Input vectors (dataset_vec, sub_vec, part_vec) must have the same length as the covariance matrix dimensions.")

    if N_part == 1:
        print(
            "The number of unique parts is 1. Subject variance (v_s) and measurement noise variance (v_m) cannot be distinguished. "
            "Returning v_i as v_s + v_m."
        )

    if N_datasets == 1:
        print(
            "The number of unique datasets is 1. Universal Variance (v_u) cannot be estimated. "
            "Returning v_g as v_u + v_d."
        )


    # Map (dataset, sub_id) to index
    subject_map = {sid: idx for idx, sid in enumerate(unique_subjects)}

    # ---------------------------------------
    # ----- Compute pairs and bad pairs -----
    # ---------------------------------------
    pairs_1 = []
    pairs_2 = []
    pairs_3 = []
    pairs_4 = []
    bad_pair_1 = 0
    bad_pair_2 = 0
    bad_pair_3 = 0
    bad_pair_4 = 0
    for i in range(N_SS):
        for k in range(i, N_SS):
            # cross-dataset pairs
            if dataset_vec[i] != dataset_vec[k]:
                if covariance_matrix[i, k] <= 0:
                    bad_pair_1 += 1
                    continue
                pairs_1.append((i, k))

            # same-dataset
            else:
                # cross-subject pairs
                if (sub_vec[i] != sub_vec[k]):
                    if covariance_matrix[i, k] <= 0:
                        bad_pair_2 += 1
                        continue
                    pairs_2.append((i, k))

                # same-subject
                else:
                    # cross-partition pairs
                    if (part_vec[i] != part_vec[k]):
                        if covariance_matrix[i, k] <= 0:
                            bad_pair_3 += 1
                            continue
                        pairs_3.append((i, k))

                    # same-partition pairs
                    else:
                        if covariance_matrix[i, k] <= 0:
                            bad_pair_4 += 1
                            continue
                        pairs_4.append((i, k))

    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)
    pairs_3 = np.array(pairs_3)
    pairs_4 = np.array(pairs_4)
    M_1 = len(pairs_1)
    M_2 = len(pairs_2)
    M_3 = len(pairs_3)
    M_4 = len(pairs_4)
    M = M_1 + M_2 + M_3 + M_4

    if N_datasets != 1:
        print(f"Bad pairs (cross-dataset): {bad_pair_1 / (M_1 + bad_pair_1) * 100:.2f}%")
    print(f"Bad pairs (cross-subject): {bad_pair_2 / (M_2 + bad_pair_2) * 100:.2f}%")
    if N_part != 1:
        print(f"Bad pairs (cross-partition): {bad_pair_3 / (M_3 + bad_pair_3) * 100:.2f}%")
    print(f"Bad pairs (same-partition): {bad_pair_4 / (M_4 + bad_pair_4) * 100:.2f}%")


    # -----------------------------------------------
    # ----- Construct A and y for least squares -----
    # -----------------------------------------------
    if single_scaling:
       N_scale = 1
    else:
       N_scale = N_subj
    if N_part == 1:
       A = np.zeros((M, N_scale + N_datasets + N_subj))
    else:
       A = np.zeros((M, N_scale + N_datasets + N_datasets + N_subj))
    y = np.zeros(M)

    # cross-dataset pairs
    for m, (i, k) in enumerate(pairs_1):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        s_k = subject_map[(dataset_vec[k], sub_vec[k])] if not single_scaling else 0
        # Set 1s for s_i, s_k, v_u
        A[m, s_i] += 1
        A[m, s_k] += 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])

    # same-dataset, cross-subject pairs
    for m, (i, k) in enumerate(pairs_2, start=M_1):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        s_k = subject_map[(dataset_vec[k], sub_vec[k])] if not single_scaling else 0
        # Set 1s for s_i, s_k
        A[m, s_i] += 1
        A[m, s_k] += 1
        # Set 1s for v_u + v_d
        d = unique_datasets.index(dataset_vec[i])
        A[m, N_scale+d] = 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])

    # same-dataset, same-subject, cross-partition pairs
    for m, (i, k) in enumerate(pairs_3, start=M_1 + M_2):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        # Set 1s for s_i, s_k
        A[m, s_i] = 2
        # Set 1s for v_u + v_d + v_s
        d = unique_datasets.index(dataset_vec[i])
        A[m, N_scale+N_datasets+d] = 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])

    # same-dataset, same-subject, same-partition pairs
    for m, (i, k) in enumerate(pairs_4, start=M_1 + M_2 + M_3):
        # Get subject IDs
        s_i = subject_map[(dataset_vec[i], sub_vec[i])] if not single_scaling else 0
        # Set 1s for s_i, s_k
        A[m, s_i] = 2
        # Set 1s for v_u + v_d + v_s + v_m
        A[m, -(N_subj-subject_map[(dataset_vec[i], sub_vec[i])])] = 1
        # Set y_m = ln(A_{i,k})
        y[m] = np.log(covariance_matrix[i, k])


    # -------------------------------------------------------
    # ----- Solve least squares and extract components ------
    # -------------------------------------------------------
    x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # Extract parameters
    sc = np.exp(x[:N_scale])                                          # scales
    type_1 = np.exp(x[N_scale:N_scale+N_datasets])                    # V_u + V_d
    type_2 = np.exp(x[N_scale+N_datasets:N_scale+N_part*N_datasets])  # V_u + V_d + V_s
    type_3 = np.exp(x[N_scale+N_part*N_datasets:])                    # V_u + V_d + V_s + V_m

    if N_datasets == 1:
        v_g = type_1
    else:
        v_u = 1
        v_d = type_1 - v_u
    if N_part == 1:
        v_i = type_3 - type_1[[unique_datasets.index(ds) for ds,_ in unique_subjects]]
    else:
        v_s = type_2 - type_1
        v_m = type_3 - type_2[[unique_datasets.index(ds) for ds,_ in unique_subjects]]


    # ----------------------------------
    # -------- Create DataFrame --------
    # ----------------------------------
    train_dataset = [sid[0] for sid in subject_map.keys()]
    subj_id = [sid[1] for sid in subject_map.keys()]

    if single_scaling:
        sc = [sc[0]] * len(train_dataset)

    data_dict = {
        'train_dataset': train_dataset,
        'subj_id': subj_id,
        'sc': sc
    }
    if N_datasets == 1:
        data_dict['v_g'] = v_g[[unique_datasets.index(ds) for ds, _ in unique_subjects]]
    else:
        data_dict['v_u'] = [v_u] * len(train_dataset)
        data_dict['v_d'] = v_d[[unique_datasets.index(ds) for ds, _ in unique_subjects]]
    if N_part == 1:
        data_dict['v_i'] = v_i
    else:
        data_dict['v_s'] = v_s[[unique_datasets.index(ds) for ds, _ in unique_subjects]]
        data_dict['v_m'] = v_m

    Q_df = pd.DataFrame(data_dict)

    return Q_df


def decompose_variance(data, vm_hat, model_type=None):
   """ Decomposes variance of group, subject, and measurement noise.
      This is an upgraded version to handle subject-specific scaling.
      With the vm_hat already estimated, there is no need for different observations.
   Args:
      data (ndarray (n_sub, n_A, n_B)): the data to decompose
      vm_hat (ndarray (n_sub)): estimated variance of measurement noise of subjects
      model_type (str): either None or 'loo':
         if 'loo': the output will be stretched by subject size
   Returns:
      vg (ndarray (n_sub,)): group variance scaled for each subject
      vs (ndarray (n_sub,)): subject variance scaled for each subject
      vm (ndarray (n_sun,)): measurement noise variance scaled for each subject
      if model_type is 'loo': outputs will be (n_sub, n_sub-1) shape
   """

   n_sub, n_A, n_B = data.shape
   n_features = n_A * n_B
   data = data.reshape((n_sub, n_features))

   product_matrices = np.einsum('sf,kf->sk', data, data) / n_features   # Shape: (n_sub, n_sub)

   if model_type == 'loo':
      n_sub_loo = n_sub - 1
      vg = np.zeros((n_sub, n_sub - 1))
      vs = np.zeros((n_sub, n_sub - 1))
      for s in range(n_sub):
         product_matrices_loo = np.delete(np.delete(product_matrices, s, axis=0), s, axis=1)

         # Masks
         mask_self_sub = np.eye(n_sub_loo, dtype=bool) # Shape: (n_sub, n_sub)
         
         # Cross-subject (type 1)
         SS_1 = np.where(mask_self_sub, 0, product_matrices_loo)   # Set self-pairs to 0

         # Within-subject, same reps (type 3)
         SS_3 = np.diag(product_matrices_loo)   # Set other-pairs to 0

         SS_2 = SS_3 - np.delete(vm_hat, s, axis=0)

         vg[s] = np.nansum(np.sqrt(SS_2[:, None] / SS_2) * SS_1, axis=1) / (n_sub_loo-1)    # Shape: (n_sub)
         vs[s] = SS_2 - vg[s]
   elif model_type is None:
      # Masks
      mask_self_sub = np.eye(n_sub, dtype=bool) # Shape: (n_sub, n_sub)
      
      # Cross-subject (type 1)
      SS_1 = np.where(mask_self_sub, 0, product_matrices)   # Set self-pairs to 0

      # Within-subject, same reps (type 3)
      SS_3 = np.diag(product_matrices)   # Set other-pairs to 0

      SS_2 = SS_3 - vm_hat

      vg = np.nansum(np.sqrt(SS_2[:, None] / SS_2) * SS_1, axis=1) / (n_sub-1)    # Shape: (n_sub)
      vs = SS_2 - vg
   else:
      raise ValueError("model_type should be 'loo' or not given")
   return vg, vs

