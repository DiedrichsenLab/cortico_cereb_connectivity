"""
script for calculating the covariance between each pair of models
@ Ali Shahbazi
"""


import os
import numpy as np
import pandas as pd
from collections import defaultdict
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.cio as cio
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.cio as cio
import Functional_Fusion.dataset as fdata


def compute_product_matrix(model_info, batch_size, model_base_path):
   """
   Compute the interaction matrix and metadata vectors for model coefficients.
   
   Args:
      model_info (list): List of tuples (dataset_id, subject_id, model_path).
                        dataset_id (int/str): Identifier for dataset.
                        subject_id (int/str): Identifier for subject.
                        logalpha (int): Logalpha value.
                        model_path (str/Path): Path to model file.
      batch_size (int): Number of models to load at once.
      model_base_path (str/Path): Base directory for model files.
      
   Returns:
      product_matrix (np.ndarray): (total_n_sub*2, total_n_sub*2) matrix of X.T @ Y.
      dataset_vec (np.ndarray): Array of dataset IDs for each coefficient.
      sub_vec (np.ndarray): Array of subject IDs for each coefficient.
      logalpha_vec (np.ndarray): Array of logalpha values for each coefficient.
      part_vec (np.ndarray): Array of partition IDs ('coef_1' or 'coef_2').
   """
   # Total number of models
   total_n_models = len(model_info)
   total_n_coef = total_n_models * 2  # Each subject has two coefficients and 6 logalpha values
   
   # Initialize the interaction matrix
   product_matrix = np.zeros((total_n_coef, total_n_coef), dtype=np.float64)
   
   # Initialize metadata vectors
   dataset_vec = np.zeros(total_n_coef, dtype=object)
   sub_vec = np.zeros(total_n_coef, dtype=object)
   logalpha_vec = np.zeros(total_n_coef, dtype=int)
   part_vec = np.zeros(total_n_coef, dtype=object)
   
   # Populate metadata vectors
   for idx, (dataset_id, subject_id, logalpha, _) in enumerate(model_info):
      base_idx = idx * 2
      dataset_vec[base_idx:base_idx+2] = dataset_id
      sub_vec[base_idx:base_idx+2] = subject_id
      logalpha_vec[base_idx:base_idx+2] = logalpha
      part_vec[base_idx] = 'coef_1'
      part_vec[base_idx+1] = 'coef_2'

   # Process models in batches
   for batch_start in range(0, total_n_models, batch_size):
      batch_end = min(batch_start + batch_size, total_n_models)
      batch_indices = range(batch_start, batch_end)
      
      # Load current batch
      batch_coefs = []
      for idx in batch_indices:
         _, _, _, rel_path = model_info[idx]
         model_path = model_base_path + f"{rel_path}"
         model, _ = cio.load_model(model_path)
         coef_1 = model.coef_1.flatten()  # Vectorize
         coef_2 = model.coef_2.flatten()  # Vectorize
         batch_coefs.append(coef_1)
         batch_coefs.append(coef_2)
        
      # Convert coefficients to array for efficient computation
      batch_coefs = np.array(batch_coefs)  # Shape: (batch_size*2, coef_length)
      
      # Compute interactions within the batch (i vs i)
      size_coef = len(coef_1)
      batch_matrix = np.dot(batch_coefs, batch_coefs.T) / size_coef  # Shape: (batch_size*2, batch_size*2)
      global_start = batch_start * 2
      global_end = batch_end * 2
      product_matrix[global_start:global_end, global_start:global_end] = batch_matrix
      
      # Compute interactions with all previous batches (i vs j where j < i)
      for prev_batch_start in range(0, batch_start, batch_size):
         prev_batch_end = min(prev_batch_start + batch_size, total_n_models)
         prev_batch_indices = range(prev_batch_start, prev_batch_end)
         
         # Load previous batch
         prev_batch_coefs = []
         for idx in prev_batch_indices:
            _, _, _, rel_path = model_info[idx]
            model_path = model_base_path + f"{rel_path}"
            model, _ = cio.load_model(model_path)
            coef_1 = model.coef_1.flatten()
            coef_2 = model.coef_2.flatten()
            prev_batch_coefs.append(coef_1)
            prev_batch_coefs.append(coef_2)
                        
         prev_batch_coefs = np.array(prev_batch_coefs)
         
         # Compute cross-batch interactions
         cross_matrix = np.dot(batch_coefs, prev_batch_coefs.T) / size_coef  # Shape: (batch_size*2, prev_batch_size*2)
         prev_global_start = prev_batch_start * 2
         prev_global_end = prev_batch_end * 2
         product_matrix[global_start:global_end, prev_global_start:prev_global_end] = cross_matrix
         product_matrix[prev_global_start:prev_global_end, global_start:global_end] = cross_matrix.T
    
   return product_matrix, dataset_vec, sub_vec, logalpha_vec, part_vec
      

if __name__ == "__main__":
   # Example model_info: list of (dataset_id, subject_id, logalpha, model_path)
   dataset_list = ["MDTB", "Language", "WMFS", "Demand", "Somatotopic", "Nishimoto"]
   logalpha_list = [2, 4, 6, 8, 10, 12]
   model_info = []

   # Fill the metadata list with dataset, subject, logalpha, and model path
   for dataset in dataset_list:
      sub_list = fdata.get_dataset_class(gl.base_dir, dataset=dataset).get_participants().participant_id
      for sub_id in sub_list:
         for la in logalpha_list:
               if dataset == "Language":
                  mname_base = f"{dataset}_ses-localizer_cond_Icosahedron1002_L2reghalf"
               else:
                  mname_base = f"{dataset}_all_Icosahedron1002_L2reghalf"
               mname_base = mname_base + f"/{mname_base}_A{la}_{sub_id}"
               model_info.append((dataset, sub_id, la, mname_base))

   batch_size = 30
   model_base_path = gl.conn_dir + "/MNISymC3/train/"

   product_matrix, dataset_vec, sub_vec, logalpha_vec, part_vec = compute_product_matrix(model_info,
                                                                                         batch_size,
                                                                                         model_base_path)

   # Save the product matrix and metadata vectors for future variance decomposition use
   np.savez(
      "/home/UWO/ashahb7/Github/bayes_temp/product_matrix.npz",
      product_matrix=product_matrix,
      dataset_vec=dataset_vec,
      sub_vec=sub_vec,
      logalpha_vec=logalpha_vec,
      part_vec=part_vec
   )
         
