# Cortico_Cereb_Connectivity
This repository includes the code to train and test cortico-cerebellar-connectivity models, as first described in King et al. (2023).
The original code underlying that paper is maintained in a different repository (https://github.com/maedbhk/cerebellum_connectivity).
This repository is a cleaned-up version of the original code, that also uses the FunctionalFusion framework (https://github.com/DiedrichsenLab/Functional_Fusion) to integrate knowledge across different datasets. The repository includes new group-based connectivity model reported in Nettekoven et al. (2024), and are being used in Shahshahani et al. (2024).

### Dependencies
The code requires the following packages:
- numpy
- sklearn
- nibabel
- FunctionalFusion (for easy mapping of atlas spaces)

### Models
The data-directory contains the connectivity weights in form of pdconn-cifti files. 
The neo-cortex is parcellated in a specific way. 
The cerebellum is voxel wise

A example of how to use these models to make predictions about new data is given in the `notebooks/0.application_example.ipynb` notebooks.

The models from Nettekoven et al. (2024):
- `Nettekoven_2024_MDTB_L2`: L2 model (alpha = exp(8)), SUIT3 space, Icosahedron1002, full MDTB dataset
- `Nettekoven_2024_Demand_L2`: L2 model (alpha = exp(8)), SUIT3 space, Icosahedron1002, full Multiple-demand dataset
- `Nettekoven_2024_HCP_L2`: L2 model (alpha = exp(-2)), SUIT3 space, Icosahedron1002, resting state from 100 unrelated HCP subjects
- `Nettekoven_2024_IBC_L2`: L2 model (alpha = exp(6)), SUIT3 space, Icosahedron1002, full IBC dataset
- `Nettekoven_2024_Nishimoto_L2`: L2 model (alpha = exp(10)), SUIT3 space, Icosahedron1002, full Nishimoto dataset
- `Nettekoven_2024_Somatotopic_L2`: L2 model (alpha = exp(8)), SUIT3 space, Icosahedron1002, full Somatotopic dataset
- `Nettekoven_2024_Fusion_L2`: SUIT3 space, Icosahedron1002, Average model from above (excluding HCP)

The models used in Shahshahani et al. (2024). The first two models are very close to the models reported in King et al. (2023), but are trained in SUIT3 space. 
- `Shahshahani_2024_MDTB_L2`: L2 model (alpha = 8), SUIT3 space, Icosahedron1002, Task set A from MDTB dataset
- `Shahshahani_2024_MDTB_L1`: L1 model (alpha = -5), SUIT3 space, Icosahedron1002, Task set A from MDTB dataset
- `Shahshahani_2024_Fusion_L2`: SUIT3 space, Icosahedron1002, Average model from L2 models above (excluding WMFS and HCP)

### References

King, M., Shahshahani, L., Ivry, R. B., & Diedrichsen, J. (2023). A task-general connectivity model reveals variation in convergence of cortical inputs to functional regions of the cerebellum. eLife, 12. https://doi.org/10.7554/eLife.81511

Nettekoven, C., Zhi, D., Ladan, S., Pinho, A. L., Saadon Grosmannn, N., Buckner, R., & Diedrichsen, J. (2023). A hierarchical atlas of the human cerebellum for functional precision mapping. BioRviv.

Shahshahani, L., King, M., Nettekoven, C., Ivry, R., & Diedrichsen, J. (2023). Selective recruitment: Evidence for task-dependent gating of inputs to the cerebellum. bioRxiv, 2023.01.25.525395.

### Details on Fusion model in Nettekoven et al. (2024)
All models are in `SUIT3` space.

Models were trained evaluated ```ccc.run_model```, which is called from ```ccc.scripts.script_train_eval_models.py```

Models are then fused (i.e. simply averaged) using ```ccc.scripts.script_fuse_models.py```

* Model 4: Demand, HCP and MDTB 
* Model 5: all datasets including HCP
* Model 6: all datasets excluding HCP
* Model 7: all datasets excluding HCP and Somatotopic

The final model evaluation results reported in the paper can be found in ```ccc/notebooks/2.Nettekoven_2024/6.Evaluate_model_int.ipynb```.

To summarize the connectivity pattern by cerebellar regions:

```
import cortico_cereb_connectivity.scripts.script_summarize_weights as csw
csw.make_weight_map('Fusion','06',method='L2Regression')
```

To summarize further by cortical ROI:
```T = csw.make_weight_table(dataset="Fusion",extension="06",cortical_roi="")```

Summary figures (by MSHBM_Prior_15_fsLR32)
```notebooks/cortical_connectivity.ipynb```

Full connectivity maps:
```notebooks/connectivity_weights.ipynb``` (Fig S5 & Fig S6)

### Details for Shahbazi et al. (in preparation)
New models are all in the `MNISymC3` space. 

More detailed exploration for connectivity models with the following non-backwards compatible changes: 

* Scaling is performed for the datasets and not learned in the model anymore. Scaling is now performed in the `run_model.get_cortical_data` and `get_cerebellar_data` functions. The old way is still in `L2regression`, but the new way is now in `L2reg`. 
* The datasets used and the scaling and whether rest is added is determined in `globals.py`. 
* Single dataset modes are trained on inidividual subjects, averaged across subejcts (avg), trained on the group data (group) and trianed the the group data with subject left our (group_loo). 
* Multi-dataset models are traiend on the concatinated data across datasets with codes (`MdWfIbDeHtNiSoScLa`) for the nine datasets (`gl.ds_code`). Using NNLS and L2Reg. 
* Regularization was determined by doing leave-one-dataset-out cross-validation and finding the log-alpha that had the best performance. This resulted in NNLS to A0 and L2Reg A2.
* For the paper we then also report the leave-one-out performances at that regualization level. 
* `avg` is the averaged individual models, and `group` are the averaged group models from the single dataset models. 

