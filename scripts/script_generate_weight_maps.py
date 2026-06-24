import numpy as np
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.summarize as cs
import nibabel as nb


def generate_weight_maps(traindata = gl.traindata_string(),
                         cortex_roi = 'Icosahedron1002',
                         method = 'NNLS',
                         model_ext = 'A0_global',
                         stats = 'mean',
                         mean_mode = '',
                         norm = True):
    
    cifti_img = cs.stats_weight_map_cortex(traindata = traindata,
                                           cortex_roi = cortex_roi,
                                           method = method,
                                           extension = model_ext,
                                           stats = stats,
                                           mean_mode = mean_mode,
                                           norm = True)

    if mean_mode == '':
        fname = f'{traindata}_{cortex_roi}_{method}_{stats}'
    else:
        fname = f'{traindata}_{cortex_roi}_{method}_{mean_mode}-{stats}'
    nb.save(cifti_img, gl.conn_dir + f'/maps/{fname}.pscalar.nii')
    print(f'{fname} map saved.')


if __name__ == "__main__":
    traindata = gl.traindata_string()
    cortex_roi = 'Icosahedron1002'
    method_list = ['NNLS', 'L2reg']
    model_ext_list = ['A0_global', 'A2_global']
    stats = 'mean'
    mean_mode_list = ['abs', 'pos']

    for method, model_ext in zip(method_list, model_ext_list):
        generate_weight_maps(traindata=traindata,
                             cortex_roi=cortex_roi,
                             method=method,
                             model_ext=model_ext,
                             stats=stats)
        
        if method == 'L2reg' and stats == 'mean':
            for mean_mode in mean_mode_list:
                generate_weight_maps(traindata=traindata,
                                    cortex_roi=cortex_roi,
                                    method='L2reg',
                                    model_ext=model_ext,
                                    stats='mean',
                                    mean_mode=mean_mode)
    
    