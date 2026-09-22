Cortical Weight Map
===================

Let's say you have a Region of Interest in cerebellum and you want to know which areas in the cortex are connected
to it. In fact, one of the main applications of a general cortico-cerebellar connectivity model is to provide
this to better understand the ROI's function. A ``cortical weight map`` is
basically a `CIFTI` file indicating average weights from the cortical areas to the cerebellar ROI.

Therefore, to generate a ``cortical weight map`` you will need to define a cerebellar ROI (or multiple ROIs with 
different labels) as a `NIFTI` file. Then, you can use the default connectivity model or specify one to get the weights from.


Creating a cerebellar ROI as `NIFTI`
------------------------------------
TO BE COMPLETED


Generating a cortical weight map
--------------------------------
First, load yout ROI using ``nibabel``:

.. code-block:: python

    import nibabel as nib
    my_ROI_nifti = nib.load("/path/to/my_ROI.nii")

Then, the ``avrg_weight_map_roi()`` function in the ``summarize.py`` module will do the computation.
To call this function, you need to specify the model you want to be used for the weight maps. You can leave it to use
our default large-scale global model, but also can pass a `Model` object.


Example with default connectivity model
---------------------------------------
The default connectivity model has been trained on 9 datasets and over 450 unique task conditions from nearly 200 subjects.
This model uses `L2regression` as the linear connectivity, uses `Icosahedron1002` as the cortical parcellation, and `MNISymC3`
as the cerebellar atlas.

Let's say your ``my_ROI_nifti`` (a ``nib.Nifti1Image``) contains labels as `[0,1,2]` for the background, an inferior left hem ROI, and a 
superior right hem ROI. To use this model, simply call:

.. code-block:: python

    import cortico_cereb_connectivity.summarize as cs
    cortical_wmap_cifti = cs.avrg_weight_map_roi(cerebellum_roi=my_ROI_nifti,
                                                 cereb_roi_labels=['my_ROI_inferior_L', 'my_ROI_superior_R'])

    # save the CIFTI
    nib.save(cortical_wmap_cifti, "/path/to/my_wmap.pscalar.nii")

You can view the generated cortical weight map using the `Workbench Viewer` or load it with `NiBabel` and process it.


Example with custom connectivity model
---------------------------------------
TO BE COMPLETED