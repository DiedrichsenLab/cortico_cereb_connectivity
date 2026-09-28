Seed Correlation
================

Let's say you have a Region of Interest in cortex/hippocampus/cerebellum and you want to do seed based correaltion
analysis. Using our 9 datasets having over 450 unique tasks conditions from nearly 200 subjects, we provide seed based
functional correlation.


Example with default atlas and all datasets
-------------------------------------------
The default setting will use all the 9 preprocessed datasets, parcellate the cortex with `Icosahedron1002`, parcellate the
cerebellum with `NettekovenSym32`, and aggregates the hippocampus data using `Platchi5` regions. To cancel data aggregation,
simply pass `None` for the corresponding roi setting. In this case, the native space regions will be used.

.. code-block:: python

    import cortico_cereb_connectivity.globals as gl
    import cortico_cereb_connectivity.summarize as cs
    corr_xy, corr_hy = cs.seed_correlation(dscode=gl.traindata_string(),
                                           cortex_roi='Icosahedron1002',
                                           cerebellum_roi='NettekovenSym32',
                                           hippocampus_roi='Platchi5',
                                           cortex='fs32k',
                                           cerebellum='MNISymC3',
                                           hippocampus='MNIAsymHippocampus')

If not using Hippocampus at all, the function will return only `corr_xy`:

.. code-block:: python

    import cortico_cereb_connectivity.summarize as cs
    corr_xy = cs.seed_correlation(cortex_roi='Icosahedron1002',
                                  cerebellum_roi=None,
                                  cortex='fs32k',
                                  cerebellum='MNISymC3',
                                  hippocampus=None)

In this case, ``hippocampus=None`` prevents loading hippocampus data, and ``cerebellum_roi=None`` prevents aggregating data within
Cerebellum space.
