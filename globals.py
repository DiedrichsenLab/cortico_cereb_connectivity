from pathlib import Path

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion_new'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion_new'
if not Path(base_dir).exists():
    base_dir = '/cifs/diedrichsen/data/FunctionalFusion_new'
if not Path(base_dir).exists():
    base_dir = 'A:\\data\\FunctionalFusion_new'

conn_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = '/srv/diedrichsen/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = '/cifs/diedrichsen/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = 'A:\\data\\Cerebellum\\connectivity'

atlas_dir = base_dir + '/Atlases'

# Default datasets and sessions for training and evaluation
datasets = ['MDTB',     'WMFS',     'IBC',  'Demand', 'HCPur100','Nishimoto','Somatotopic','Social','Language']
sessions = ['all',      'all',     'all',   'all'  ,  'all',     'all',       'all',       ['ses-social'], ['ses-localizer']]
add_rest = [False,      True ,     True,     True,     True,      False,       True,        False,  False]
std_cortex = ['parcel', 'parcel', 'parcel', 'global', 'parcel', 'parcel',   'global',       'parcel', 'parcel']
dscode   = ['Md',      'Wf',        'Ib',   'De',     'Ht',      'Ni',        'So',         'Sc',     'La']
