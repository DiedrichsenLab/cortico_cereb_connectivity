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

fig_dir = '/Users/jdiedrichsen/Dropbox/Talks/2025/07_Gordon/Gordon_connectivity/figure_parts'

# Default datasets and sessions for training and evaluation
datasets = ['MDTB',     'WMFS',     'IBC',  'Demand', 'HCPur100','Nishimoto','Somatotopic','Social','Language']
sessions = ['all',      'all',     'all',   'all'  ,  'all',     'all',       'all',       'ses-social', 'ses-localizer']
add_rest = [False,      True ,     True,     True,     True,      False,       True,        False,  False]
std_cortex = ['parcel', 'global', 'parcel', 'parcel', 'parcel', 'parcel',   'global',       'parcel', 'parcel']
dscode   = ['Md',      'Wf',        'Ib',   'De',     'Ht',      'Ni',        'So',         'Sc',     'La']

def get_ldo_names():
   num_ds = len(dscode)
   ldo_names = []
   for i in range(num_ds):
      ldo_names.append(''.join(dscode[:i]+dscode[i+1:]))
   return ldo_names