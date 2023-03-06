from pathlib import Path

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/home/ladi/diedrichsen_data/data/FunctionalFusion'

conn_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = '/srv/diedrichsen/data/Cerebellum/connectivity'
if not Path(conn_dir).exists():
    conn_dir = '/home/ladi/diedrichsen_data/data/Cerebellum/connectivity'

atlas_dir = base_dir + '/Atlases'