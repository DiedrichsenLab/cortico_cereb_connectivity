"""
script for training models 
@ Ladan Shahshahani, Joern Diedrichsen Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import re
import sys
from collections import defaultdict
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as cm
import json


def train_models(dataset_name, ses_id, type, atlas, model_name, model_type, model_params, model_dir, model_file):
    pass

def avrg_model(dataset_name, ses_id, type, atlas, model_name, model_type, model_params, model_dir, model_file):
    pass

def eval_models(dataset_name, ses_id, type, atlas, model_name, model_type, model_params, model_dir, model_file):
    pass