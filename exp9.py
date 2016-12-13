from exp_helpers import base_experiment
from numpy import arange

myalpha = 1. / 5050.
base_experiment(expnum = '9', aging_flag = 1, damages_values = (arange(100)+1)*myalpha, pie_chart = [.3, .45, .2, .05], sparsity_cutoff = 34.7)