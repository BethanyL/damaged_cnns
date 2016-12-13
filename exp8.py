from exp_helpers import base_experiment
from numpy import tile

base_experiment(expnum = '8', aging_flag = 1, damages_values = tile(.01, 100), pie_chart = [.3, .45, .2, .05], sparsity_cutoff = 34.7)