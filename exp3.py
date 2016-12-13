from exp_helpers import base_experiment
from numpy import arange

base_experiment(expnum = 3, damages_values = arange(0,50,1), histogram_flag = 1, filter_type = "outside", max_trials = 1)