from exp_helpers import base_experiment

# combination of damage types, on sparified network 
# get this distribution of damage types from Wang paper
base_experiment(expnum = '5', pie_chart = [.3, .45, .2, .05], sparsity_cutoff = 34.7)