import sys
from .pcaeval.pca_eval import *
from .version import __version__

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['generate_linear_data', 'evaluate_matrix',
'generate_nonlinear_data', 'generate_uncorrelated_data', 'knn_spearman_r', 'preprocess', 'process', 'evaluate_anndata_file_list', 'plot_silhouette', 'plot_sing_vals_exp_var']})

del sys