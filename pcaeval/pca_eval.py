import numpy as np
import gc
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import issparse
from sklearn.datasets import make_classification, make_sparse_uncorrelated
from matplotlib import pyplot as plt
from umap.umap_ import UMAP
try:
    import scanpy as sc
    _HAS_SCANPY = True
except ImportError:
    _HAS_SCANPY = False
try:
    import topo as tp
    _HAS_TOPO = True
except ImportError:
    _HAS_TOPO = False


# Define stereotypical data
# Linearly correlated
def generate_linear_data(n=1000, d=100, n_classes=10, redundancy=0.1, noise=0.1, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=int(redundancy*d), n_redundant=int((1-redundancy) * d), n_classes=n_classes, random_state=seed)
    rng = np.random.RandomState(seed)
    X += noise * rng.uniform(size=X.shape)
    return X

# Uncorrelated
def generate_uncorrelated_data(n=1000, d=100, seed=0):
    X, y = make_sparse_uncorrelated(
        n_samples=n, n_features=d, random_state=seed)
    return X


# Score metrics from TopOMetry
def geodesic_distance(data, method='D', unweighted=False, directed=True):
    G = shortest_path(data, method=method,
                      unweighted=unweighted, directed=directed)
    # guarantee symmetry
    G = (G + G.T) / 2
    G[(np.arange(G.shape[0]), np.arange(G.shape[0]))] = 0
    return G

def symmetrize(knn):
    knn = (knn + knn.T)/2
    knn[(np.arange(knn.shape[0]), np.arange(knn.shape[0]))] = 0
    if issparse(knn):
        knn = knn.toarray()
    return knn

# Score metrics from TopOMetry
def graph_spearman_r(data_graph, pca_grap):
    res, _ = spearmanr(squareform(symmetrize(data_graph)), squareform(symmetrize(pca_grap)))
    return res

def knn_spearman_r(data_graph, embedding_graph, path_method='D', subsample_idx=None, unweighted=True):
    # data_graph is a (N,N) similarity matrix from the reference high-dimensional data
    # embedding_graph is a (N,N) similarity matrix from the lower dimensional embedding
    geodesic_dist = geodesic_distance(
        data_graph, method=path_method, unweighted=unweighted)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    embedded_dist = geodesic_distance(
        embedding_graph, method=path_method, unweighted=unweighted)
    res, _ = spearmanr(squareform(geodesic_dist), squareform(embedded_dist))
    return res


# Functions for dealing with AnnData objects
if _HAS_SCANPY:
    from anndata import AnnData

    def preprocess(AnnData, norm_log=True, verbose=False, **kwargs):
        AnnData.var_names_make_unique()
        if verbose:
            print('Preprocessing...')
        sc.pp.filter_cells(AnnData, min_genes=200)
        sc.pp.filter_genes(AnnData, min_cells=3)
        if norm_log:
            sc.pp.normalize_total(AnnData, target_sum=1e4)
            sc.pp.log1p(AnnData)
        sc.pp.highly_variable_genes(AnnData, **kwargs)
        AnnData.raw = AnnData
        AnnData = AnnData[:, AnnData.var['highly_variable']].copy()
        sc.pp.scale(AnnData, max_value=10)
        if verbose:
            print('...done!')
        return AnnData.copy()

    def process(AnnData, n_pcs=100, n_neighbors=10, metric='cosine', n_jobs=-1, verbose=False, **kwargs):
        if verbose:
            print('Processing...')
        if 'X_pca' not in AnnData.obsm.keys():
            sc.tl.pca(AnnData, n_comps=n_pcs)
        # PCA derived
        umapper_pca = UMAP(n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs, **kwargs).fit(AnnData.obsm['X_pca'][:, :n_pcs])
        umap_pca = umapper_pca.transform(AnnData.obsm['X_pca'][:, :n_pcs])
        umap_pca_graph = umapper_pca.graph_
        AnnData.obsp['PCA_kNN_connectivities'] = umap_pca_graph
        AnnData.obsm['X_UMAP_on_PCA'] = umap_pca
        sc.tl.leiden(AnnData, resolution=0.8, directed=False, key_added='PCA_kNN_leiden', adjacency=umap_pca_graph)
        # Data-derived; UMAP does not like scaled data
        ad_for_umap = AnnData.raw.to_adata()
        ad_for_umap = ad_for_umap[:, ad_for_umap.var['highly_variable']].copy()
        umapper_data = UMAP(n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs, **kwargs).fit(ad_for_umap.X)
        umap_data = umapper_data.transform(ad_for_umap.X)
        umap_data_graph = umapper_data.graph_
        AnnData.obsp['Data_kNN_connectivities'] = umap_data_graph
        AnnData.obsm['X_UMAP_on_Data'] = umap_data
        sc.tl.leiden(AnnData, resolution=0.8, directed=False, key_added='Data_kNN_leiden', adjacency=umap_data_graph)
        if verbose:
            print('...done!')
        return AnnData.copy()

    def evaluate_anndata(AnnData, preprocessed=False, processed=False, n_pcs=100, metric='euclidean', n_neighbors=10, n_jobs=-1, verbose=False, **kwargs):
        if not preprocessed:
            AnnData = preprocess(AnnData, norm_log=True,
                                 verbose=verbose)
        pca = PCA(n_components=n_pcs)
        pca_Y = pca.fit_transform(AnnData.X)
        AnnData.obsm['X_pca'] = pca_Y
        if not processed:
            AnnData = process(AnnData, n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs)
        # Compute scores
        pca_graph = AnnData.obsp['PCA_kNN_connectivities']
        data_graph = AnnData.obsp['Data_kNN_connectivities']
        umap_on_pca_graph = kneighbors_graph(
            AnnData.obsm['X_UMAP_on_PCA'], n_neighbors=n_neighbors, n_jobs=n_jobs, mode='distance', metric=metric)
        umap_on_data_graph = kneighbors_graph(
            AnnData.obsm['X_UMAP_on_Data'], n_neighbors=n_neighbors, n_jobs=n_jobs, mode='distance', metric=metric)
        results_dict = {'PCA_estimator': pca,
                        'singular_values': pca.singular_values_,
                        'explained_variance': pca.explained_variance_ratio_.cumsum(),
                        'graph_correlation': graph_spearman_r(pca_graph, data_graph),
                        'embedding_correlation': graph_spearman_r(umap_on_pca_graph, umap_on_data_graph)}
        gc.collect()
        return results_dict, AnnData.copy()

    def evaluate_anndata_file_list(adata_files, n_pcs=100, metric='euclidean', n_neighbors=5, save_intermediate=False, save_dir=None, verbose=False,  not_normalize_idx=None, return_dict=True, **kwargs):
        if not return_dict:
            pca_estimators = []
            sin_vals = []
            explained_vars = []
            graph_correlation = []
            embedding_correlation = []
            cluster_score_pca = []
            cluster_score_data = []
        else:
            results_dict = {}
        if save_intermediate:
            if save_dir is None:
                raise ValueError('Save directory not specified!')
        for i in range(len(adata_files)):
            if verbose:
                print('Processing file {} of {}'.format(i+1, len(adata_files)))
            adata = sc.read_h5ad(adata_files[i])
            if not_normalize_idx is not None:
                if i in not_normalize_idx:
                    adata = preprocess(adata, norm_log=False,
                                       verbose=verbose)
                else:
                    adata = preprocess(adata, min_mean=0.0125,
                                       max_mean=3, min_disp=0.5)
            else:
                adata = preprocess(adata, min_mean=0.0125,
                                   max_mean=3, min_disp=0.5)
            pca = PCA(n_components=n_pcs)
            pca_Y = pca.fit_transform(adata.X)
            adata.obsm['X_pca'] = pca_Y
            if not return_dict:
                pca_estimators.append(pca)
                sin_vals.append(pca.singular_values_)
                explained_vars.append(pca.explained_variance_ratio_.cumsum())
            # With PCA
            adata = process(adata, pca=True, **kwargs)
            # Without PCA
            adata = process(adata, pca=False, **kwargs)
            if save_intermediate:
                adata.write_h5ad(save_dir + 'processed_' +
                                 adata_files[i].split('/')[-1])
            # Compute scores
            pca_graph = adata.obsp['PCA_kNN_connectivities']
            data_graph = adata.obsp['Data_kNN_connectivities']
            umap_on_pca_graph = kneighbors_graph(
                adata.obsm['X_UMAP_on_PCA'], n_neighbors=n_neighbors, mode='distance', metric=metric)
            umap_on_data_graph = kneighbors_graph(
                adata.obsm['X_UMAP_on_Data'], n_neighbors=n_neighbors, mode='distance', metric=metric)
            if not return_dict:
                graph_correlation.append(graph_spearman_r(pca_graph, data_graph))
                embedding_correlation.append(graph_spearman_r(
                    umap_on_pca_graph, umap_on_data_graph))
            else:
                results_dict[str(adata_files[i].split('/')[-1])[:-5]] = {'PCA_estimator': pca,
                                                                         'singular_values': pca.singular_values_,
                                                                         'explained_variance': pca.explained_variance_ratio_.cumsum(),
                                                                         'graph_correlation': graph_spearman_r(pca_graph, data_graph),
                                                                         'embedding_correlation': graph_spearman_r(umap_on_pca_graph, umap_on_data_graph)}
            del adata
            gc.collect()
            if verbose:
                print('Done with file: ' + adata_files[i])
        if not return_dict:
            return pca_estimators, sin_vals, explained_vars, graph_correlation, embedding_correlation, cluster_score_pca, cluster_score_data
        else:
            return results_dict


def print_dict_results(results_dict):
    for i, dataset_name in enumerate(results_dict):
        print('\n \n  --- ' + str(dataset_name) + ' --- ' +
              '\n Spearman R correlation between the k-nearest-neighbors graphs learned from the data and the Principal Components: %f' % results_dict[dataset_name]['graph_correlation'] +
              '\n Spearman R correlation between the UMAP embeddings learned from these graphs %f' % results_dict[dataset_name]['embedding_correlation'] +
              '\n Total explained variance with the first 100 PCs: %f' % results_dict[dataset_name]['explained_variance'].max())


# Function for evaluating generic numpy data array
def evaluate_matrix(X, n_pcs=100, dimred_estimator=None, precomputed_knn=None, metric='euclidean', n_neighbors=5, n_jobs=1, verbose=False, **kwargs):
    results_dict = {}
    _skip_dimred = False
    pca = PCA(n_components=n_pcs)
    pca_Y = pca.fit_transform(X)
    if precomputed_knn is None:
        if verbose:
            print('No precomputed kNN graph specified, computing...')
        data_graph = kneighbors_graph(
            X, n_neighbors=n_neighbors, mode='distance', metric=metric, n_jobs=n_jobs)
    else:
        data_graph = precomputed_knn
    pca_graph = kneighbors_graph(
        pca_Y[:, :n_pcs], n_neighbors=n_neighbors, mode='distance', metric=metric, n_jobs=n_jobs)
    if isinstance(dimred_estimator, bool):
        if not dimred_estimator:
            _skip_dimred = True
    if not _skip_dimred:
        if dimred_estimator is None:
            if verbose:
                print('No non-linear dimensionality reduction estimator specified, using UMAP...')
            dimred_estimator = UMAP(n_neighbors=n_neighbors, metric=metric, verbose=verbose)
        dim_red_Y = dimred_estimator.fit_transform(X)
        dim_red_with_PCA_Y = dimred_estimator.fit_transform(pca_Y[:, :n_pcs])
        dimred_on_data_graph = kneighbors_graph(
            dim_red_Y, n_neighbors=n_neighbors, mode='distance', metric=metric, n_jobs=n_jobs)
        dimred_on_pca_graph = kneighbors_graph(
            dim_red_with_PCA_Y, n_neighbors=n_neighbors, mode='distance', metric=metric, n_jobs=n_jobs)
        embedding_correlation = graph_spearman_r(
            dimred_on_pca_graph, dimred_on_data_graph)
    else:
        embedding_correlation = 0
    results_dict = {'PCA_estimator': pca,
                    'singular_values': pca.singular_values_,
                    'explained_variance': pca.explained_variance_ratio_.cumsum(),
                    'graph_correlation': graph_spearman_r(pca_graph, data_graph),
                    'embedding_correlation': embedding_correlation}
    return results_dict

# Plotting functions


def plot_sing_vals_exp_var(results_dict, dataset=None, fontsize=16, figsize=(8, 4), **kwargs):
    # If no dataset is specified, plot for all datasets
    if isinstance(dataset, bool):
        plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0.2, right=0.98, bottom=0.001,
                            top=0.9, wspace=0.15, hspace=0.01)
        if not dataset:
            plt.subplot(1, 2, 1)
            _plot_singular_values(
                results_dict['singular_values'], fontsize, **kwargs)
            plt.subplot(1, 2, 2)
            _plot_explained_variance(
                results_dict['explained_variance'], fontsize, **kwargs)
            plt.tight_layout()
    elif dataset is not None:
        plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0.2, right=0.98, bottom=0.001,
                            top=0.9, wspace=0.15, hspace=0.01)
        if not isinstance(dataset, str):
            raise ValueError('dataset must be a string!')
        if dataset not in results_dict.keys():
            raise ValueError('Dataset not found in results dictionary!')
        plt.suptitle(dataset, fontsize=fontsize+4)
        plt.subplot(1, 2, 1)
        _plot_singular_values(
            results_dict[dataset]['singular_values'], fontsize, **kwargs)
        plt.subplot(1, 2, 2)
        _plot_explained_variance(
            results_dict[dataset]['explained_variance'], fontsize, **kwargs)
        plt.tight_layout()
    else:
        for i, dataset_name in enumerate(results_dict):
            plt.figure(figsize=figsize)
            plt.subplots_adjust(left=0.2, right=0.98,
                                bottom=0.001, top=0.9, wspace=0.15, hspace=0.01)
            plt.suptitle(dataset_name, fontsize=fontsize+4)
            plt.subplot(1, 2, 1)
            plt.plot(range(0, len(results_dict[dataset_name]['singular_values'])),
                     results_dict[dataset_name]['singular_values'], marker='o', **kwargs)
            plt.xlabel('Number of PCs', fontsize=fontsize)
            plt.ylabel('Singular values', fontsize=fontsize)
            plt.subplot(1, 2, 2)
            plt.plot(range(0, len(results_dict[dataset_name]['explained_variance'])),
                     results_dict[dataset_name]['explained_variance'], marker='o', **kwargs)
            plt.xlabel('Number of PCs', fontsize=fontsize)
            plt.ylabel('Total explained variance', fontsize=fontsize)
            plt.tight_layout()
            plt.show()


def _plot_singular_values(sin_vals, fontsize=13, **kwargs):
    plt.plot(range(0, len(sin_vals)), sin_vals, marker='o', **kwargs)
    plt.xlabel('Number of PCs', fontsize=fontsize)
    plt.ylabel('Singular values', fontsize=fontsize)


def _plot_explained_variance(exp_var, fontsize=13, **kwargs):
    plt.plot(range(0, len(exp_var)), exp_var, marker='o', **kwargs)
    plt.xlabel('Number of PCs', fontsize=fontsize)
    plt.ylabel('Total explained variance', fontsize=fontsize)
