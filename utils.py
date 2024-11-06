import os
import ot
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone

def load_data_ori(dataset, data_path='datasets'):
    """
    Load dataset from datsets folder.
    
    """
    print('Loading {} dataset...'.format(dataset))
    adj_file = os.path.join(data_path, dataset, dataset + '_A.txt')
    graph_id_file = os.path.join(data_path, dataset, dataset + '_graph_indicator.txt')
    graph_label_file = os.path.join(data_path, dataset, dataset + '_graph_labels.txt')
    node_label_file = os.path.join(data_path, dataset, dataset + '_node_labels.txt')
    node_attr_file = os.path.join(data_path, dataset, dataset + '_node_attributes.txt')
    edge_label_file = os.path.join(data_path, dataset, dataset + '_edge_labels.txt')
    edge_attr_file = os.path.join(data_path, dataset, dataset + '_edge_attributes.txt')
    
    if not os.path.exists(adj_file):
        raise FileNotFoundError(f'Dataset {dataset} not found.')
    
    has_label = False
    has_attr = False
    has_edge_attr = False
    has_edge_label = False
    if os.path.exists(adj_file):
        adj = pd.read_csv(adj_file, header=None, index_col=None).values
    if os.path.exists(graph_id_file):
        graph_indicator = pd.read_csv(graph_id_file, header=None, index_col=None).values
    if os.path.exists(graph_label_file):
        graph_label = pd.read_csv(graph_label_file, header=None, index_col=None).values
    if os.path.exists(node_label_file):
        node_label = pd.read_csv(node_label_file, header=None, index_col=None).values
        has_label = True
    if os.path.exists(node_attr_file):
        node_attr = pd.read_csv(node_attr_file, header=None, index_col=None).values
        has_attr = True
    if os.path.exists(edge_label_file):
        edge_label = pd.read_csv(edge_label_file, header=None, index_col=None).values
        has_edge_label = True
    if os.path.exists(edge_attr_file):
        edge_attr = pd.read_csv(edge_attr_file, header=None, index_col=None).values
        has_edge_attr = True
    graphs = []
    num_graph = np.max(graph_indicator)
    edge_ind = 0
    for i in range(num_graph):
        g = nx.Graph()
        g.graph['label'] = graph_label[i][0]
        g.graph['id'] = i
        
        # add nodes
        node_indicator = np.where(graph_indicator == i+1)[0]
        node_indicator = node_indicator + 1    # node id start from 1
        for node in node_indicator:
            g.add_node(node)
            if has_label:
                g.nodes[node]['label'] = node_label[node-1][0]
            if has_attr:
                g.nodes[node]['attr'] = node_attr[node-1]
        
        # add edges
        while edge_ind < len(adj):
            edge = adj[edge_ind]
            if edge[0] in node_indicator and edge[1] in node_indicator:
                g.add_edge(edge[0], edge[1])
                if has_edge_attr:
                    g.edges[edge[0], edge[1]]['attr'] = edge_attr[edge_ind][0]
                edge_ind += 1
            else:
                break
        
        # convert node labels to integers from 0
        g = nx.convert_node_labels_to_integers(g, first_label=0)
        if not has_label:
            # if no node label, use degree as node label
            node_labels = [g.degree[node] for node in g.nodes()]
            nx.set_node_attributes(g, dict(zip(g.nodes(), node_labels)), 'label')
        
        graphs.append(g)
        
    return graphs


def compute_wasserstein_distance(label_sequences, sinkhorn=False, 
                                    categorical=False, sinkhorn_lambda=1e-2):
    """
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    """
    # Get the iteration number from the embedding file
    n = len(label_sequences)
    
    M = np.zeros((n,n))
    # Iterate over pairs of graphs
    for graph_index_1, graph_1 in enumerate(label_sequences):
        # Only keep the embeddings for the first h iterations
        labels_1 = label_sequences[graph_index_1]
        for graph_index_2, graph_2 in enumerate(label_sequences[graph_index_1:]):
            labels_2 = label_sequences[graph_index_2 + graph_index_1]
            # Get cost matrix
            ground_distance = 'hamming' if categorical else 'euclidean'
            costs = ot.dist(labels_1, labels_2, metric=ground_distance)

            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                    np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                    numItermax=50)
                M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, graph_index_2 + graph_index_1] = \
                    ot.emd2([], [], costs)
                    
    M = (M + M.T)
    return M

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5, random_state=42):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc.get('test_scores'))
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    # print(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


def get_node_labels(graphs, label_type="label"):
    """
    Get node labels from graphs
    """
    labels = {}
    for i, g in enumerate(graphs):
        if label_type == "label":
            labels[i] = np.array([g.nodes[n]["label"] for n in g.nodes()])
        elif label_type == "degree":
            labels[i] = np.array([g.degree(n) for n in g.nodes()])
        elif label_type == "uniform":
            labels[i] = np.ones(len(g.nodes()))
    return labels


def dfs_paths_with_depth(G, source, depth, current_depth=0, path=None):
    """
    Generate all possible paths with a most depth from source node
    """
    if path is None:
        path = [source]
    if 1 <= current_depth <= depth:
        yield from [path]
    if current_depth == depth:
        return
    for neighbor in G.neighbors(source):
        if neighbor not in path:
            yield from dfs_paths_with_depth(
                G, neighbor, depth, current_depth + 1, path + [neighbor]
            )