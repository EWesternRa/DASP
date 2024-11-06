import argparse
import time

import numpy as np
import igraph as ig
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pmd import compute_probability_Minkowski_distance
from simple_path_tree import simple_path_tree
from utils import get_node_labels, load_data_ori, custom_grid_search_cv


def compute_feats(
    graphs,
    maxh,
    depth,
    size,
    window,
    label_type="label",
    ngram_type=1,
    sampling_type=0,
):

    all_labels = {
        0: get_node_labels(graphs, label_type=label_type),
    }
    # generate all simple paths
    igraphs = [ig.Graph.from_networkx(g) for g in graphs]
    sps = []
    for i in range(len(graphs)):
        paths_graph = [
            igraphs[i].get_all_simple_paths(vs, cutoff=depth) for vs in igraphs[i].vs
        ]
        sps.append(paths_graph)

    # generate labels, for each deep
    for deep in range(1, maxh):
        labeledtrees = []
        labeledtrees_set = set()

        for igraph, graph in zip(igraphs, graphs):
            # generate simple path tree encoding
            subtrees = simple_path_tree(igraph, graph, deep)

            labeledtrees.append(subtrees)
            labeledtrees_set.update(subtrees)
        labeledtrees_set = sorted(list(labeledtrees_set))

        # extend labels
        all_labels[deep] = {}
        for gid, lt in enumerate(labeledtrees):
            all_labels[deep][gid] = np.array([labeledtrees_set.index(t) for t in lt])

    # compute node embeddings
    all_node_embeddings = []
    for deep in range(maxh):

        corpus = []  # corpus for word2vec
        graph_label_paths = []
        for gid, graph_sps in enumerate(sps):
            graph_label_paths.append([])
            all_label_paths = []

            for node, sp in enumerate(graph_sps):
                graph_label_paths[gid].append([])
                for path in sp:
                    path_str = ",".join([str(all_labels[deep][gid][n]) for n in path])
                    graph_label_paths[gid][node].append(path_str)
                # sort the simple paths from the same node
                graph_label_paths[gid][node].sort()  # by default, sort by lexicographical order
                all_label_paths.append(graph_label_paths[gid][node])

            corpus.extend(all_label_paths)

        # word2vec
        model = Word2Vec(
            corpus,
            vector_size=size,
            window=window,
            min_count=0,
            workers=16,
            sg=ngram_type,
            hs=sampling_type,
        ).wv

        # every node's embedding is the sum of its simple paths embeddings
        node_embeddings = []
        for gid, graph_sps in enumerate(sps):
            graph_node_embeddings = []
            for node, label_sp in enumerate(graph_label_paths[gid]):
                node_embed = np.zeros(size)
                for path in label_sp:
                    node_embed += model[path]
                graph_node_embeddings.append(node_embed)
            node_embeddings.append(graph_node_embeddings)

        all_node_embeddings.append(node_embeddings)

    return all_node_embeddings


def main(
    dataset,
    K,
    H,
    size,
    label_type="label",
    data_path="datasets",
    gridsearch=True,
    crossvalidation=True,
    random_state=42,
    window=10,
    gamma=None,
):
    print(f"Running DASP on {dataset} with K={K}, H={H}, size={size}")
    graphs = load_data_ori(dataset, data_path)
    print("loading done.")

    start = time.time()

    # compute node embeddings
    graph_embeds = compute_feats(graphs, K, H, size, window, label_type)
    print("compute node embedding done.")

    distance_matrix = np.zeros((len(graphs), len(graphs)))
    for i in tqdm(range(K), desc="computing distance matrix"):
        means = []
        vars = []
        for graph_embed in graph_embeds[i]:
            means.append(np.mean(graph_embed, axis=0))
            var_temps = np.var(graph_embed, axis=0)
            for k in range(len(var_temps)):
                if var_temps[k] <= 0.001:
                    var_temps[k] = 0.001
            vars.append(var_temps)
        distance_matrix += compute_probability_Minkowski_distance(means, vars)

    end = time.time()
    print(f"total time: {end - start} s")

    if gridsearch:
        if gamma is not None:
            gammas = gamma
        else:
            gammas = np.logspace(-6, 1, num=8)
        param_grid = [{"C": np.logspace(-3, 3, num=7)}]
    else:
        gammas = [0.001]

    kernel_matrices = []
    kernel_params = []
    # Generate the full list of kernel matrices from which to select
    M = distance_matrix
    for ga in gammas:
        K = np.exp(-ga * M)
        kernel_matrices.append(K)
        kernel_params.append(ga)

    print(
        f"Running SVMs, crossvalidation: {crossvalidation}, gridsearch: {gridsearch}."
    )

    y = np.array([g.graph["label"] for g in graphs])

    accuracy_scores = []
    np.random.seed(random_state)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # Hyperparam logging
    best_C = []
    best_gamma = []

    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]

        # Gridsearch
        if gridsearch:
            gs, best_params = custom_grid_search_cv(
                SVC(kernel="precomputed"),
                param_grid,
                K_train,
                y_train,
                cv=5,
                random_state=random_state,
            )
            # Store best params
            C_ = best_params["params"]["C"]
            gamma_ = kernel_params[best_params["K_idx"]]
            y_pred = gs.predict(K_test[best_params["K_idx"]])
        else:
            gs = SVC(C=100, kernel="precomputed").fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            gamma_, C_ = gammas[0], 100
        best_C.append(C_)
        best_gamma.append(gamma_)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        if not crossvalidation:
            break

    # ---------------------------------
    # Printing and logging
    # ---------------------------------
    if crossvalidation:
        print(
            "Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %".format(
                np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100
            )
        )
    else:
        print("Final accuracy: {:2.3f} %".format(np.mean(accuracy_scores) * 100))

    return (
        np.mean(accuracy_scores),
        np.std(accuracy_scores),
        end - start,
    )


def arg_parser():
    arg_parser = argparse.ArgumentParser()

    # DASP parameters
    arg_parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="K for simple-path-tree")
    arg_parser.add_argument(
        "--H",
        type=int,
        default=2,
        help="H for simple paths to generate node embeddings",
    )

    # dataset parameters
    arg_parser.add_argument("--dataset",
        type=str,
        default="MUTAG",
        help="Dataset name")
    arg_parser.add_argument(
        "--data_path", type=str, default="datasets", help="Dataset path"
    )
    arg_parser.add_argument(
        "--label_type",
        type=str,
        default="label",
        help="Node label type, label or degree or uniform",
    )
    arg_parser.add_argument(
        "--gridsearch", type=bool, default=True, help="Whether to perform grid search"
    )
    arg_parser.add_argument(
        "--crossvalidation",
        type=bool,
        default=True,
        help="Whether to perform cross validation",
    )
    arg_parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state")

    # word2vec parameters
    arg_parser.add_argument("--size", type=int, default=16, help="Embedding size")
    arg_parser.add_argument("--window", type=int, default=10, help="Window size")

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(
        args.dataset,
        args.K,
        args.H,
        args.size,
        args.label_type,
        args.data_path,
        args.gridsearch,
        args.crossvalidation,
        args.random_state,
        args.window,
    )
