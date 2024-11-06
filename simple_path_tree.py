def simple_path_tree(igraph, graph, deep):
    """
    Generate simple path tree from igraph graph with depth deep.
    Sorted by lexicographical order.
    """
    subtrees = []
    # graph
    all_paths = [
        igraph.get_all_simple_paths(vs, cutoff=deep)
        for vs in igraph.vs
    ]
    for simple_paths in all_paths:
        sorted_paths = sorted(simple_paths, key=lambda x: len(x))
        edges = [(sp[-2], sp[-1]) for sp in sorted_paths]

        tree_list = []
        if edges:
            prev_start = edges[0][0]
            subtree_list = [
                f"{graph.nodes[edges[0][0]]['label']},{graph.nodes[edges[0][1]]['label']}"
            ]
            for u, v in edges[1:]:
                if u == prev_start:
                    # the same subtree
                    subtree_list.append(
                        f"{graph.nodes[u]['label']},{graph.nodes[v]['label']}"
                    )
                else:
                    # new subtree
                    tree_list.append(",".join(sorted(subtree_list)))
                    prev_start = u
                    subtree_list = [
                        f"{graph.nodes[u]['label']},{graph.nodes[v]['label']}"
                    ]
        tree_list.append(",".join(sorted(subtree_list)))
        subtrees.append("-".join(tree_list))
    
    return subtrees