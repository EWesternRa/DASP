import numpy as np
import math


def compute_probability_Minkowski_distance(means, variance):
    """
    Compute the probability Minkowski distance between two graphs.
    """

    n = len(means)
    M_tmp = np.zeros((n, n))

    d1_values = [
        matching_probability(means[i], variance[i], means[i], variance[i])
        for i in range(n)
    ]

    for i in range(n):
        d1 = d1_values[i]
        for j in range(i + 1, n):
            d2 = d1_values[j]
            d3 = matching_probability(means[i], variance[i], means[j], variance[j])
            dis = d1 + d2 - 2 * d3
            if dis < 0:
                dis = 0
            M_tmp[i][j] = math.sqrt(dis)
            M_tmp[j][i] = M_tmp[i][j]  # by symmetry

    return M_tmp


def matching_probability(means_A, variance_A, means_B, variance_B):
    """
    Compute the matching probability between two graphs.
    """

    tmp = -((means_A - means_B) ** 2) / (
        2 * variance_A + 2 * variance_B + 0.0001
    ) - 0.5 * np.log(2 * math.pi * (variance_A + variance_B) + 0.0001)
    dis_tmp = np.sum(tmp)

    return dis_tmp