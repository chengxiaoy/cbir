from core.layers.functional import weight_Heat
import numpy as np
from core.layers.functional import get_potential_inv_re


def recall(queries, features, n_recall=10):
    """
    :param n_recall: the recall numbers
    :param queries: the vectors of query image
    :param features: vectors of the indexed images
    :return: ids of recall images and the features
    """
    pass


def rerank(query_id, recall_ids, recall_features):
    """
    rerank the recall result
    :param query_id:
    :param recall_ids:
    :param recall_features:
    :return:
    """
    pass


def HeR(ranks, ids, qvec):
    """
    ranks is top N rank vectors of C*N shape
    :param ranks:
    :return:
    """

    conc = np.concatenate((ranks, qvec.reshape(-1, 1)), axis=1)
    mean = np.mean(conc, axis=1)
    temp = conc - np.expand_dims(mean, axis=1).repeat(conc.shape[1], axis=1)
    a = np.dot(temp.T, temp)
    a = a + np.diag(-np.diag(a))
    a = np.where(a > 0.1, a, 0)
    const_z = 0.1
    z = const_z * np.mean(a[a > 0])
    weights = get_potential_inv_re(a, z)
    # descend
    indexs = np.argsort(-weights)
    return ids[indexs]


def qe(ranks, ids, qvec):
    for i in range(10):
        qvec = qvec + ranks[i]
    qvec = qvec / 11
    return recall(qvec)


def HeR_qe(ranks, ids, qvec):
    for i in range(10):
        qvec = qvec + ranks[i]
    qvec = qvec / 11
    return HeR_qe(ranks, ids, qvec)
