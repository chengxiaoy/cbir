from core.layers.functional import weight_Heat
import numpy as np
from core.layers.functional import get_potential_inv_re

import numpy as np
from scipy import io
from PIL import Image
from time import time
from sklearn import preprocessing
import joblib
import cv2
import matplotlib.pyplot as plt
import os
from core.rerank import get_rerank_score, get_rerank_score_multiprocess
from sklearn.preprocessing import normalize as sknormalize
from config.config import *
import faiss
import logging
from lib.log import log
from core.helper import extract
from core.preprocess import get_transform


class Search:

    def __init__(self, model, features_path, pca_path, device, args):
        self.model = model
        self.features, self.paths = joblib.load(features_path)
        self.features = self.features.astype(np.float32)
        self.features = self.normalize(self.features)
        if args.pca:
            self.pca = joblib.load(pca_path)
        self.device = device
        self.args = args
        self.invert_index = self.get_invert_index(feature=self.features)

    def get_invert_index(self, feature):
        nlist = 2000
        d = len(feature[0])
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search
        # index = faiss.index_factory(d, "IVF1000,PQ128")
        assert not index.is_trained
        index.train(feature)
        assert index.is_trained
        index.add(feature)
        index.nprobe = 700
        return index

    def normalize(self, x, copy=False):
        """
        A helper function that wraps the function of the same name in sklearn.
        This helper handles the case of a single column vector.
        """
        if type(x) == np.ndarray and len(x.shape) == 1:
            return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
            # return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
        else:
            return sknormalize(x, copy=copy)
            # return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]

    def search(self, image_path, recall_num):
        img = get_transform(self.args)(image_path)
        query = extract(self.model, img, self.args, self.device)
        if self.args.pca:
            feature = self.pca.transform(np.array(query, dtype=np.float32))[0]
            feature = self.normalize(feature)
        else:
            feature = query[0]
        if self.args.rerank == 'her':
            D, I = self.invert_index.search(np.array([feature], dtype=np.float32), 1000)
        else:
            D, I = self.invert_index.search(np.array([feature], dtype=np.float32), recall_num)

        idxs, coarse_scores = I[0].tolist(), D[0].tolist()
        paths = []
        for idx in idxs:
            paths.append(self.paths[idx])
        paths = np.array(paths)

        if self.args.rerank == 'her':
            features = self.features[idxs]
            rerank_paths = HeR(features.T, paths, feature)
            return rerank_paths[:recall_num], coarse_scores[:recall_num]

        return paths, coarse_scores

    def save_query_res(self, image_path, similar_paths, dist, save_path, show=False):
        im = cv2.imread(image_path.encode('utf-8', 'surrogateescape').decode('utf-8'))
        image_id = image_path.split("/")[-1].split(".")[0]
        plt.figure(image_id, figsize=(14, 13))
        # gray()
        plt.subplot(5, 4, 1)
        plt.imshow(im[:, :, ::-1])
        plt.axis('off')
        i = 0
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for similar_path, score in zip(similar_paths, dist):
            img = Image.open(similar_path)
            plt.gray()
            plt.subplot(5, 4, i + 5)
            plt.title('{}'.format('%.2f' % score))
            plt.imshow(img)
            plt.axis('off')
            i += 1
        plt.savefig(save_path + os.path.sep + str(image_id) + '.jpg')
        if show:
            plt.show()


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
