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
from core.encode import get_feature_map, extract_vector
from core.network import get_model
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getFeature(image_path, encode='r-mac', rpool=False, aggregate='sum'):
    img = get_transform()(image_path).to(device)
    model = get_model('resnet50')

    fm = get_feature_map(img, model)
    vectors = extract_vector(fm, encode, rpool, aggregate)

    return vectors


if __name__ == '__main__':
    print(getFeature('test/1/116-1.jpg','r-mac',False,'sum'))