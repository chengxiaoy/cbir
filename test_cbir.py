import joblib
import os
import torch
from core.validation import Evaluate
from attrdict import AttrDict
from core.helper import extract
from core.preprocess import *
from core.network import get_model
from core.search import Search
import argparse
from preprecess.file_helper import get_image_paths
from core.network import get_model
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from core.preprocess import Pic_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="index images")

parser.add_argument("--dir", '-d', default="../bgy_test/1", required=False, help="the dir need to be indexed")
parser.add_argument("--num", '-n', default=200000, required=False, type=int)
parser.add_argument("--gpu", '-g', default=0, choices=[0, 1], type=int)
parser.add_argument("--encoder", '-e', default='mac', required=False,
                    choices=['gem', 'crow', 'spoc', 'mac', 'hew'],
                    help='the encoder method for feature_map to vector')
parser.add_argument("--aggregate", '-a', default='sum', required=False, choices=['sum', 'gmm', 'gmp'])
parser.add_argument("--rpool", '-r', action='store_false', help="region pool")
parser.add_argument("--model", '-m', default='resnet50', required=False,
                    choices=['resnet50', 'resnet101', 'resnet34', 'dla34', 'dla102x', 'eff-net', 'attention'],
                    help='which model as the backbone')
parser.add_argument("--pca", '-p', action='store_false', help="need pca")
parser.add_argument("--multi_scale", '-s', action='store_true')
parser.add_argument("--rerank", '-k', default='none')

parser.add_argument('--id', '-i', default="5")

image_helper = ImageHelper(1024, np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :,
                                 None, None])


def get_feature(args, image_path):
    trans = Pic_transform(args)
    model = get_model(args.model)
    model.to(device)
    return extract(model, trans.transform(image_path)[0], args, device)


def rerank_test(args):
    query_paths = get_image_paths('../bgy_test/1')
    model = get_model(args.model).to(device)

    search = Search(model, args.id + "vectors_.pkl", args.id + "pca.pkl", device, args=args)

    query_res = {}

    for query_path in query_paths:
        paths, scores = search.search(query_path, 10)
        query_res[query_path] = (paths, scores)

    joblib.dump(query_res, args.id + "query_res.pkl")

    eva = Evaluate("error.jpg")
    mAP = eva.mAP(query_res)
    print("map is {}".format(round(mAP)))


if __name__ == '__main__':
    args = AttrDict(
        {"model": "attention", "rpool": True, "aggregate": "sum", "encoder": "mac", "pca": False, "multi_scale": True,
         'id': "10", 'rerank': "none"})
    feature2 = get_feature(args, "bgy_test/1/116-1.jpg")
    feature1 = get_feature(args, "bgy_test/1/27-1.jpg")

    pca_10 = joblib.load('data/10pca.pkl')
    pca_15 = joblib.load('data/15pca.pkl')
    feature2 = pca_10.transform(feature2)
    feature2 = normalize(feature2)
    feature1 = pca_10.transform(feature1)
    feature1 = normalize(feature1)
    print(np.dot(feature2[0], feature1[0]))
    print(pairwise_distances(feature2, feature1))
    # print("===feature1====")
    # print(feature1)
    # model = get_model(args.model).to(device)
    # search = Search(model, "10vectors_.pkl", "none", device, args)
    # print(search.search("../bgy_test/1/137-1.jpg", 10))
    # res_show("data/10query_res.pkl")
