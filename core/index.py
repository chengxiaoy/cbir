import sys
import argparse
from core.network import get_model
from core.preprocess import get_dataloader, get_dataset
from core.encode import get_feature_map, extract_vector
import joblib
import json
from sklearn.preprocessing import normalize
import torch
import time
from core.helper import batch_extract, get_mean
from sklearn.decomposition import PCA
from core.search import Search
import numpy as np
from preprecess.file_helper import get_image_paths
from core.validation import valid
from core.helper import partIndex
# from multiprocessing import Pool
from torch.multiprocessing import Pool
import multiprocessing as mp
import math
import os

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
parser.add_argument("--vector_len", '-l', default=512, type=int)

parser.add_argument('--id', '-i', default="5")

args = parser.parse_args()

# if args.encoder == 'hew':
#     data_set = get_dataset(args.dir, 20000, args=args)
#     data_loader = get_dataloader(data_set)
#     mean_vector = get_mean(model, data_loader, device)
#     joblib.dump(mean_vector, 'hew_means.pkl')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    print(json.dumps(args.__dict__))

    # slice_n = 100000
    # # index the file
    # features = np.zeros((1, 2048))
    # paths = []
    #
    # p = Pool(round(args.num / slice_n))
    #
    # pool_result = []
    # for i in range(round(args.num / slice_n)):
    #
    #     r = p.apply_async(partIndex, (args, i * slice_n, (i + 1) * slice_n,))
    #     pool_result.append(r)
    # p.close()
    # p.join()
    #
    # for r in pool_result:
    #     vectors, paths_ = r.get()
    #     features = np.concatenate((features, vectors)).astype(np.float32)
    #     paths.extend(paths_)
    #
    # features = features[1:]

    #
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)
    slice_num = 100000

    pca_path = args.id + "_"+str(args.vector_len)+"pca.pkl"
    vectors_path = args.id + "vectors.pkl"
    vectors_ori_path = args.id + "vectors_ori.pkl"
    # for i in range(math.ceil(args.num / slice_num)):
    #     data_set = get_dataset(args.dir, i * slice_num, (i + 1) * slice_num, args=args)
    #     data_loader = get_dataloader(data_set)
    #     vectors, paths = batch_extract(model, data_loader, device, args)
    #     # vectors, paths = joblib.load(args.id +"vectors.pkl")
    #     # #
    #
    #     if os.path.exists(vectors_ori_path):
    #         vectors_, paths_ = joblib.load(vectors_ori_path)
    #         vectors = np.concatenate((vectors, vectors_)).astype(np.float32)
    #         paths.extend(paths_)
    #
    #     joblib.dump((vectors, paths), vectors_ori_path)

    vectors_ori, paths = joblib.load(vectors_ori_path)

    if args.pca:
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
        else:
            pca = PCA(args.vector_len, whiten=True)
            pca.fit(vectors_ori[:100000])
        vectors = pca.transform(vectors_ori)

        joblib.dump(pca, pca_path)
        joblib.dump((vectors, paths), vectors_path)

    features_path = vectors_path if args.pca else vectors_ori_path

    mAP = valid(args=args, features_path=features_path, pca_path=pca_path)

    print("map is {}".format(mAP))

# 1 resnet50 + rpool + mac + sum
# 2 dla34 + rpool + mac +sum
# 3 resnet34 + rpool + mac + sum
# 4 dla102 + rpool + mac + sum
# 5 resnet50  + hew
# 6 eff_net + rpool + mac +sum
# 7 resnet50 + rpool + mac + sum + ms
# 8 resnet101 + rpool + mac + sum
# 9 resnet50 + rpool + gem + sum
# 10 attention
# 12 resnet50 + hew + her + 50W
# 13 resnet50 +  rpool + mac + sum + ms + pca + 50W
# 14 attention + pca + 50W
# 15 attention + pca + 100W
# 16 resnet50 +  rpool + mac + sum + ms + + pca + 100W
# 17 attention + pca + 100W + 5wpca
# 18 resnet50 +  rpool + mac + sum + ms + + 5Wpca + 100W
