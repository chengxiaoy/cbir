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

parser = argparse.ArgumentParser(description="index images")

parser.add_argument("--dir", '-d', default="../test/1", required=False, help="the dir need to be indexed")
parser.add_argument("--num", '-n', default=200000, required=False)
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

parser.add_argument('--id', '-i', default="1")

args = parser.parse_args()
print(json.dumps(args.__dict__))
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

model = get_model(args.model)
model = model.to(device)

if args.encoder == 'hew':
    data_set = get_dataset(args.dir, 20000, args=args)
    data_loader = get_dataloader(data_set)
    mean_vector = get_mean(model, data_loader, device)
    joblib.dump(mean_vector, 'hew_means.pkl')

# index the file

data_set = get_dataset(args.dir, args.num, args=args)
data_loader = get_dataloader(data_set)

vectors, paths = batch_extract(model, data_loader, device, args)
# vectors, paths = joblib.load("vectors.pkl")
#
if args.pca:
    pca = PCA(512, whiten=True)
    pca.fit(vectors[:20000])
    vectors = pca.transform(vectors)

    joblib.dump(pca, args.id + "pca.pkl")
joblib.dump((vectors, paths), args.id + "vectors.pkl")

mAP = valid(model, args=args, device=device, features_path=args.id + "vectors.pkl", pca_path=args.id + 'pca.pkl')

print("map is {}".format(mAP))

# 1 resnet50 + rpool + mac + sum
# 2 dla34 + rpool + mac +sum
# 3 resnet34 + rpool + mac + sum
# 4 dla102 + rpool + mac + sum
# 5 resnet50  + hew
# 6 eff_net + rpool + mac +sum
# 7 resnet50 + rpool + mac + sum + ms
# 8 resnet101 + rpool + mac + sum
