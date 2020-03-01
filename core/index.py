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
parser.add_argument("--gpu", '-g', default=0, choices=[0, 1])
parser.add_argument("--encoder", '-e', default='hew', required=False,
                    choices=['gem', 'crow', 'spoc', 'mac', 'hew'],
                    help='the encoder method for feature_map to vector')
parser.add_argument("--aggregate", '-a', default='sum', required=False, choices=['sum', 'gmm', 'gmp'])
parser.add_argument("--rpool", '-r', action='store_true', help="region pool")
parser.add_argument("--model", '-m', default='dla34', required=False,
                    choices=['resnet50', 'resnet34', 'dla34', 'eff_net', 'attention'],
                    help='which model as the backbone')

args = parser.parse_args()
print(json.dumps(args.__dict__))
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

model = get_model(args.model)
model = model.to(device)

# if args.encoder == 'hew':
#     data_set = get_dataset(args.dir, 20000, args=args)
#     data_loader = get_dataloader(data_set)
#     mean_vector = get_mean(model, data_loader, device)
#     joblib.dump(mean_vector, 'hew_means.pkl')

# index the file

data_set = get_dataset(args.dir, args.num, args=args)
data_loader = get_dataloader(data_set)

vectors, paths = batch_extract(model, data_loader, device, args)

pca = PCA(512, whiten=True)
pca.fit(vectors[:20000])
vectors = pca.transform(vectors)

joblib.dump((vectors, paths), "vectors.pkl")
# joblib.dump(pca, "pca.pkl")

mAP = valid(model, args=args, device=device, features_path="vectors.pkl", pca_path='pca.pkl')

print("map is {}".format(mAP))

# resnet50 +sum is 0.68575
# resnet34 + gmp 0.685
# resnet34 + sum 0.625
# resnet34 + hew 0.696
