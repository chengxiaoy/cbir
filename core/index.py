import sys
import argparse
from core.network import get_model
from core.preprocess import get_dataloader, get_dataset
from core.encode import get_feature_map, extract_vector
import joblib
import json
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser(description="index images")

parser.add_argument("--dir", '-d', default="../test/", required=False, help="the dir need to be indexed")
parser.add_argument("--encoder", '-e', default='mac', required=False,
                    choices=['gem', 'crow', 'spoc', 'mac', 'hew'],
                    help='the encoder method for feature_map to vector')
parser.add_argument("--aggregate", '-a', default='gmp', required=False, choices=['sum', 'gmm', 'gmp'])
parser.add_argument("--rpool", '-r', action='store_false', help="region pool")
parser.add_argument("--model", '-m', default='resnet50', required=False,
                    choices=['resnet50', 'dla34', 'eff_net', 'attention'],
                    help='which model as the backbone')

args = parser.parse_args()
print(json.dumps(args.__dict__))
model = get_model(args.model)
data_set = get_dataset(args.dir)
data_loader = get_dataloader(data_set)

indexed_vectors = []
indexed_ids = []

for imgs, ids in data_loader:
    for id, img in zip(ids, imgs):
        fm = get_feature_map(img, model)
        vectors = extract_vector(fm, args.encoder, args.rpool, args.aggregate)
        # vectors = normalize(vectors)

        ids = [id] * len(vectors)
        indexed_vectors.extend(vectors)
        indexed_ids.extend(ids)

joblib.dump((indexed_ids, indexed_vectors), 'vectors.pkl')
