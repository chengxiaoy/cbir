from core.encode import get_feature_map, extract_vector
from tqdm import tqdm
import time
from sklearn.decomposition import PCA
from core.layers.functional import get_potential_inv_re, create_mean
import torch
import numpy as np
from sklearn.preprocessing import normalize
from core.network import get_model
from core.preprocess import get_dataloader,get_dataset


def partIndex(args, start_n, end_n):
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    model = get_model(args.model)
    model = model.to(device)
    data_set = get_dataset(args.dir, start_n, end_n, args=args)
    data_loader = get_dataloader(data_set)
    vectors, paths = batch_extract(model, data_loader, device, args)
    return vectors, paths


def get_mean(model, data_loader, device):
    fm_list = []
    for imgs, paths in data_loader:
        for path, img in zip(paths, imgs):
            try:
                since = time.time()

                if path == 'error_path':
                    print("get {} image feature failed!".format(path))
                    continue
                img = img.to(device)

                fm = get_feature_map(img, model)
                fm_list.append(fm.squeeze(0).detach().cpu().numpy().T)
                # print("get fm cost {} s".format(time.time() - since))


            except Exception as e:
                print(e)

    return create_mean(fm_list)


def batch_extract(model, data_loader, device, args):
    indexed_vectors = []
    indexed_ids = []

    for imgs, paths in tqdm(data_loader):
        for path, img in zip(paths, imgs):
            try:
                if path == 'error_path':
                    # print("get {} image feature failed!".format(path))
                    continue
                since = time.time()

                vectors = extract(model, img, args, device)
                if isinstance(vectors, torch.Tensor):
                    vectors = vectors.detach().cpu().numpy()
                # vectors = normalize(vectors)

                paths = [path] * len(vectors)
                indexed_vectors.extend(vectors)
                indexed_ids.extend(paths)
                # print("cost {} s".format(time.time() - since))
            except Exception as e:
                print(e)

    return indexed_vectors, indexed_ids


def extract(model, img_tensor, args, device):
    vector_list = []
    if args.multi_scale:
        for img in img_tensor:
            img = img.to(device)
            if args.model == 'attention':
                return model(img).cpu().detach().numpy()
            fm = get_feature_map(img, model, args)
            vectors = extract_vector(fm, args.encoder, args.rpool, args.aggregate, device)
            vector_list.extend(vectors)

        return normalize(np.array([np.array(vector_list, dtype=np.float32).sum(axis=0)]))

    else:
        img_tensor = img_tensor.to(device)

        if args.model == 'attention':
            return model(img_tensor).cpu().detach().numpy()
        fm = get_feature_map(img_tensor, model, args)
        vectors = extract_vector(fm, args.encoder, args.rpool, args.aggregate, device)
        return vectors
