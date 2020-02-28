from core.encode import get_feature_map, extract_vector

import time
from sklearn.decomposition import PCA
from core.layers.functional import get_potential_inv_re, create_mean


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
                print("get fm cost {} s".format(time.time() - since))


            except Exception as e:
                print(e)

    return create_mean(fm_list)


def batch_extract(model, data_loader, device, args):
    indexed_vectors = []
    indexed_ids = []

    for imgs, paths in data_loader:
        for path, img in zip(paths, imgs):
            try:
                if path == 'error_path':
                    print("get {} image feature failed!".format(path))
                    continue
                img = img.to(device)
                since = time.time()

                vectors = extract(model, img, args)
                # vectors = normalize(vectors)

                paths = [path] * len(vectors)
                indexed_vectors.extend(vectors)
                indexed_ids.extend(paths)
                print("cost {} s".format(time.time() - since))
            except Exception as e:
                print(e)

    return indexed_vectors, indexed_ids


def extract(model, img_tensor, args):
    fm = get_feature_map(img_tensor, model)
    vectors = extract_vector(fm, args.encoder, args.rpool, args.aggregate)
    return vectors
