from core.encode import get_feature_map, extract_vector

import time
from sklearn.decomposition import PCA

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
