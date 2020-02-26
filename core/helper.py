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

                fm = get_feature_map(img, model)
                vectors = extract_vector(fm, args.encoder, args.rpool, args.aggregate)
                # vectors = normalize(vectors)

                ids = [id] * len(vectors)
                indexed_vectors.extend(vectors)
                indexed_ids.extend(ids)
                print("cost {} s".format(time.time() - since))
            except Exception as e:
                print(e)

    return indexed_vectors, indexed_ids


def get_pca_whitening(self, image_paths, n_components, whitening=True):
    _, features = self.model[kind].extract_feature_batch(image_paths)

    pca = PCA(n_components, whiten=whitening)
    pca.fit(features)
    return pca
