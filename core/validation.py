from core.search import Search
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from core.preprocess import get_dataset, get_dataloader
from core.helper import batch_extract
from sklearn.preprocessing import normalize
import joblib
import numpy as np
from preprecess.file_helper import get_image_paths


def valid(model, device, args, features_path, pca_path):
    # add the need retrieval pic
    data_set = get_dataset('../bgy_test/2', 100, args)
    data_loader = get_dataloader(data_set)

    vectors_, paths_ = batch_extract(model, data_loader, device, args)

    # pca = joblib.load(pca_path)
    # vectors_ = pca.transform(np.array(vectors_, dtype=np.float32))
    # vectors_ = normalize(vectors_)
    features, paths = joblib.load(features_path)

    features = np.concatenate((features, vectors_)).astype(np.float32)
    paths.extend(paths_)

    joblib.dump((features, paths), "vectors_.pkl")

    query_paths = get_image_paths('../bgy_test/1')
    search = Search(model, "vectors_.pkl", "pca.pkl", device, args=args)

    query_res = {}

    for query_path in query_paths:
        paths, scores = search.search(query_path, 10)
        query_res[query_path] = (paths, scores)

    joblib.dump(query_res, "query_res_att.pkl")

    eva = Evaluate("1")
    mAP = eva.mAP(query_res)
    return mAP


class Evaluate:
    def __init__(self, default_path='error.jpg'):
        self.default_path = default_path

    def get_label_info(self, path):
        files = os.listdir(path)
        label_dict = {}
        for file in files:
            if file == '.DS_Store':
                continue
            label = self.get_label(file)
            if label == 0:
                continue
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
        return label_dict

    def show(self, image_path, similar_paths, dist, save_path, show=False):
        im = cv2.imread(image_path.encode('utf-8', 'surrogateescape').decode('utf-8'))
        image_id = image_path.split("/")[-1].split(".")[0]
        plt.figure(image_id, figsize=(14, 13))
        # gray()
        plt.subplot(5, 4, 1)
        plt.imshow(im[:, :, ::-1])
        plt.axis('off')
        i = 0
        right_index = "nan"
        for similar_path, score in zip(similar_paths, dist):
            if self.get_label(image_path) == self.get_label(similar_path):
                right_index = i + 1

            if not os.path.exists(similar_path):
                similar_path = self.default_path
            img = Image.open(similar_path)
            plt.gray()
            plt.subplot(5, 4, i + 5)
            plt.title('{}'.format('%.2f' % score))
            plt.imshow(img)
            plt.axis('off')
            i += 1
        plt.savefig(save_path + str(image_id) + "_" + str(right_index) + '.jpg')
        if show:
            plt.show()
        plt.close()

    def precision(self, query_res, top_k):
        count = 0
        right_count = 0
        for query_path in query_res:
            count = count + 1
            res_list = query_res[query_path][0][:top_k]
            for res in res_list:
                if self.get_label(query_path) == self.get_label(res):
                    right_count = right_count + 1
                    break
        return right_count / count

    def mAP(self, query_res):
        count = 0
        score_sum = 0.0
        for query_path in query_res:
            query_label = self.get_label(query_path)
            if query_label == 0:
                continue
            count += 1
            retrieval_paths, _ = query_res[query_path]
            retrieval_count = 0
            score = 0.0
            for index, retrieval_path in enumerate(retrieval_paths):
                if self.get_label(retrieval_path) == query_label:
                    retrieval_count += 1
                    score += retrieval_count / (index + 1)
            if retrieval_count == 0:
                average_score = 0.0
            else:
                average_score = score / retrieval_count
            score_sum += average_score
        return score_sum / count

    def mAP_threshold(self, query_res, threshold=1.0):
        count = 0
        score_sum = 0.0
        for query_path in query_res:
            query_label = self.get_label(query_path)
            if query_label == 0:
                continue
            count += 1
            retrieval_paths, scores = query_res[query_path]
            retrieval_paths_thr = []
            for retrieval_path, score in zip(retrieval_paths, scores):
                if score < threshold:
                    retrieval_paths_thr.append(retrieval_path)
            retrieval_count = 0
            score = 0.0
            for index, retrieval_path in enumerate(retrieval_paths_thr):
                if self.get_label(retrieval_path) == query_label:
                    retrieval_count += 1
                    score += retrieval_count / (index + 1)
            if retrieval_count == 0:
                average_score = 0.0
            else:
                average_score = score / retrieval_count
            score_sum += average_score
        return score_sum / count

    def get_label(self, file_path):
        if file_path.__contains__('-'):
            return file_path.split("/")[-1].split(".")[0].split('-')[0].strip()
        elif file_path.__contains__("\\"):
            return file_path.split('/')[-1].split(".")[0].split('\\')[1].strip()
        elif file_path.__contains__('_'):
            return file_path.split('/')[-1].split('.')[0].split('_')[0].split()
        else:
            return 0


if __name__ == '__main__':
    eva = Evaluate()
    hew = joblib.load("../data/query_res_dla.pkl")
    presion = eva.precision(hew, 10)
    print("map is {}".format(eva.mAP(hew)))
    print(presion)
