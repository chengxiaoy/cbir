import joblib
import os
import torch
from core.validation import Evaluate
from attrdict import AttrDict
from core.helper import extract
from core.preprocess import *
from core.network import get_model
from core.search import Search

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





image_helper = ImageHelper(1024, np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :,
                                 None, None])


def get_feature(args, image_path):
    if args.model == 'attention':
        img = Variable(torch.from_numpy(image_helper.load_and_prepare_image(image_path)))
    else:

        trans = get_transform(args)
        img = trans(image_path)
    model = get_model(args.model)
    model.to(device)
    img.to(device)
    return extract(model, img, args, device)


if __name__ == '__main__':
    # args = AttrDict({"model": "attention", "pca": False, "multi_scale": False, 'id': "10",'rerank':"none"})
    # feature2 = get_feature(args, "../bgy_test/2/137-2.jpg")
    # feature1 = get_feature(args, "../bgy_test/1/137-1.jpg")
    # print(np.dot(feature2[0], feature1[0]))
    # print("===feature1====")
    # print(feature1)
    # model = get_model(args.model).to(device)
    # search = Search(model, "10vectors_.pkl", "none", device, args)
    # print(search.search("../bgy_test/1/137-1.jpg", 10))
    res_show("data/10query_res.pkl")
