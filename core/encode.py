from core.layers.pooling import MAC, RMAC, SPoC, GeM, Rpool, Hew
import torch
import torch.nn as nn

POOLING = {
    'mac': MAC,
    'spoc': SPoC,
    'gem': GeM,
    'r-mac': RMAC,
    'hew': Hew,
}


def get_feature_map(image_tensor, model):
    """
    image(pil) ---> feature_map
    :param model:
    :param image:
    :return:
    """

    # image_tensor = torch.unsqueeze(image_tensor, 0)
    return model(image_tensor)


def extract_vector(feature_map, encode_type, rpool=False, aggregate='sum'):
    """
    feature_map ---> vector or vectors
    :param encode_type:
    :param feature_map:
    :return:
    """
    if not rpool:
        vector = POOLING[encode_type]()(feature_map)
    else:
        vector = Rpool(POOLING[encode_type]()).forward(feature_map, aggregate=aggregate)
    return vector
