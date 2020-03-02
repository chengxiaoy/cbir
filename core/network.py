from torchvision.models import resnet50, resnet34,resnet101
from efficientnet_pytorch import EfficientNet
from core.models.dlav0 import dla34,dla102x
from torch import nn
import torch
import numpy as np
from core.attention import OurNet

models = {
    "resnet50": nn.Sequential(*list(resnet50(pretrained=True).children())[:-2]),
    "resnet101": nn.Sequential(*list(resnet101(pretrained=True).children())[:-2]),
    "resnet34": nn.Sequential(*list(resnet34(pretrained=True).children())[:-2]),
    "eff-net": EfficientNet.from_pretrained('efficientnet-b0'),
    'dla34': nn.Sequential(*list(dla34(True).children())[:-2]),
    'dla102x': nn.Sequential(*list(dla102x(True).children())[:-2]),
    'attention': None,

}


def get_model(model_name):
    """
    return the respect model for the model_name
    :param model_name:
    :return:
    """
    if model_name == 'attention':
        S = 1024  # Maximum dimension
        weight_path = 'weights/ContextAwareRegionalAttention_weights.pth'
        means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]
        net = OurNet()
        net.eval()

        # Load the trained weights of regional attention network
        dic = torch.load(weight_path, map_location='cpu')
        net.region_attention.load_state_dict(dic, strict=True)
        return net
    return models[model_name].eval()


def test(model_name):
    x = torch.randn(4, 3, 64, 64)
    # model = get_model('basic')
    # y = model(x)
    # print(y.size())

    model = get_model(model_name)
    y = model(x)
    print(y.size())


if __name__ == '__main__':
    test('eff-net')
