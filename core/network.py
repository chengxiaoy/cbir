from torchvision.models import resnet50,resnet34
from efficientnet_pytorch import EfficientNet
from core.models.dlav0 import dla34
from torch import nn
import torch



models = {
    "resnet50": nn.Sequential(*list(resnet50(pretrained=True).children())[:-2]),
    "resnet34": nn.Sequential(*list(resnet34(pretrained=True).children())[:-2]),
    "eff-net": nn.Sequential(*list(EfficientNet.from_pretrained('efficientnet-b0').children())[:-2]),
    'dla34': nn.Sequential(*list(dla34(True).children())[:-2]),
    'attention': None,

}


def get_model(model_name):
    """
    return the respect model for the model_name
    :param model_name:
    :return:
    """
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
    test('resnet50')
