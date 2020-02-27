from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
import sys
from PIL import Image
import numpy as np
import torch

# normalize = transforms.Normalize()
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # normalize,
])


def image_loader_ms(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [550, 800, 1050]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        images.append(image)
    return images


def image_loader(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [800]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [

                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        images.append(image)
    return images[0]


def get_transform():
    """
    transform the image
    :param image: the pillow format
    :return:
    """
    return image_loader


class DirDataset(Dataset):
    """
    the id is the file name, the file under the dir is all need collected in to the dataset
    """

    def __init__(self, root_dir, nums):
        super(DirDataset, self).__init__()
        self.root_dir = root_dir
        self.nums = nums
        self.image_paths = self.get_image_paths(self.root_dir)

    def __getitem__(self, index):
        try:
            image_path = self.image_paths[index]
            trans = get_transform()
            return trans(image_path), image_path
        except Exception as e:
            return torch.zeros(0), "error_path"

    def __len__(self):
        return len(self.image_paths)

    def get_image_paths(self, root_dir):
        image_path_list = []
        files = os.listdir(root_dir)
        for file in files:
            file = os.path.sep.join([root_dir, file])
            if os.path.isfile(file):
                image_path_list.append(file)
            else:
                image_path_list.extend(self.get_image_paths(file))
        return image_path_list[:self.nums]


def get_dataset(root_dir, nums):
    """
    get the dataset of the iamges under the root_dir
    :param nums:
    :param root_dir:
    :return:
    """
    return DirDataset(root_dir, nums)


def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def get_dataloader(dataset):
    """

    :param dataset:
    :return:
    """
    return DataLoader(
        dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=False, sampler=None,
        drop_last=False, collate_fn=collate_tuples
    )
