from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
import sys
from PIL import Image
import numpy as np
import torch
import cv2
from torch.autograd import Variable

# normalize = transforms.Normalize()
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # normalize,
])


class ImageHelper:
    def __init__(self, S, means):
        self.S = S
        self.means = means

    def get_features(self, I, net, gpu_num):
        # output = net(Variable(torch.from_numpy(I).cuda(gpu_num)))
        output = net(Variable(torch.from_numpy(I)))
        output = np.squeeze(output.cpu().data.numpy())
        return output

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)

        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean

        I = im_resized.transpose(2, 0, 1) - self.means
        return I


def image_loader_ms(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [512, 800, 1024]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float) // 32 * 32).astype(np.int32))
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
        new_size = tuple(np.round(im_size_hw * ratio.astype(float) // 32 * 32).astype(np.int32))
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


def get_transform(args):
    """
    transform the image
    :param image: the pillow format
    :return:
    """
    if args.multi_scale:
        return image_loader_ms
    return image_loader


class DirDataset(Dataset):
    """
    the id is the file name, the file under the dir is all need collected in to the dataset
    """

    def __init__(self, root_dir, start_n, nums, args):
        super(DirDataset, self).__init__()
        self.root_dir = root_dir
        self.start_n = start_n
        self.nums = nums
        self.image_paths = self.get_image_paths(self.root_dir)
        self.args = args
        self.image_helper = ImageHelper(1024,
                                        np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :,
                                        None, None])

    def __getitem__(self, index):
        try:
            image_path = self.image_paths[index]
            if self.args.model == 'attention':
                return Variable(torch.from_numpy(self.image_helper.load_and_prepare_image(image_path))), image_path

            trans = get_transform(self.args)
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

        image_path_list = sorted(image_path_list)
        return image_path_list[self.start_n:self.start_n + self.nums]


def get_dataset(root_dir, start_n, nums, args):
    """
    get the dataset of the iamges under the root_dir
    :param nums:
    :param root_dir:
    :return:
    """
    return DirDataset(root_dir, start_n, nums, args=args)


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
        num_workers=4, pin_memory=True, sampler=None,
        drop_last=False, collate_fn=collate_tuples
    )
