from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
import sys
from PIL import Image

# normalize = transforms.Normalize()
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # normalize,
])


def get_transform():
    """
    transform the image
    :param image: the pillow format
    :return:
    """
    return transform


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
        image_path = self.image_paths[index]
        image_id = image_path.split("/")[-1].split(".")[0]
        trans = get_transform()
        return trans(Image.open(image_path).convert("RGB")), image_id

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
