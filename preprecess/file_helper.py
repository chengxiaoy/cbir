import os
import shutil


def get_image_paths(root_dir):
    image_path_list = []
    files = os.listdir(root_dir)
    for file in files:
        file = os.path.sep.join([root_dir, file])
        if os.path.isfile(file):
            image_path_list.append(file)
        else:
            image_path_list.extend(get_image_paths(file))
    return image_path_list


def get_label(path):
    return path.split('/')[-1].split('.')[0].split('-')[-1]


def get_id(path):
    return path.split('/')[-1].split('.')[0].split('-')[0]


def get_filename(path):
    return path.split('/')[-1]


dst_dir = '../test'
src_dir = '/Users/tezign/Desktop/bgy_test'
image_paths = get_image_paths(src_dir)
image1_paths = list(filter(lambda x: get_label(x) == '1', image_paths))
image2_paths = list(filter(lambda x: get_label(x) == '2', image_paths))

path_dir = {"1": image1_paths, "2": image2_paths}
for key in path_dir:
    paths = path_dir[key]

    for image_path in paths:
        dst_sub_dir = os.path.sep.join((dst_dir, key))
        if not os.path.exists(dst_sub_dir):
            os.mkdir(dst_sub_dir)
        shutil.move(image_path, os.path.sep.join((dst_sub_dir, get_filename(image_path))))
