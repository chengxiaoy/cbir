from uuid import uuid1
import requests
import os


def download_pic(url, save_path, id):
    headers = {
        'User-Agent': 'Mozilla/4.0(compatible;MSIE 5.5;Windows NT)', }
    response = requests.get(url, stream=True, headers=headers, timeout=10)

    items = url.split(".")
    ext = items[len(items) - 1]
    with open(save_path + os.path.sep + str(id), 'wb') as file:
        # 每128个流遍历一次
        for data in response.iter_content(128):
            # 把流写入到文件，这个文件最后写入完成就是，selenium.png
            file.write(data)  # data相当于一块一块数据写入到我们的图片文件中
    return response.status_code


def remove_pic(save_path, id):
    os.remove(save_path + os.path.sep + str(id))
