import requests
import os
# from lib.log import log
from config.config import *


def download_csvline(line):
    id = int(line[0])
    url = line[1]
    dir_path = config['local']['pic_path'] + os.path.sep + "data" + str(id // 20000)
    try:
        download_image(id, url, file_dir=dir_path, proxy=False)
    except Exception as e:
        print("download id: {},url: {} error {}".format(id, url, e))


def download_image(id, url, file_dir, proxy=False):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
    }

    proxies = {
        "http": "socks5://127.0.0.1:1080",
        'https': 'socks5://127.0.0.1:1080'
    }

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    filename = file_dir + os.path.sep + str(id)

    # 不再使用缓存
    if os.path.exists(filename):
        # log.warn("filepath {} is already exits!".format(filename))
        return filename
    if proxy:
        response = requests.get(url, stream=True, headers=headers, timeout=5, proxies=proxies, verify=False)
    else:
        response = requests.get(url, stream=True, headers=headers, timeout=10, verify=False)

    print((url,response.__getstate__()['status_code']))

    assert response.__getstate__()['status_code'] == 200, 'status_code 不等于200'

    with open(filename, 'wb') as file:
        # 每128个流遍历一次
        for data in response.iter_content(256):
            # 把流写入到文件，这个文件最后写入完成就是，selenium.png
            file.write(data)  # data相当于一块一块数据写入到我们的图片文件中

    return filename


if __name__ == '__main__':
    download_image(3, "http://img.hb.aicdn.com/06a756222f6ece273dec792232a5051e2c9822b4f6af8-unELWo", "hehe")
