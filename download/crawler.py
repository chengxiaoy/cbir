from download.helper import download_csvline
from multiprocessing import Manager, Pool

csv_file_path = "idp_copyright_crawler_pic.csv"

f = open(csv_file_path)
lines = f.readlines()

lines = [[int(x.split(',')[0]), x.split(',')[1].strip('"')] for x in lines[1:]]

# download_lines = filter(
#     lambda x: x[1].startswith("https://file.digitaling.com/eImg") or x[1].startswith("https://img.zcool.cn"), lines)
pool_num = 50
pool = Pool(pool_num)

pool.map(download_csvline, list(lines)[:500000])
