import cv2
import numpy as np
import math
from multiprocessing import Pool
import os
import time
from lib.log import log


def get_rerank_score_multiprocess(query_image, coarse_top_images):
    pool_num = 5
    p = Pool(pool_num)
    sub_size = len(coarse_top_images) // pool_num
    result = []
    for i in range(pool_num):
        if i == pool_num - 1:
            result.append(p.apply_async(computer_spatial_score,
                                        args=(query_image, coarse_top_images[i * sub_size:],)))
        else:
            result.append(p.apply_async(computer_spatial_score,
                                        args=(query_image, coarse_top_images[i * sub_size:(i + 1) * sub_size],)))
    p.close()
    p.join()
    scores = []
    for res in result:
        scores.extend(res.get())
    return scores


def get_rerank_score(query_image, coarse_top_images):
    return computer_spatial_score(query_image, coarse_top_images)


def get_resize_pic(image_path):
    orig_image = cv2.imread(image_path)
    height, width = orig_image.shape[:2]
    max_length = max(height, width)
    max_dest_length = 1000
    # 过小的图片不缩放
    if max_length < max_dest_length:
        return orig_image
    else:
        ratio = max_dest_length / max_length
        new_height = ratio * height
        new_width = ratio * width
        return cv2.resize(orig_image, (int(new_height), int(new_width)), interpolation=cv2.INTER_CUBIC)



def computer_spatial_score(orig_image, skewed_images):
    orig_image = cv2.imread(orig_image)
    # orig_image = get_resize_pic(orig_image)
    orig_image = np.array(orig_image)
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
    except Exception:
        surf = cv2.xfeatures2d.SIFT_create(400)
    kp1, des1 = surf.detectAndCompute(orig_image, None)
    scores = []
    for skewed_image in skewed_images:
        since = time.time()
        try:
            if not os.path.exists(skewed_image):
                scores.append(0.0)
                continue
            skewed_image = cv2.imread(skewed_image)
            # skewed_image = get_resize_pic(skewed_image)

            skewed_image = np.array(skewed_image)
            kp2, des2 = surf.detectAndCompute(skewed_image, None)
            number_count = computer_spatial(kp1, des1, kp2, des2)
            scores.append(-0.4 if number_count > 10 else 0.0)
        except Exception as e:
            log.error("an error occur in rerank {}".format(e))
            scores.append(0.0)
    return scores


def image_resize():
    pass


def computer_spatial(kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # print("good points {} ".format(len(good)))
    # if len(good) > 0.5 * len(kp1):
    #     return len(good)
    # return 0

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return 0
        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        # ss = M[0, 1]
        # sc = M[0, 0]
        # scaleRecovered = math.sqrt(ss * ss + sc * sc)
        # thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        # log.info("MAP: Calculated scale difference: %.2f, "
        #       "Calculated rotation difference: %.2f" %
        #       (scaleRecovered, thetaRecovered))

        # deskew image
        # im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M),
        #                              (orig_image.shape[1], orig_image.shape[0]))
        return mask.sum()

    else:
        # log.info("MAP: Not  enough  matches are found   -   %d/%d"
        #       % (len(good), MIN_MATCH_COUNT))
        return 0
