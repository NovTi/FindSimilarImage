import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import shutil
import os

from collections import defaultdict
from einops import rearrange


def list_all_files(rootdir):
    extention = ['.jpg', '.png', '.JPG']
    _files = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir + '/', list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            if '/find' not in path and path[-4:] in extention:
                _files.append(path)
    return _files


def get_square_position(x, y, width, height):
    result = 0
    if (x <= height / 3):
        result += 0
    elif (x <= 2 * height / 3):
        result += 3
    else:
        result += 6

    if (y <= width / 3):
        result += 0
    elif (y <= 2 * width / 3):
        result += 1
    else:
        result += 2
    return result


def make_pixel_bucket(image):
    check = np.zeros((8, 8, 8, 9))
    pixel_bucket = defaultdict(int)
    h, w, _ = image.shape
    for i in range(0, h, 150):
        for j in range(0, w, 150):
            square_position = get_square_position(i, j, h, w)
            pixel = image[i][j]
            # >> 5 equals // 5(integer division of 5)
            check[pixel[0] >> 5][pixel[1] >> 5][pixel[2] >> 5][square_position] += 1
    return check


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def get_target_array(search_dir):
    search_img_list = list_all_files(search_dir)
    all_target_vector_lst = []
    #     hw_info_lst = []
    for i in search_img_list:
        target = cv_imread(i)
        #         hw_info_lst.append()
        target_vector = rearrange(make_pixel_bucket(target), 'a b c d -> (a b c d)')
        all_target_vector_lst.append(target_vector)
    all_target_vector = np.stack(all_target_vector_lst, axis=0)
    np.savetxt('target_image_array.txt', all_target_vector, fmt='%d')

    with open('name_info.json', 'w') as f:
        json.dump(search_img_list, f)


def get_source_array(source_img_list):
    all_source_vector_lst = []
    for i in source_img_list:
        source = cv_imread(i)
        source_vector = rearrange(make_pixel_bucket(source), 'a b c d -> (a b c d)')
        all_source_vector_lst.append(source_vector)
    all_source_vector = np.stack(all_source_vector_lst, axis=0)
    return all_source_vector


def update_check(search_dir):
    with open('name_info.json', 'r') as openfile:
        name_info = json.load(openfile)
    for i in name_info:
        if not os.path.exists(i):
            return True
    return False


def find_similar(search_dir, source_dir):
    if not os.path.exists('target_image_array.json'):
        get_target_array(search_dir)

    need_update = update_check(search_dir)

    if need_update:
        get_target_array(search_dir)

    source_img_list = list_all_files(source_dir)
    # create store folder
    if not os.path.exists(source_dir + 'find/'):
        os.mkdir(source_dir + 'find/')

    source_array = get_source_array(source_img_list)
    target_array = np.loadtxt('target_image_array.txt', dtype=int)

    with open('name_info.json', 'r') as openfile:
        target_name_info = json.load(openfile)

    similar_score = np.matmul(source_array, target_array.T)

    for i in range(source_array.shape[0]):
        # make dir
        path = source_img_list[i].split('/')
        path[-1] = 'find'
        path = '/'.join(path)
        if not os.path.exists(path):
            os.mkdir(path)

        index = similar_score[i].argmax()
        target_file_name = target_name_info[index]
        shutil.copy(target_file_name, path)


if __name__ == "__main__":
    find_similar('./search/', './source/')
