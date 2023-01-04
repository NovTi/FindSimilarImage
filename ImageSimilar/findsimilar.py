import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import shutil
import os
from tqdm import tqdm
import pdb

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
    h, w, _ = image.shape
    h1 = h // 100
    w1 = w // 100
    for i in range(h1, h, h1):
        for j in range(w1, w, w1):
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
    for i in tqdm(range(len(search_img_list))):  # for i in search_img_list:
        try:
            target = cv_imread(search_img_list[i])
            target_vector = rearrange(make_pixel_bucket(target), 'a b c d -> (a b c d)')
        except:
            print('Cannot load the target image:', search_img_list[i])
            target_vector = np.zeros(4608)
        all_target_vector_lst.append(target_vector)
    all_target_vector = np.stack(all_target_vector_lst, axis=0)
    np.savetxt('target_image_array.txt', all_target_vector, fmt='%d')

    with open('name_info.json', 'w') as f:
        json.dump(search_img_list, f)


def get_source_array(source_img_list):
    all_source_vector_lst = []
    for i in tqdm(range(len(source_img_list))):  # for i in source_img_list:
        try:
            source = cv_imread(source_img_list[i])
            source_vector = rearrange(make_pixel_bucket(source), 'a b c d -> (a b c d)')
        except:
            print('Cannot load the source image:', source_img_list[i])
            source_vector = np.zeros(4608)
        all_source_vector_lst.append(source_vector)
    all_source_vector = np.stack(all_source_vector_lst, axis=0)

    return all_source_vector


def update_check(search_dir):
    with open('name_info.json', 'r') as openfile:
        name_info = json.load(openfile)
    # print(name_info)
    current_files = list_all_files(search_dir)
    missing_index = []
    missing_vector = []
    missing_file_name = []
    need_update = False
    # pdb.set_trace()
    for i in tqdm(range(len(current_files))):
        if current_files[i] not in name_info:
            need_update = True
            missing_index.append(i)
            missing_target = cv_imread(current_files[i])
            missing_file_name.append(current_files[i])
            missing_target_vector = rearrange(make_pixel_bucket(missing_target), 'a b c d -> (a b c d)')
            missing_vector.append(missing_target_vector)
    return need_update, missing_index, missing_vector, missing_file_name


def update(missing_index, missing_vector):
    old_target_array = np.loadtxt('target_image_array.txt', dtype=int)
    old_target_array = list(old_target_array)
    for i in tqdm(range(len(missing_index))):
        old_target_array.insert(missing_index[i], missing_vector[i])
    all_target_vector = np.stack(old_target_array, axis=0)
    np.savetxt('target_image_array.txt', all_target_vector, fmt='%d')


def find_similar(search_dir, source_dir):
    print('==== Checking if generating target array needed ====')
    if not os.path.exists('target_image_array.txt'):
        print('======= Getting the target image array =======')
        get_target_array(search_dir)
        print('\n')
    else:
        print('======= Target array already existed =======\n')

    print('======= Check if updating needed =======')
    need_update, missing_index, missing_vector, missing_file_name = update_check(search_dir)

    if need_update:
        print('======= Need update and updating =======')
        update(missing_index, missing_vector)
        print('\n')
    else:
        print('\n========== No updating needed ==========\n')

    print('======= Getting the source image names =======')
    source_img_list = list_all_files(source_dir)
    print('============= Done =============\n')

    # # create store folder
    # if not os.path.exists(os.path.join(source_dir + 'find/')):
    #     os.mkdir(os.path.join(source_dir + 'find/'))

    print('======= Getting the source image array =======')
    source_array = get_source_array(source_img_list)
    target_array = np.loadtxt('target_image_array.txt', dtype=int)

    with open('name_info.json', 'r') as openfile:
        target_name_info = json.load(openfile)

    similar_score = np.matmul(source_array, target_array.T)

    print('\n\n========== Comparing the Images ==========')
    for i in tqdm(range(source_array.shape[0])):  # for i in range(source_array.shape[0]):
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
    # target file path, source file path
    find_similar('./testimage/source', './testimage/target')
