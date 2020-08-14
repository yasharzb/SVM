from sklearn import svm, preprocessing
import numpy as np
import random
from matplotlib.image import imread
import glob

train_rate = 0.8
train_spots = []
train_list = []
test_ones = []
classes = []
total_counts = 300
test_counts = 2007
dir_names = ['2', '3', '7', 'S', 'W']
files = []


def import_files():
    for dir in dir_names:
        files.append(glob.glob('persian_LPR/' + str(dir) + '/*.bmp'))


def train():
    for dir in dir_names:
        index = dir_names.index(dir)
        rand_list = random.sample(range(0, total_counts + 1), int(total_counts * train_rate))
        train_list.append(rand_list)
        train_exs = []
        for item in files[index]:
            num_start = item.index(str(dir)) + 2
            num = int(item[num_start:item[num_start:].index('_') + num_start])
            if num in rand_list:
                train_exs.append(item)
        for ex in train_exs:
            image = imread(ex)
            train_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    train_item.append(image[i][j])
            train_spots.append(preprocessing.scale(train_item))
            classes.append(index)


def init_svm():
    print("init svm")
    clf = svm.SVC()
    clf.fit(np.array(train_spots), classes)
    return clf


def test(clf: svm.SVC):
    print("init_test")
    print(clf.get_params())
    error_percentage = [0] * (len(dir_names) + 1)
    for dir in dir_names:
        index = dir_names.index(dir)
        test_exs = []
        for item in files[index]:
            num_start = item.index(str(dir)) + 2
            num = int(item[num_start:item[num_start:].index('_') + num_start])
            if num not in train_list[index]:
                test_exs.append(item)
        for ex in test_exs:
            image = imread(ex)
            test_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    test_item.append(image[i][j])
            if index not in clf.predict([preprocessing.scale(test_item)]):
                error_percentage[index] += 1
                error_percentage[-1] += 1
        error_percentage[index] /= int((1 - train_rate) * total_counts)
        error_percentage[index] *= 100
    error_percentage[-1] /= len(dir_names) * int((1 - train_rate) * total_counts)
    error_percentage[-1] *= 100
    return error_percentage


import_files()
train()
clf = init_svm()
error_percentage = test(clf)
print(error_percentage)
