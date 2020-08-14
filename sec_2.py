from sklearn import svm, preprocessing
import numpy as np
from matplotlib.image import imread

train_spots = []
test_ones = []
classes = []
train_counts = 7291
train_indices = [1, 1195, 2200, 2931, 3589, 4241, 4797, 5461, 6106, 6648, 7291]
test_counts = 2007
test_indices = [1, 360, 624, 822, 988, 1188, 1348, 1518, 1665, 1831, 2008]


def train():
    for index in range(0, len(train_indices) - 1):
        for k in range(train_indices[index], train_indices[index + 1]):
            image = imread('usps/train/' + str(index) + '_' + str(k) + '.jpg')
            train_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    train_item.append(image[i][j])
            train_item = preprocessing.scale(train_item)
            train_spots.append(train_item)
            classes.append(index)


def init_svm():
    print("init svm")
    clf = svm.SVC()
    clf.fit(np.array(train_spots), classes)
    return clf


def test(clf: svm.SVC):
    print("init_test")
    print(clf.get_params())
    error_percentage = [0] * len(test_indices)
    for index in range(0, len(test_indices) - 1):
        for k in range(test_indices[index], test_indices[index + 1]):
            image = imread('usps/test/' + str(index) + '_' + str(k) + '.jpg')
            test_item = []
            for i in range(0, 16):
                for j in range(0, 16):
                    test_item.append(image[i][j])
            if index not in clf.predict([preprocessing.scale(test_item)]):
                error_percentage[index] += 1
                error_percentage[len(test_indices) - 1] += 1
        error_percentage[index] /= (test_indices[index + 1] - test_indices[index])
        error_percentage[index] *= 100
    error_percentage[len(test_indices) - 1] /= test_counts / 100
    return error_percentage


train()
clf = init_svm()
error_percentage = test(clf)
print(error_percentage)
