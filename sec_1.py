import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

a_heavy_side = -4
b_heavy_side = 4
a_poly_2 = -5
b_poly_2 = 5
a_homo = 0.01
b_homo = 4
a_sin = - math.pi
b_sin = math.pi
gen_data_counts = 100
test_percentage = 0.5
ja = 1
nein = -1


# y = 0.5
def rand_heavy_side(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_heavy_side, b_heavy_side)
        y = math.floor(1 + np.sign(x) * 0.5)
        if not [x, y] in X:
            X.append([x, y])
            if y >= 0.5:
                Y.append(ja)
            else:
                Y.append(nein)
        else:
            i -= 1
    return X, Y


# y = x
def rand_homo(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_homo, b_homo)
        y = 1 / x
        if not [x, y] in X:
            X.append([x, y])
            if y >= x:
                Y.append(ja)
            else:
                Y.append(nein)
        else:
            i -= 1
    return X, Y


# x = 0
def rand_poly_2(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_poly_2, b_poly_2)
        y = x ** 2 + 2
        if not [x, y] in X:
            X.append([x, y])
            if x >= 0:
                Y.append(ja)
            else:
                Y.append(nein)
        else:
            i -= 1
    return X, Y


# y = 0
def rand_sin(n: int):
    X = []
    Y = []
    for i in range(0, n):
        x = random.uniform(a_sin, b_sin)
        y = math.sin(x)
        if not [x, y] in X:
            X.append([x, y])
            X.append([x, -y])
            Y.append(ja)
            Y.append(nein)
        else:
            i -= 1
    return X, Y


def test_heavy_side():
    X_train, Y_train = rand_heavy_side(gen_data_counts)
    X_test, Y_test = rand_heavy_side(int(gen_data_counts * test_percentage))
    X_train, X_test = np_array(X_train, X_test)
    clf = train_clf(X_train, Y_train, 'linear')
    predict = clf.predict(X_test)
    draw_plot(X_train, Y_train, X_test, Y_test, predict, clf)


def test_homo():
    X_train, Y_train = rand_homo(gen_data_counts)
    X_test, Y_test = rand_homo(int(gen_data_counts * test_percentage))
    X_train, X_test = np_array(X_train, X_test)
    clf = train_clf(X_train, Y_train, 'linear')
    predict = clf.predict(X_test)
    draw_plot(X_train, Y_train, X_test, Y_test, predict, clf)


def test_poly_2():
    X_train, Y_train = rand_poly_2(gen_data_counts)
    X_test, Y_test = rand_poly_2(int(gen_data_counts * test_percentage))
    X_train, X_test = np_array(X_train, X_test)
    clf = train_clf(X_train, Y_train, 'linear')
    predict = clf.predict(X_test)
    draw_plot(X_train, Y_train, X_test, Y_test, predict, clf)


def test_sin():
    X_train, Y_train = rand_sin(gen_data_counts)
    X_test, Y_test = rand_sin(int(gen_data_counts * test_percentage))
    X_train, X_test = np_array(X_train, X_test)
    clf = train_clf(X_train, Y_train, 'rbf')
    predict = clf.predict(X_test)
    draw_plot(X_train, Y_train, X_test, Y_test, predict, clf)


def np_array(X_train, X_test):
    return np.array(X_train), np.array(X_test)


def train_clf(X_train, Y_train, kernel: str):
    clf = svm.SVC(kernel=kernel, C=10)
    clf.fit(X_train, Y_train)
    return clf


def draw_plot(X_train, Y_train, X_test, Y_test, predict, clf: svm.SVC):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
    draw_lines(clf)
    plt.show()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predict)
    draw_lines(clf)
    plt.show()


def draw_lines(clf: svm.SVC):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')


test_heavy_side()
test_homo()
test_poly_2()
test_sin()