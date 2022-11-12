from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
import time
import matplotlib.pyplot as plt


def calculate_accuracy(pred, test_labels):
    correct = 0
    for i in range(len(test_labels)):
        if pred[i] == test_labels[i]:
            correct += 1
    return correct / len(test_labels)


def knn_classification(train_set, train_labels, test_set, test_labels, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, train_labels)
    pred = knn.predict(test_set)
    report_knn = classification_report(test_labels, pred)
    print(report_knn)
    acc = calculate_accuracy(pred, test_labels)
    print(f"Accuracy: {acc}")


def image_show(i, data, label):
    x = data[i]  # get the vectorized image
    x = x.reshape((28, 28))  # reshape it into 28x28 format
    print('The image label of index %d is %d.' % (i, label[i]))
    plt.imshow(x, cmap='gray')  # show the image
    plt.show()


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = train_X.reshape(-1, 28 * 28) / 255.0
    test_X = test_X.reshape(-1, 28 * 28) / 255.0

    # image_show(200, train_X, train_y)

    start_time = time.time()
    knn_classification(train_X, train_y, test_X, test_y)
    print("--- %s seconds ---" % (time.time() - start_time))