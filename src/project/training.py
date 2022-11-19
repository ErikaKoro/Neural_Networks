# Extra libraries
import time
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import numpy as np


def general_model_2layers(name,
                          train_X,
                          train_y,
                          testing_X,
                          test_y,
                          activation,
                          loss,
                          optimizer,
                          epochs,
                          batch_size,
                          number_of_neurons,
                          number_of_layers):
    """
    :param name: The name of the model
    :param train_X: The training set
    :param train_y: The training labels
    :param testing_X: The testing set
    :param test_y: The testing labels
    :param activation: The activation functions
    :param loss: The loss function
    :param optimizer: The optimizer
    :param epochs: The number of epochs
    :param batch_size: The batch size
    :param number_of_neurons: The number of neurons in each layer
    :param number_of_layers: The number of layers

    This function creates a model with 2 layers and the parameters are given as input
    """
    # Each layer gives output to the next layer only
    model = Sequential(name=name)
    # Dense layer is a layer where each neuron is connected to all neurons in the next layer
    # The hidden layer has 64 neurons and the input layer has 784 neurons

    for i in range(number_of_layers):
        if i == 0:
            model.add(Dense(number_of_neurons[i], activation=activation[i], input_shape=(784,)))
        else:
            model.add(Dense(number_of_neurons[i], activation=activation[i]))

    # The last layer has 10 neurons, one for each class
    # The activation function is softmax, which gives the probability of each class
    # The class with the highest probability is the predicted class
    model.summary()

    # loss function is categorical cross-entropy because the labels are one-hot encoded
    model.compile(optimizer=optimizer(learning_rate=1e-3), loss=loss, metrics=['accuracy'])

    # batch size is the number of training samples that are used to update the weights
    # Given that the batch size is 100, it means that the weights are updated after every 100 samples
    start = time.time()
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=2)
    end = time.time()
    print(f"--- {end - start} seconds ---")

    # Convert the one-hot encoded labels to the original labels
    test_y = np.argmax(test_y, axis=1)
    pred = model.predict(testing_X)
    for i in range(pred.shape[0]):
        if pred[i].argmax() == test_y[i]:
            print(f"Correct prediction: {pred[i].argmax()}")
        else:
            print(f"Wrong prediction: {pred[i].argmax()}")

    # Evaluate the model with train dataset
    print('\n\nEvaluation train dataset process:')
    evaluate_train = model.evaluate(X_train, y_train, verbose=1)
    print('Train loss:', evaluate_train[0])
    print('Train accuracy:', evaluate_train[1])
    # Evaluate the model with test dataset
    print('\n\nEvaluation test dataset process:')
    evaluate_test = model.evaluate(test_X, test_y, verbose=1)
    # model.save('seq_1relu64_2softmax10_SGD')
    print('Test loss:', evaluate_test[0])


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    test_X = X_test.reshape(-1, 28 * 28) / 255.0

    # One hot representation of the labels
    # Each label is represented as a vector of 10 elements
    # The index of the element that is 1 represents the class
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    activation_functions = ['relu', 'relu', 'softmax']
    general_model_2layers(name="1relu64_2softmax10_SGD",
                          train_X=X_train,
                          train_y=y_train,
                          testing_X=test_X,
                          test_y=y_test,
                          number_of_neurons=[64, 10, 10],
                          number_of_layers=3,
                          activation=activation_functions,
                          loss='categorical_crossentropy',
                          optimizer=SGD,
                          epochs=10,
                          batch_size=32
                          )

