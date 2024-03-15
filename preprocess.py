import cv2
import numpy as np
import tensorflow as tf

def preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = [cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_GRAY2RGB) for img in x_train]
    x_test = [cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_GRAY2RGB) for img in x_test]
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = preprocess_data()
    # Save preprocessed data for use in the next stage
    data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test
    }
    np.save('preprocessed_data.npy', data)


