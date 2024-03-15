# train.py
import tensorflow as tf
import numpy as np

def train_model(x_train, y_train):
    # Check if a GPU is available
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # If you have a GPU, set the device to GPU
    if tf.config.experimental.list_physical_devices('GPU'):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

    model = tf.keras.models.load_model('modified_model.h5')
    new_model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    new_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    new_model.fit(x_train, y_train, epochs=5)
    new_model.save('trained_model.h5')

if __name__ == "__main__":
    # Load preprocessed data
    data = np.load('preprocessed_data.npy', allow_pickle=True).item()
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    train_model(x_train, y_train)
