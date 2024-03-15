import tensorflow as tf
import numpy as np

def evaluate_model(x_test, y_test):
    model = tf.keras.models.load_model('trained_model.h5')
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    
    # Prepare the metrics string
    metrics_str = f'Test Accuracy: {test_accuracy * 100:.3f}%\nTest Loss: {test_loss}'
    
    # Write the metrics to a file
    with open('metrics.txt', 'w') as file:
        file.write(metrics_str)

if __name__ == "__main__":
    # Load preprocessed data
    data = np.load('preprocessed_data.npy', allow_pickle=True).item()
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    evaluate_model(x_test, y_test)

