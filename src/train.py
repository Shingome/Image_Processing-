import click
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Rescaling, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import os

@click.command()
@click.option('--epochs', default=1)
@click.option('--optimizer', default="adam")
@click.option('--batch_size', default=32)
def train(epochs, optimizer, batch_size):
    data_dir = "./../dataset/"

    img_size = (8, 8)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode='grayscale',
        image_size=img_size,
        batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode='grayscale',
        image_size=img_size,
        batch_size=batch_size)

    classes = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    model = keras.Sequential()
    model.add(InputLayer((8, 8, 1)))
    model.add(Flatten())
    model.add(Rescaling(1. / 255))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, shuffle=True)

    models_path = '../models/'
    count = f"{len(os.listdir(models_path))}".zfill(6)

    model_dir = os.path.join(models_path, f"{optimizer}_{epochs}_{batch_size}_{count}")
    os.mkdir(model_dir)

    filename_weights = os.path.join(model_dir, "weights.keras")
    filename_plot_model = os.path.join(model_dir, "model_plot.jpg")
    filename_plot_train = os.path.join(model_dir, "train_plot.jpg")

    model.save(filename_weights)

    plot_model(model,
               to_file=filename_plot_model,
               show_shapes=True,
               show_layer_names=True)

    fig, ax1 = plt.subplots()

    ax1.plot(history.history['accuracy'], color='b', label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], color='g', linestyle='dashed', label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(history.history['loss'], color='r', label='Train Loss')
    ax2.plot(history.history['val_loss'], color='c', linestyle='dashed', label='Test Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='c')
    ax2.legend(loc='upper right')

    plt.title('Model Accuracy and Loss')

    plt.savefig(filename_plot_train)

    plt.show()


if __name__ == "__main__":
    train()