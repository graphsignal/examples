import logging
import time
import random
import tensorflow as tf
import tensorflow_datasets as tfds
# Graphsignal: import module
import graphsignal
from graphsignal.profilers.keras import GraphsignalCallback

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: configure module
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(workload_name='MNIST Keras Training')

tfds.disable_progress_bar()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(128)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'])

# Graphsignal: add profiler callback
model.fit(ds_train,
        epochs=10,
        validation_data=ds_test,
        callbacks=[GraphsignalCallback()])
