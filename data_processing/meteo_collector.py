import os

import cv2
import xarray
import numpy as np
import tensorflow as tf

source_dir = "H:/MGSTR/source_data"
stages = [
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "other",
]
layers = [
    "horizontal_wind",
    "humidity",
    "sea_level_pressure",
]

IMG_WIDTH = 64
IMG_HEIGHT = IMG_WIDTH
# BATCH_SIZE = 16

target_layers_count = 4
target_spec = (
        tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, target_layers_count), dtype=tf.float32, name=None),
        tf.TensorSpec(shape=(), dtype=tf.int32, name=None)
    )

resize_layer = tf.keras.layers.experimental.preprocessing.Resizing(IMG_HEIGHT, IMG_WIDTH, interpolation='nearest')


def get_files_dir(stage):
    return os.path.join(source_dir, stage, "EFR")


def get_dataset(stage, filename):
    return xarray.open_dataset(
        f"{get_files_dir(stage)}/{filename}/tie_meteo.nc",
        engine='netcdf4'
    )


def get_image_tensor(ds, layer):
    subset = ds[layer]
    reps = np.ones(subset.shape[1], dtype=int) * 64
    axis = len(subset.shape) - 2
    full_image = np.repeat(subset.data.transpose(), reps, axis=axis).transpose()
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / subset.valid_max)
    if layer == layers[0]:
        subset_0 = np.expand_dims(scale_down_array(full_image[:, :, 0]), axis=-1)
        subset_1 = np.expand_dims(scale_down_array(full_image[:, :, 1]), axis=-1)
        image_tensor = tf.constant(np.concatenate([subset_0, subset_1], axis=-1))
    else:
        image_tensor = tf.constant(scale_down_array(full_image))
    return normalization_layer(image_tensor)


def scale_down_array(array):
    scaled_array = cv2.resize(
        array.repeat(3).reshape([array.shape[0], array.shape[1], 3]).astype("uint8"),
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )
    return scaled_array[:, :, 1].reshape([IMG_WIDTH, IMG_HEIGHT]).astype('float32')


def generate_data(stage):
    data = []
    files = list(filter(lambda filename: ".zip" not in filename, os.listdir(get_files_dir(stage))))
    for file in files:
        print(f"Generating tensor {files.index(file) + 1}/{len(files)} for {stage}")
        object_tensor = None
        for layer in layers:
            image_tensor = get_image_tensor(get_dataset(stage, file), layer)
            if object_tensor is None:
                object_tensor = image_tensor
            else:
                object_tensor = tf.concat([object_tensor, tf.expand_dims(image_tensor, axis=-1)], axis=-1)
        data.append(object_tensor)
    return data


def generate_labels(data, stage):
    return np.zeros(len(data), dtype="int32") if stage != "other" else tf.ones(len(data), dtype="int32")


def generate_dataset(_stages):
    print(f"Generating dataset for {_stages}")
    data = []
    labels = np.array([], dtype="int32")
    for stage in _stages:
        print(f"Collecting data from {stage}")
        local_data = generate_data(stage)
        labels = np.concatenate([labels, generate_labels(local_data, stage)])
        data += local_data
    print("Dataset completed!")
    return tf.data.Dataset.from_tensors((data, labels)).unbatch()


def generate_datasets():
    _ds = generate_dataset(["stage_4"])
    train_ds = _ds.shard(4, 1).concatenate(_ds.shard(4, 2)).concatenate(_ds.shard(4, 3))
    val_ds = generate_dataset(["stage_3", "stage_2", "stage_1", "other"]).concatenate(_ds.shard(4, 0))

    tf.data.experimental.save(
        train_ds,
        path=os.path.join(source_dir, "exported_meteo", "train")
    )
    tf.data.experimental.save(
        val_ds,
        path=os.path.join(source_dir, "exported_meteo", "val")
    )
    return train_ds, val_ds


def load_datasets(batch_size, augmentation=0):
    train_ds = tf.data.experimental.load(
        path=os.path.join(source_dir, "exported_meteo", "train"),
        element_spec=target_spec
    ).batch(batch_size)

    val_ds = tf.data.experimental.load(
        path=os.path.join(source_dir, "exported_meteo", "val"),
        element_spec=target_spec
    ).batch(batch_size)

    autotune = tf.data.experimental.AUTOTUNE

    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical",
            input_shape=(IMG_WIDTH, IMG_WIDTH, 4),
            seed=123
        ),
        tf.keras.layers.experimental.preprocessing.RandomRotation((0.5, 0.5), interpolation="nearest", seed=123),
      ]
    )

    if augmentation != 0:
        aug_train_ds = train_ds

        for i in range(augmentation):
            aug_train_ds = aug_train_ds.concatenate(train_ds.map(lambda x, y: (data_augmentation(x), y)))

        train_ds = aug_train_ds

    train_ds = train_ds.cache().shuffle(1000, seed=123).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds



# ds = generate_dataset(["demo"]).unbatch().batch(BATCH_SIZE)
#
# tf.data.experimental.save(
#     ds,
#     path=os.path.join(source_dir, "exported_meteo", "test")
# )
#
# ds = tf.data.experimental.load(
#     path=os.path.join(source_dir, "exported_meteo", "test"),
#     element_spec=target_spec
# )
