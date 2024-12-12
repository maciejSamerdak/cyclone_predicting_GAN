# import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


SOURCES_DIR = "G:\\MGSTR\\source_data\\"

IMAGES_DIR = "G:\\MGSTR\\source_data\\exported_images"


img_height = 64
img_width = 64
num_classes = 2


def get_datasets(batch_size, height=img_height, width=img_width, augmentation=0):
    _ds = tf.keras.preprocessing.image_dataset_from_directory(
      directory="H:/MGSTR/source_data/exported_images/train",
      seed=123,
      image_size=(height, width),
    ).unbatch()

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      directory="H:/MGSTR/source_data/exported_images/val",
      seed=123,
      image_size=(height, width),
    ).unbatch()

    train_ds = _ds.shard(4, 1).concatenate(_ds.shard(4, 2)).concatenate(_ds.shard(4, 3)).batch(batch_size)
    val_ds = val_ds.concatenate(_ds.shard(4, 0)).batch(batch_size)

    autotune = tf.data.experimental.AUTOTUNE

    train_set_size = 183

    train_ds = train_ds.cache().shuffle(train_set_size * (augmentation + 1), seed=123).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    data_augmentation = tf.keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical",
            input_shape=(height, width, 3),
            seed=123
        ),
        layers.experimental.preprocessing.RandomRotation((0.5, 0.5), interpolation="nearest", seed=123),
      ]
    )

    # normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # # image_batch, labels_batch = next(iter(normalized_train_ds))
    # # first_image = image_batch[0]
    # # # Notice the pixels values are now in `[0,1]`.
    # # print(np.min(first_image), np.max(first_image))
    # normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    # normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # train_ds = normalized_train_ds
    # val_ds = normalized_val_ds
    # test_ds = normalized_test_ds
    if augmentation != 0:
        aug_train_ds = train_ds

        for i in range(augmentation):
            aug_train_ds = aug_train_ds.concatenate(train_ds.map(lambda x, y: (data_augmentation(x), y)))

        train_ds = aug_train_ds

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds
# dp = 0
# for batch in train_ds:
#     dp += len(batch[0].numpy())
# print(dp)
# dp = 0
# for batch in val_ds:
#     dp += len(batch[0].numpy())
# print(dp)
