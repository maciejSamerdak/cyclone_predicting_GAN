from networks.meteo_gan import MeteoGAN
from data_processing.meteo_collector import load_datasets

import tensorflow as tf

EPOCHS = 1000

# rang = 4
# offst = 3
# arg = "refinement_deepest"
#
# for j in range(rang - offst):
#     (train_ds, val_ds) = load_datasets(batch_size=4)

# rang = 4
# offst = 0
# arg = "batch"
#
# for i in [1, 2, 4, 8, 16]:
#     for j in range(rang - offst):
#         (train_ds, val_ds) = load_datasets(batch_size=i)
#
#         MeteoGAN().train(train_ds, val_ds, epochs=EPOCHS, filedesc=f"{arg}_{i}_{j + 1 + offst}_of_{rang}")

(train_ds, val_ds) = load_datasets(batch_size=1, augmentation=1)
MeteoGAN().train(train_ds, val_ds, epochs=EPOCHS, augmentation=1, acc_treshold=0.65)
# gan = MeteoGAN()
# gan.discrimination_network.load_weights("trained_weights/meteo/disc_0.5722819593787336_epoch_3")
# gan.refinement_network.load_weights("trained_weights/meteo/ref_0.5722819593787336_epoch_3")
# val_ds = val_ds.unbatch().batch(32)
# output = 0
# size = 0
# for batch, labels in val_ds:
#     output += tf.reduce_sum(gan.accuracy(labels, gan.predict(batch))).numpy()
#     size += len(labels)
#
# print(output / size)
