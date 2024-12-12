from networks.gan import GAN
from data_processing.preprocessor import get_datasets

import tensorflow as tf

EPOCHS = 1000

# for x in [32, 64, 128, 256]:
#     (train_ds, val_ds) = get_datasets(width=x, height=x, batch_size=1)
#
#         # total_sum = 0
#         # for batch in val_ds:
#         #     total_sum += batch[1].numpy().sum()
#
#     gan = GAN()
#     gan.IMG_HEIGHT = x
#     gan.IMG_WIDTH = x
#     gan.train(train_ds, val_ds, epochs=EPOCHS)

# rang = 4
# offst = 3
# arg = "refinement_deepest"
#
# (train_ds, val_ds) = get_datasets(batch_size=4)
#
# for j in range(rang - offst):
#     GAN().train(train_ds, val_ds, epochs=EPOCHS, filedesc=f"{arg}_{j + 1 + offst}_of_{rang}")

(train_ds, val_ds) = get_datasets(batch_size=1, augmentation=2)
GAN().train(train_ds, val_ds, epochs=EPOCHS, augmentation=2, acc_treshold=0.65)
# gan = GAN()
# gan.discrimination_network.load_weights("trained_weights/gan/disc_0.49760765550239233")
# gan.refinement_network.load_weights("trained_weights/gan/ref_0.49760765550239233")
# val_ds = val_ds.unbatch().batch(32)
# output = 0
# size = 0
# for batch, labels in val_ds:
#     output += tf.reduce_sum(gan.accuracy(labels, gan.predict(batch))).numpy()
#     size += len(labels)
#
# print(output / size)
