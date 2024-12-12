import os

from networks.other_one import Model
from data_processing.preprocessor import get_datasets
import tensorflow as tf

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

EPOCHS = 50

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

# for i in [1, 2, 4, 8, 16]:
#     for _ in range(4):
(train_ds, val_ds) = get_datasets(batch_size=16)
ref_dat =

    # total_sum = 0
    # for batch in val_ds:
    #     total_sum += batch[1].numpy().sum()

model = Model()
model.train(train_ds, ref_dat, epochs=EPOCHS)
model.predict(val_ds, train_ds)
