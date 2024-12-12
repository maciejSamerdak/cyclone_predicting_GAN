from networks.gan import *
import tensorflow as tf
import os
import time
import json
from IPython import display
# from data_processing.meteo_collector import load_datasets


class MeteoGAN(GAN):
    BUFFER_SIZE = 60000

    train_set_size = 186
    val_set_size = 928

    threshold = 0.5
    lbd = 0.4
    target_generator_loss = 5

    epsilon = 10 ** -6
    decay = 0.9

    checkpoint_dir = './meteo_gan_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    def create_refinement_network(self):
        return Sequential([
            layers.InputLayer(input_shape=(img_width, img_height, 4)),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            # layers.Conv2D(1024, 3, padding='same', activation='relu'),
            # layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            # layers.Conv2D(2048, 3, padding='same', activation='relu'),
            # layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            # layers.Conv2DTranspose(1024, 3, padding='same', activation='relu'),
            # layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            # layers.Conv2DTranspose(512, 3, padding='same', activation='relu'),
            # layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # # layers.LeakyReLU(),
            layers.Conv2DTranspose(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(4, 3, padding='same', activation='relu'),
          ])

    def create_discrimination_network(self):
        return Sequential([
            layers.InputLayer(input_shape=(img_width, img_height, 4)),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            # layers.LeakyReLU(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(1),  # layers.Dense(1, activation='softmax') <- non-logits
          ])

    def train(self, train_dataset, val_dataset=None, epochs=None, augmentation=0, filedesc=None, acc_treshold=1.0):
        train_set_size = self.train_set_size * (augmentation + 1)
        self.refinement_network = self.create_refinement_network()
        self.discrimination_network = self.create_discrimination_network()
        train_loss = []
        val_acc = []
        epoch = 0
        image_loss = 10.0 ** 6
        while image_loss > self.target_generator_loss if not epochs else epoch < epochs:
            start = time.time()

            # print('Starting epoch {}'.format(epoch + 1))

            local_gen_losses = []
            local_disc_losses = []
            val_acc_score = 0
            local_val_set_size = 0
            image_losses = []
            for image_batch in train_dataset.shuffle(buffer_size=train_set_size, seed=123):
                loss = self.train_step(image_batch[0])
                local_gen_losses.append(loss[0].numpy())
                local_disc_losses.append(loss[1].numpy())
                image_losses.append(loss[2].numpy())

            if val_dataset:
                for image_batch in val_dataset.shuffle(buffer_size=self.val_set_size, seed=123):
                    acc_score = self.val_step(image_batch[0], image_batch[1])
                    val_acc_score += tf.reduce_sum(acc_score).numpy()
                    local_val_set_size += image_batch[1].shape[0]

            # Produce images for the GIF as you go
            # display.clear_output(wait=True)
            # generate_and_save_images(refinement_network,
            #                          epoch + 1,
            #                          seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

            image_loss = max(image_losses)

            accuracy = val_acc_score / local_val_set_size
            print(f"Val accuracy: {accuracy}")
            if accuracy >= acc_treshold:
                self.refinement_network.save_weights(f"trained_weights/meteo/ref_{accuracy}_epoch_{epoch}")
                self.discrimination_network.save_weights(f"trained_weights/meteo/disc_{accuracy}_epoch_{epoch}")

            if val_dataset:
                val_acc.append(accuracy)
            train_loss.append([tf.reduce_mean(local_gen_losses).numpy(), tf.reduce_mean(local_disc_losses).numpy(), image_loss])

            print(f'Reconstruction loss: {image_loss}')
            epoch += 1

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.checkpoint.save(file_prefix="meteo_gan-finish")
        with open(f'meteo_gan_train_loss_{filedesc if filedesc is not None else time.asctime().replace(" ", "_").replace(":", "_")}.json', "w") as file:
            json.dump([[str(x), str(y), str(z)] for [x, y, z] in train_loss], file)
        if val_dataset:
            with open(f'meteo_gan_val_acc_{filedesc if filedesc is not None else time.asctime().replace(" ", "_").replace(":", "_")}.json', "w") as file:
                json.dump([str(a) for a in val_acc], file)
