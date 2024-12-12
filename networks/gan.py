import numpy as np
import json
import statistics

import tensorflow as tf
from tensorflow.keras.models import Sequential
# import imageio
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time

from IPython import display

from data_processing.preprocessor import img_width
from data_processing.preprocessor import img_height

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class GAN:
    BUFFER_SIZE = 60000

    IMG_WIDTH = img_width
    IMG_HEIGHT = img_height

    train_set_size = 183
    val_set_size = 836

    threshold = 0.5
    lbd = 0.4
    target_generator_loss = 5

    epsilon = 10 ** -6
    decay = 0.9

    noise_dim = [64, 64, 3]
    num_examples_to_generate = 16

    def __init__(self):
        self.refinement_network = self.create_refinement_network()
        self.discrimination_network = self.create_discrimination_network()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        checkpoint_dir = './gan_training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.refinement_network,
                                              discriminator=self.discrimination_network)

    def create_refinement_network(self):
        return Sequential([
            layers.InputLayer(input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, 3)),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
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
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2DTranspose(3, (3, 3), padding='same', activation='relu'),
          ])

    def create_discrimination_network(self):
        return Sequential([
            layers.InputLayer(input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, 3)),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            # layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(epsilon=self.epsilon, momentum=self.decay),
            # layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(1),  # layers.Dense(1, activation='softmax') <- non-logits
          ])

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def accuracy(labels, output):
        target_probability = tf.math.abs(labels - 1)
        return tf.ones_like(labels) - tf.math.abs(target_probability - tf.reshape(output, [-1]))

    def generator_loss_1(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def generator_loss_2(real_image, fake_image):
        return tf.math.reduce_max(
            tf.math.reduce_mean(tf.math.square(real_image - fake_image), axis=[-1, 1])
        )

    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.refinement_network(images, training=True)

            real_output = self.discrimination_network(images, training=True)
            fake_output = self.discrimination_network(generated_images, training=True)

            gen_loss_1 = self.generator_loss_1(fake_output)
            gen_loss_2 = self.generator_loss_2(images, generated_images)
            gen_loss = gen_loss_1 + self.lbd * gen_loss_2
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.refinement_network.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discrimination_network.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.refinement_network.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discrimination_network.trainable_variables))

        train_accuracy = self.accuracy(np.zeros(len(fake_output), dtype="int32"), tf.cast(tf.greater(tf.nn.sigmoid(fake_output), self.threshold), dtype=tf.int32))

        return [gen_loss, disc_loss, gen_loss_2, gen_loss_1, train_accuracy]

    @tf.function
    def val_step(self, images, labels):
        output = self.predict(images)

        return self.accuracy(labels, output)

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
            base_gen_loss = []
            local_train_set_size = 0
            train_acc_score = 0
            for image_batch in train_dataset.shuffle(buffer_size=train_set_size, seed=123):
                loss = self.train_step(image_batch[0])

                train_acc_score += tf.reduce_sum(loss.pop()).numpy()
                base_gen_loss.append(tf.reduce_mean(loss.pop()).numpy())
                local_train_set_size += image_batch[1].shape[0]

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
                self.refinement_network.save_weights(f"trained_weights/gan/ref_{accuracy}_epoch_{epoch}")
                self.discrimination_network.save_weights(f"trained_weights/gan/disc_{accuracy}_epoch_{epoch}")

            if val_dataset:
                val_acc.append(accuracy)
            train_loss.append([tf.reduce_mean(local_gen_losses).numpy(), tf.reduce_mean(local_disc_losses).numpy(), image_loss])

            # print(f'Image reconstruction loss: {image_loss}')
            # print(f'Generator base loss: {statistics.mean(base_gen_loss)}')
            # print(f'Generator combined loss: {tf.reduce_mean(local_gen_losses).numpy()}')
            # print(f"Train accuracy: {train_acc_score / local_train_set_size}")
            epoch += 1

        # Generate after the final epoch
        display.clear_output(wait=True)
        # testu_data = train_dataset.as_numpy_iterator().next()
        # generate_and_save_images(refinement_network, epochs, testu_data)
        self.checkpoint.save(file_prefix="gan-finish")
        with open(f'gan_train_loss_{filedesc if filedesc is not None else time.asctime().replace(" ", "_").replace(":", "_")}.json', "w") as file:
            json.dump([[str(x), str(y), str(z)] for [x, y, z] in train_loss], file)
        if val_dataset:
            with open(f'gan_val_acc_{filedesc if filedesc is not None else time.asctime().replace(" ", "_").replace(":", "_")}.json', "w") as file:
                json.dump([str(a) for a in val_acc], file)

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def predict(self, data):
        output = self.discrimination_network(self.refinement_network(data, training=False), training=False)

        return tf.cast(tf.greater(tf.nn.sigmoid(output), self.threshold), dtype=tf.int32)

