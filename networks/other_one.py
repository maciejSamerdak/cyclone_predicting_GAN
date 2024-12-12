import json
import time

import tensorflow as tf
from sklearn.svm import OneClassSVM


class Model:
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()
    lbd = 0.1
    threshold = 0.5
    reference_optimizer = tf.keras.optimizers.Adam(1e-4)
    secondary_optimizer = tf.keras.optimizers.Adam(1e-4)
    reference_data_size = 1000
    target_data_size = 183
    classifier = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale", cache_size=2000)
    target_reference = None
    flattener = tf.keras.layers.Flatten()

    def __init__(self):
        base_model = tf.keras.applications.vgg16.VGG16()
        self.reference_network = tf.keras.models.Sequential(base_model.layers[:-1])
        self.secondary_network = tf.keras.models.Sequential(base_model.layers[:-1])
        self.reference_classifier = tf.keras.models.Sequential(base_model.layers[-1:])
        self.secondary_classifier = tf.keras.models.Sequential(base_model.layers[-1:])

    def descriptiveness_loss(self, reference_output, original_labels):
        return self.crossentropy(original_labels, reference_output)

    @staticmethod
    def compactness_loss(secondary_output):
        m = (tf.stack([tf.reduce_sum(secondary_output, axis=0) for _ in range(secondary_output.shape[0])], axis=0) - secondary_output) / (secondary_output.shape[0] - 1)
        z = secondary_output - m
        return tf.norm(z, ord="euclidean")

    def setup_classifier(self, reference_data):
        reference_features = self.reference_network(reference_data)
        self.target_reference = self.flattener(reference_features)
        self.classifier.fit(self.target_reference)

    def classify(self, input_features):
        return self.classifier.predict(self.flattener(input_features))

    def predict(self, input_data, reference_data):
        print("Training classifier...")
        self.setup_classifier(reference_data)
        print("Classifier trained!")
        return self.classify(self.secondary_network(input_data))

    def train(self, target_data, reference_data, epochs=500):
        epoch = 0
        train_loss = []
        while epoch < epochs:
            start = time.time()
            local_ref_losses = []
            local_sec_losses = []
            local_tot_losses = []
            reference_iterator = reference_data.shuffle(buffer_size=self.reference_data_size).as_numpy_iterator()
            for batch, labels in target_data.shuffle(buffer_size=self.target_data_size):
                reference_batch, reference_labels = reference_iterator.next()
                losses = self.train_step(batch, reference_batch, reference_labels)
                local_ref_losses.append(losses[0].numpy())
                local_sec_losses.append(losses[1].numpy())
                local_tot_losses.append(losses[2].numpy())
            train_loss.append([
                tf.reduce_mean(local_ref_losses).numpy(),
                tf.reduce_mean(local_sec_losses).numpy(),
                tf.reduce_mean(local_tot_losses).numpy(),
            ])
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            epoch += 1
        with open(f'other_one_train_loss_{time.asctime().replace(" ", "_").replace(":", "_")}.json', "w") as file:
            json.dump([[str(x), str(y), str(z)] for [x, y, z] in train_loss], file)

    @tf.function
    def train_step(self, target_images, reference_images, reference_labels):
        with tf.GradientTape() as reference_tape, tf.GradientTape() as other_tape:
            reference_output = self.reference_classifier(self.reference_network(reference_images, training=True))
            secondary_output = self.secondary_classifier(self.secondary_network(target_images, training=True))
            descriptiveness_loss = self.descriptiveness_loss(reference_output, reference_labels)
            compactness_loss = self.compactness_loss(secondary_output)
            total_loss = descriptiveness_loss + self.lbd * compactness_loss
        gradients_of_reference = reference_tape.gradient(total_loss, self.reference_network.trainable_variables)
        gradients_of_secondary = other_tape.gradient(total_loss, self.secondary_network.trainable_variables)

        self.reference_optimizer.apply_gradients(zip(gradients_of_reference, self.reference_network.trainable_variables))
        self.secondary_optimizer.apply_gradients(zip(gradients_of_secondary, self.secondary_network.trainable_variables))

        return [descriptiveness_loss, compactness_loss, total_loss]
