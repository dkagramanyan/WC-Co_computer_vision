import tensorflow as tf
import os
import numpy as np


class SaveImageCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_images, save_dir, n_samples, net):
        super().__init__()
        self.images = test_images
        self.safe_dir = save_dir
        self.n_samples = n_samples
        self.net = net

        if not os.path.exists(save_dir):
            os.makedirs(self.safe_dir)

    def on_epoch_end(self, epoch, logs=None):
        indeces = np.random.randint(0, len(self.images), size=self.n_samples)
        for i in indeces:
            predict = self.net(tf.expand_dims(self.images[i], axis=0))[0][0]
            original = self.images[i]
            concat = np.concatenate([predict.numpy(), original], axis=1)
            tf.keras.preprocessing.image.save_img(f'{self.safe_dir}epoch={epoch}_index={i}.png', concat * 255)
