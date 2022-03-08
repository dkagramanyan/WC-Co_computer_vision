from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
# import tensorflow_probability as tfp
from src.nn import RSU7, RSU6, RSU5, RSU4, RSU4F, ConvBlock
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Reshape, UpSampling2D
import time

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.run_functions_eagerly(True)

# уже отнормированы
images = np.load('data/saved np/images_no_filters.npy')
images = images.reshape((-1, 1024, 1024, 1)).astype(np.float32)

train_images, test_images = train_test_split(images, test_size=0.2, random_state=421)

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, input_shape):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            RSU7(16, 32),
            tf.keras.layers.MaxPool2D((2, 2), 2),
            RSU6(32, 64),
            tf.keras.layers.MaxPool2D((2, 2), 2),
            RSU5(64, 128),
            tf.keras.layers.MaxPool2D((2, 2), 2),
            RSU4(128, 256),
            tf.keras.layers.MaxPool2D((2, 2), 2),
            RSU4F(256, 256),
            tf.keras.layers.MaxPool2D((2, 2), 2),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dense(self.latent_dim + self.latent_dim)

        ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),

                Dense(512),
                tf.keras.layers.LeakyReLU(),

                Dense(256),
                tf.keras.layers.LeakyReLU(),

                Reshape(target_shape=(16, 16, 1)),

                UpSampling2D(size=(2, 2), interpolation='bilinear'),
                RSU4F(256, 256),

                UpSampling2D(size=(2, 2), interpolation='bilinear'),
                RSU4F(128, 128),

                UpSampling2D(size=(2, 2), interpolation='bilinear'),
                RSU4(64, 64),

                UpSampling2D(size=(2, 2), interpolation='bilinear'),
                RSU5(32, 32),

                UpSampling2D(size=(2, 2), interpolation='bilinear'),
                RSU6(16, 16),

                UpSampling2D(size=(2, 2), interpolation='bilinear'),
                RSU6(8, 1)

            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x), logpx_z, logpz


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss, rec_loss, lat_loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, rec_loss, lat_loss

epochs = 10

latent_dim = 256
image_shape = (1024, 1024, 1)
num_examples_to_generate = 4
batch_size = 5

model = CVAE(latent_dim, image_shape)

train_size = train_images.shape[0]
test_size = test_images.shape[0]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

# Pick a sample of the test set for generating output images

assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(20, 20))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


generate_and_save_images(model, 0, test_sample)

optimizer = tf.keras.optimizers.Adam(1e-4,amsgrad=True)

n_batches = len(train_dataset)
total_loss=[]
total_rec_loss=[]
total_lat_loss=[]

model_save_path='data/logs'

for epoch in range(1, 10000 + 1):
    start_time = time.time()

    for i, train_x in enumerate(train_dataset):
        loss, rec_loss, lat_loss = train_step(model, train_x, optimizer)
        total_loss.append(loss)
        total_rec_loss.append(rec_loss)
        total_lat_loss.append(lat_loss)
        print(f'Epoch: {epoch}, batch {i}/{n_batches}, elbo: {np.mean(total_loss)}, rec_loss: {np.mean(total_rec_loss)}, lat_loss: {np.mean(total_lat_loss)}')
    end_time = time.time()

    test_loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss, rec_loss, lat_loss=compute_loss(model, test_x)
        test_loss(loss)
    elbo = test_loss.result()

    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, round(end_time - start_time,2)))
    generate_and_save_images(model, epoch, test_sample)
    model.save_weights(f'{model_save_path}/cvae_elbo_{np.mean(total_loss)}_test_elbo_{elbo}.h5')

#%%
