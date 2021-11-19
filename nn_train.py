from u2net import U2NET
import numpy as np
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split

import csv
import os


class SaveCallback(tf.keras.callbacks.Callback):

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


all_images = np.load('data/saved np/all_images_no_preprocess.npy', allow_pickle=True)
all_images_rgb = []
for i, images_list in enumerate(all_images):
    for image_gray in images_list:
        tf_image = tf.expand_dims(image_gray / 255, 2)
        #    tf_rgb = tf.image.grayscale_to_rgb(tf_image)
        tf_preproc = tf.image.resize(tf_image, (1024, 1024))
        all_images_rgb.append(tf_preproc)

all_images_rgb = np.array(all_images_rgb)

image_shape = (1024, 1024, 1)
inputs = tf.keras.Input(shape=image_shape)
net = U2NET(1)
out = net(inputs)
# net.built = True
# net.load_weights('data/saved_models/u2net_loss=0.0089.h5')

model = tf.keras.Model(inputs=inputs, outputs=out[0], name='u2netmodel')

x_train, x_test, y_train, y_test = train_test_split(all_images_rgb[:10], all_images_rgb[:10], test_size=0.2)

name=f'u2net_{datetime.datetime.now().date()}'
log_dir=f'data/logs/{name}_tensorboard/'
checkpoint_filepath = f'data/logs/{name}_checkpoint/checkpoints/'
csv_log_path= f'data/logs/train_csv/'
csv_log_filepath= csv_log_path+f'{name}.csv'
images_save_dir=f'data/logs/{name}_val_images/'

if not os.path.exists(csv_log_filepath):
    os.makedirs(csv_log_path)

if not os.path.exists(csv_log_filepath):
    with open(csv_log_filepath) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)



optim=tf.keras.optimizers.RMSprop(learning_rate=0.00015, rho=0.9, momentum=0.1, epsilon=1e-07, centered=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
early_stop_callback=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

csv_logger = tf.keras.callbacks.CSVLogger(csv_log_filepath)

n_samples=3

save_callback=SaveCallback(x_test,images_save_dir,n_samples,net)


model.compile(optimizer=optim, loss='mse', metrics=['MAE'])
history=model.fit(x_train, y_train,
                  epochs=50,
                  batch_size=2,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  callbacks=[tensorboard_callback,
                             model_checkpoint_callback,
                             early_stop_callback,
                             csv_logger,
                             save_callback])