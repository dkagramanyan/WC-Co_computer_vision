from src.nn import U2NET
import numpy as np
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split

import csv
import os

from src.nn_utils import SaveImageCallback

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

model = tf.keras.Model(inputs=inputs, outputs=out[0], name='u2netmodel')
model.built = True
model.load_weights('data/logs/u2net_2021-11-19_checkpoint/checkpoints/')

x_train, x_test, y_train, y_test = train_test_split(all_images_rgb, all_images_rgb, test_size=0.2)

name = f'u2net_{datetime.datetime.now().date()}'
log_dir = f'data/logs/{name}_tensorboard/'
checkpoint_filepath = f'data/logs/{name}_checkpoint/checkpoints/'
csv_log_path = f'data/logs/train_csv/'
csv_log_filepath = csv_log_path + f'{name}.csv'
images_save_dir = f'data/logs/{name}_val_images/'

if not os.path.exists(csv_log_path):
    os.makedirs(csv_log_path)

if not os.path.exists(csv_log_filepath):
    with open(csv_log_filepath, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

# optim=tf.keras.optimizers.RMSprop(learning_rate=0.000015, rho=0.9, momentum=0.1, epsilon=1e-07, centered=True)
optim = tf.keras.optimizers.Adam(learning_rate=0.000015, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

csv_logger = tf.keras.callbacks.CSVLogger(csv_log_filepath)

n_samples = 3

save_callback = SaveImageCallback(x_test, images_save_dir, n_samples, net)

model.compile(optimizer=optim, loss='mse', metrics=['MAE'])
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=2,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[tensorboard_callback,
                               model_checkpoint_callback,
                               early_stop_callback,
                               csv_logger,
                               save_callback])
