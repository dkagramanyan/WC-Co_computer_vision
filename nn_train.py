from u2net import U2NET
import numpy as np
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split

import csv
import os


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

x_train, x_test, y_train, y_test = train_test_split(all_images_rgb, all_images_rgb, test_size=0.2)

optim = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9, momentum=0.1, epsilon=1e-07, centered=True)

# train

name = f'u2net_{datetime.datetime.now().date()}'
log_dir = f'data/logs/{name}_tensorboard/'
checkpoint_filepath = f'data/logs/{name}_checkpoint/checkpoints/'
csv_log_filepath = f'data/logs/train_csv/{name}.csv'

if not os.path.exists(csv_log_filepath):
    with open(csv_log_filepath, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_MAE',
    mode='max',
    save_best_only=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
csv_logger = tf.keras.callbacks.CSVLogger(csv_log_filepath)

model.compile(optimizer=optim, loss='mse', metrics=['MAE'])
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=3,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[tensorboard_callback,
                               model_checkpoint_callback,
                               early_stop_callback,
                               csv_logger])
