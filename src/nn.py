import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D, Conv2DTranspose, \
    LeakyReLU
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.activations import sigmoid

import numpy as np

# Optimizer / Loss
adam = keras.optimizers.Adam(learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-08)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model.ckpt', save_weights_only=True, verbose=1)
bce = keras.losses.BinaryCrossentropy()


def bce_loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = bce(y_true, y_pred[0])
    loss1 = bce(y_true, y_pred[1])
    loss2 = bce(y_true, y_pred[2])
    loss3 = bce(y_true, y_pred[3])
    loss4 = bce(y_true, y_pred[4])
    loss5 = bce(y_true, y_pred[5])
    loss6 = bce(y_true, y_pred[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6


class ConvBlock(keras.layers.Layer):
    def __init__(self, out_ch=3, dirate=1):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(out_ch, (3, 3), strides=1, padding='same', dilation_rate=dirate,
                           kernel_initializer=he_normal())
        self.bn = BatchNormalization()
        self.relu = LeakyReLU()

    def __call__(self, inputs):
        hx = inputs

        x = self.conv(hx)
        x = self.bn(x)
        x = self.relu(x)

        return x


class RSU7(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=(2, 2))

        self.conv_b3 = ConvBlock(mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=(2, 2))

        self.conv_b4 = ConvBlock(mid_ch, dirate=1)
        self.pool4 = MaxPool2D(2, strides=(2, 2))

        self.conv_b5 = ConvBlock(mid_ch, dirate=1)
        self.pool5 = MaxPool2D(2, strides=(2, 2))

        self.conv_b6 = ConvBlock(mid_ch, dirate=1)
        self.conv_b7 = ConvBlock(mid_ch, dirate=2)

        self.conv_b6_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b5_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b4_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b3_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
        self.upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def __call__(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv_b3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv_b4(hx)
        hx = self.pool4(hx4)

        hx5 = self.conv_b5(hx)
        hx = self.pool5(hx5)

        hx6 = self.conv_b6(hx)

        hx7 = self.conv_b7(hx6)

        hx6d = self.conv_b6_d(tf.concat([hx7, hx6], axis=3))
        hx6dup = self.upsample_5(hx6d)

        hx5d = self.conv_b5_d(tf.concat([hx6dup, hx5], axis=3))
        hx5dup = self.upsample_4(hx5d)

        hx4d = self.conv_b4_d(tf.concat([hx5dup, hx4], axis=3))
        hx4dup = self.upsample_3(hx4d)

        hx3d = self.conv_b3_d(tf.concat([hx4dup, hx3], axis=3))
        hx3dup = self.upsample_2(hx3d)

        hx2d = self.conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
        hx2dup = self.upsample_1(hx2d)

        hx1d = self.conv_b1_d(tf.concat([hx2dup, hx1], axis=3))

        return hx1d + hxin


class RSU6(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=(2, 2))

        self.conv_b3 = ConvBlock(mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=(2, 2))

        self.conv_b4 = ConvBlock(mid_ch, dirate=1)
        self.pool4 = MaxPool2D(2, strides=(2, 2))

        self.conv_b5 = ConvBlock(mid_ch, dirate=1)
        self.pool5 = MaxPool2D(2, strides=(2, 2))

        self.conv_b6 = ConvBlock(mid_ch, dirate=2)

        self.conv_b5_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b4_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b3_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
        self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def __call__(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv_b3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv_b4(hx)
        hx = self.pool4(hx4)

        hx5 = self.conv_b5(hx)

        hx6 = self.conv_b6(hx5)

        hx5d = self.conv_b5_d(tf.concat([hx6, hx5], axis=3))
        hx5dup = self.upsample_4(hx5d)

        hx4d = self.conv_b4_d(tf.concat([hx5dup, hx4], axis=3))
        hx4dup = self.upsample_3(hx4d)

        hx3d = self.conv_b3_d(tf.concat([hx4dup, hx3], axis=3))
        hx3dup = self.upsample_2(hx3d)

        hx2d = self.conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
        hx2dup = self.upsample_1(hx2d)

        hx1d = self.conv_b1_d(tf.concat([hx2dup, hx1], axis=3))

        return hx1d + hxin


class RSU5(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=(2, 2))

        self.conv_b3 = ConvBlock(mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=(2, 2))

        self.conv_b4 = ConvBlock(mid_ch, dirate=1)
        self.pool4 = MaxPool2D(2, strides=(2, 2))

        self.conv_b5 = ConvBlock(mid_ch, dirate=2)

        self.conv_b4_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b3_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
        self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def __call__(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv_b3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv_b4(hx)

        hx5 = self.conv_b5(hx4)

        hx4d = self.conv_b4_d(tf.concat([hx5, hx4], axis=3))
        hx4dup = self.upsample_3(hx4d)

        hx3d = self.conv_b3_d(tf.concat([hx4dup, hx3], axis=3))
        hx3dup = self.upsample_2(hx3d)

        hx2d = self.conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
        hx2dup = self.upsample_1(hx2d)

        hx1d = self.conv_b1_d(tf.concat([hx2dup, hx1], axis=3))

        return hx1d + hxin


class RSU4(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)

        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.pool1 = MaxPool2D(2, strides=(2, 2))

        self.conv_b2 = ConvBlock(mid_ch, dirate=1)
        self.pool2 = MaxPool2D(2, strides=(2, 2))

        self.conv_b3 = ConvBlock(mid_ch, dirate=1)
        self.pool3 = MaxPool2D(2, strides=(2, 2))

        self.conv_b4 = ConvBlock(mid_ch, dirate=2)

        self.conv_b3_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b2_d = ConvBlock(mid_ch, dirate=1)
        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def __call__(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv_b2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv_b3(hx)

        hx4 = self.conv_b4(hx3)

        hx3d = self.conv_b3_d(tf.concat([hx4, hx3], axis=3))
        hx3dup = self.upsample_2(hx3d)

        hx2d = self.conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
        hx2dup = self.upsample_1(hx2d)

        hx1d = self.conv_b1_d(tf.concat([hx2dup, hx1], axis=3))

        return hx1d + hxin


class RSU4F(keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.conv_b0 = ConvBlock(out_ch, dirate=1)
        self.conv_b1 = ConvBlock(mid_ch, dirate=1)
        self.conv_b2 = ConvBlock(mid_ch, dirate=2)
        self.conv_b3 = ConvBlock(mid_ch, dirate=4)
        self.conv_b4 = ConvBlock(mid_ch, dirate=8)
        self.conv_b3_d = ConvBlock(mid_ch, dirate=4)
        self.conv_b2_d = ConvBlock(mid_ch, dirate=2)
        self.conv_b1_d = ConvBlock(out_ch, dirate=1)

    def __call__(self, inputs):
        hx = inputs
        hxin = self.conv_b0(hx)

        hx1 = self.conv_b1(hxin)
        hx2 = self.conv_b2(hx1)
        hx3 = self.conv_b3(hx2)
        hx4 = self.conv_b4(hx3)
        hx3d = self.conv_b3_d(tf.concat([hx4, hx3], axis=3))
        hx2d = self.conv_b2_d(tf.concat([hx3d, hx2], axis=3))
        hx1d = self.conv_b1_d(tf.concat([hx2d, hx1], axis=3))
        return hx1d + hxin


class U2NET(keras.models.Model):
    def __init__(self, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(32, 64)
        self.pool12 = MaxPool2D((2, 2), 2)

        self.stage2 = RSU6(32, 128)
        self.pool23 = MaxPool2D((2, 2), 2)

        self.stage3 = RSU5(64, 256)
        self.pool34 = MaxPool2D((2, 2), 2)

        self.stage4 = RSU4(128, 512)
        self.pool45 = MaxPool2D((2, 2), 2)

        self.stage5 = RSU4F(256, 512)
        self.pool56 = MaxPool2D((2, 2), 2)

        self.stage6 = RSU4F(256, 512)

        self.stage5d = RSU4F(256, 512)
        self.stage4d = RSU4(128, 256)
        self.stage3d = RSU5(64, 128)
        self.stage2d = RSU6(32, 64)
        self.stage1d = RSU7(16, 64)

        self.side1 = Conv2D(out_ch, (3, 3), padding='same')
        self.side2 = Conv2D(out_ch, (3, 3), padding='same')
        self.side3 = Conv2D(out_ch, (3, 3), padding='same')
        self.side4 = Conv2D(out_ch, (3, 3), padding='same')
        self.side5 = Conv2D(out_ch, (3, 3), padding='same')
        self.side6 = Conv2D(out_ch, (3, 3), padding='same')

        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.upsample_out_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_out_3 = UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upsample_out_4 = UpSampling2D(size=(8, 8), interpolation='bilinear')
        self.upsample_out_5 = UpSampling2D(size=(16, 16), interpolation='bilinear')
        self.upsample_out_6 = UpSampling2D(size=(32, 32), interpolation='bilinear')

        self.outconv = Conv2D(out_ch, (1, 1), padding='same')

    def __call__(self, inputs):
        hx = inputs

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = self.upsample_6(hx6)
        # side6 = self.upsample_out_6(self.side6(hx6))

        # hx5d = self.stage5d(tf.concat([hx6up, hx5], axis=3))
        hx5d = self.stage5d(hx6up)
        hx5dup = self.upsample_5(hx5d)
        # side5 = self.upsample_out_5(self.side5(hx5d))

        # hx4d = self.stage4d(tf.concat([hx5dup, hx4], axis=3))
        hx4d = self.stage4d(hx5dup)
        hx4dup = self.upsample_4(hx4d)
        # side4 = self.upsample_out_4(self.side4(hx4d))

        # hx3d = self.stage3d(tf.concat([hx4dup, hx3], axis=3))
        hx3d = self.stage3d(hx4dup)
        hx3dup = self.upsample_3(hx3d)
        # side3 = self.upsample_out_3(self.side3(hx3d))

        # hx2d = self.stage2d(tf.concat([hx3dup, hx2], axis=3))
        hx2d = self.stage2d(hx3dup)
        hx2dup = self.upsample_2(hx2d)
        # side2 = self.upsample_out_2(self.side2(hx2d))

        # hx1d = self.stage1d(tf.concat([hx2dup, hx1], axis=3))
        hx1d = self.stage1d(hx2dup)
        side1 = self.side1(hx1d)

        sig = keras.activations.sigmoid
        # return sig(side1), hx6, hx5, hx4, hx3, hx2, hx1
        return sig(side1), hx6


#########################
#        ENCODER        #
#########################

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim, out_ch=1):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.stage1 = RSU7(16, 64)
        self.pool12 = MaxPool2D((2, 2), 2)

        self.stage2 = RSU6(32, 64)
        self.pool23 = MaxPool2D((2, 2), 2)

        self.stage3 = RSU5(64, 128)
        self.pool34 = MaxPool2D((2, 2), 2)

        self.stage4 = RSU4(128, 256)
        self.pool45 = MaxPool2D((2, 2), 2)

        self.stage5 = RSU4F(256, 512)
        self.pool56 = MaxPool2D((2, 2), 2)

    def __call__(self, conditional_input):
        # Encoder block 1
        hx = conditional_input

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        x = self.pool56(hx5)

        x = tf.keras.layers.Flatten() (x)

        x = tf.keras.layers.Dense(self.latent_dim * 2)(x)
        x = LeakyReLU()(x)

        return x


#########################
#        DECODER        #
#########################

class Decoder(tf.keras.Model):

    def __init__(self, batch_size=32, out_ch=1):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.dense = tf.keras.layers.Dense(4 * 4 * self.batch_size * 8)
        self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, self.batch_size * 8))

        self.stage5d = RSU4F(256, 512)
        self.stage4d = RSU4(128, 256)
        self.stage3d = RSU5(64, 128)
        self.stage2d = RSU6(32, 64)
        self.stage1d = RSU7(16, 64)

        self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def __call__(self, z_cond):
        # Reshape input
        x = self.dense(z_cond)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)

        hx5d = self.stage5d(x)
        hx5dup = self.upsample_5(hx5d)

        hx4d = self.stage4d(hx5dup)
        hx4dup = self.upsample_4(hx4d)

        hx3d = self.stage3d(hx4dup)
        hx3dup = self.upsample_3(hx3d)

        hx2d = self.stage2d(hx3dup)
        hx2dup = self.upsample_2(hx2d)

        hx1d = self.stage1d(hx2dup)
        x = self.side1(hx1d)
        x = sigmoid(x)

        return x


#########################
#       Conv-CVAE       #
#########################

class ConvCVAE(tf.keras.Model):

    def __init__(self,
                 encoder,
                 decoder,
                 label_dim,
                 latent_dim,
                 batch_size=32,
                 beta=1,
                 image_dim=(64, 64, 3)):
        super(ConvCVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.image_dim = image_dim

    def __call__(self, inputs):
        input_img, input_label, conditional_input = self.conditional_input(inputs)

        z_mean, z_log_var = tf.split(self.encoder(conditional_input), num_or_size_splits=2, axis=1)
        z_cond = self.reparametrization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond)

        recon_img = tf.nn.sigmoid(logits)

        # Loss computation #
        latent_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                                            axis=-1)  # KL divergence

        # очень странная метрика для изображений
        reconstr_loss = np.prod((64, 64)) * tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(input_img),
                                                                                tf.keras.backend.flatten(
                                                                                    recon_img))  # over weighted MSE
        loss = reconstr_loss + self.beta * latent_loss  # weighted ELBO loss
        loss = tf.reduce_mean(loss)

        return {
            'recon_img': recon_img,
            'latent_loss': latent_loss,
            'reconstr_loss': reconstr_loss,
            'loss': loss,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }

    def conditional_input(self, inputs):
        """ Builds the conditional input and returns the original input images, their labels and the conditional input."""

        input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype='float32')(inputs[0])
        input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype='float32')(inputs[1])
        labels = tf.reshape(inputs[1], [-1, 1, 1, self.label_dim])  # batch_size, 1, 1, label_size
        ones = tf.ones([inputs[0].shape[0]] + self.image_dim[0:-1] + [self.label_dim])  # batch_size, 64, 64, label_size
        labels = ones * labels  # batch_size, 64, 64, label_size
        conditional_input = tf.keras.layers.InputLayer(
            input_shape=(self.image_dim[0], self.image_dim[1], self.image_dim[2] + self.label_dim), dtype='float32')(
            tf.concat([inputs[0], labels], axis=3))

        return input_img, input_label, conditional_input

    def reparametrization(self, z_mean, z_log_var, input_label):
        """ Performs the riparametrization trick"""

        eps = tf.random.normal(shape=(input_label.shape[0], self.latent_dim), mean=0.0, stddev=1.0)
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1)  # (batch_size, label_dim + latent_dim)

        return z_cond
