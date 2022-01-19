import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

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
        self.conv = Conv2D(out_ch, (3, 3), strides=1, padding='same', dilation_rate=dirate)
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
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

    def call(self, inputs):
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

    def call(self, inputs):
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

    def call(self, inputs):
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

    def call(self, inputs):
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

    def call(self, inputs):
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

    def call(self, inputs):
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

        # def call(self, inputs):
        #     hx = inputs
        #
        #     hx1 = self.stage1(hx)
        #     hx = self.pool12(hx1)
        #
        #     hx2 = self.stage2(hx)
        #     hx = self.pool23(hx2)
        #
        #     hx3 = self.stage3(hx)
        #     hx = self.pool34(hx3)
        #
        #     hx4 = self.stage4(hx)
        #     hx = self.pool45(hx4)
        #
        #     hx5 = self.stage5(hx)
        #     hx = self.pool56(hx5)
        #
        #     hx6 = self.stage6(hx)
        #     hx6up = self.upsample_6(hx6)
        #     side6 = self.upsample_out_6(self.side6(hx6))
        #
        #     hx5d = self.stage5d(tf.concat([hx6up, hx5], axis=3))
        #     hx5dup = self.upsample_5(hx5d)
        #     side5 = self.upsample_out_5(self.side5(hx5d))
        #
        #     hx4d = self.stage4d(tf.concat([hx5dup, hx4], axis=3))
        #     hx4dup = self.upsample_4(hx4d)
        #     side4 = self.upsample_out_4(self.side4(hx4d))
        #
        #     hx3d = self.stage3d(tf.concat([hx4dup, hx3], axis=3))
        #     hx3dup = self.upsample_3(hx3d)
        #     side3 = self.upsample_out_3(self.side3(hx3d))
        #
        #     hx2d = self.stage2d(tf.concat([hx3dup, hx2], axis=3))
        #     hx2dup = self.upsample_2(hx2d)
        #     side2 = self.upsample_out_2(self.side2(hx2d))
        #
        #     hx1d = self.stage1d(tf.concat([hx2dup, hx1], axis=3))
        #     side1 = self.side1(hx1d)
        #
        #
        #     sig = keras.activations.sigmoid
        #
        #     return tf.stack([sig(fused_output), sig(side1), sig(side2), sig(side3), sig(side4), sig(side5), sig(side6)])



from tensorflow.keras import backend
from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras import layers
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import data_utils

WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/'
                'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg19/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')



def VGG19(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the VGG19 architecture.
    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
        https://arxiv.org/abs/1409.1556) (ICLR 2015)
    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).
    The default input size for this model is 224x224.
    Note: each Keras Application expects a specific kind of input preprocessing.
    For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your
    inputs before passing them to the model.
    `vgg19.preprocess_input` will convert the input images from RGB to BGR,
    then will zero-center each color channel with respect to the ImageNet dataset,
    without scaling.
    Args:
      include_top: whether to include the 3 fully-connected
        layers at the top of the network.
      weights: one of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)`
        (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.  '
                         f'Received: `weights={weights}.`')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000.  '
                         f'Received: `classes={classes}.`')


    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)


    inputs = img_input

    model = tf.keras.Model(inputs, x, name='vgg19')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = data_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = data_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

