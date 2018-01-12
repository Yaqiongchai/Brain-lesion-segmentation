import numpy as np
# import math
import tensorflow as tf
# from tensorflow.python.framework import ops
# from utils import *
## The packages originally there for deconv2D
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

K.set_image_data_format('channels_first')

def unet_model_2d(input_shape, downsize_filters_factor=1, pool_size=(2, 2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    conv1 = Conv2D(int(32/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(inputs)
    conv1 = Conv2D(int(64/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(int(64/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(pool1)
    conv2 = Conv2D(int(128/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(int(128/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(pool2)
    conv3 = Conv2D(int(256/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(int(256/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(pool3)
    conv4 = Conv2D(int(512/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv4)
    up5 = get_upconv2D(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                       nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv2D(int(256/downsize_filters_factor), (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(int(256/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv5)

    up6 = get_upconv2D(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                       nb_filters=int(256/downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv2D(int(128/downsize_filters_factor), (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(int(128/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv2D(int(64/downsize_filters_factor), (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(int(64/downsize_filters_factor), (3, 3), activation='relu',
                   padding='same')(conv7)

    conv8 = Conv2D(n_labels, (1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=SGD(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape[-2:],np.multiply(pool_size,depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv2D(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2), strides=(2, 2),
               deconvolution=False):
    if deconvolution:
        #     import Deconvolution3D
        # except ImportError:
        #     raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")
# commented by Y.C.: the deconv2D is defined below, not imported from Keras
# Website to refer to: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py#L89-L90
        # return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
        #                        output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
        #                                                                pool_size=pool_size, image_shape=image_shape),
        #                       strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
        #                                                                                depth=depth+1,
        #                                                                                pool_size=pool_size,
        #                                                                                image_shape=image_shape))
        return deconv2d(input_shape=compute_level_output_shape(filters=nb_filters,
                                                               depth=depth+1,
                                                               pool_size=pool_size,
                                                               image_shape=image_shape),
                        output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                pool_size=pool_size, image_shape=image_shape))
    else:
        return UpSampling2D(size=pool_size)

def deconv2d(input_shape, output_shape,k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_shape[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
    try:
        deconv = tf.nn.conv2d_transpose(input_shape, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
        deconv = tf.nn.deconv2d(input_shape, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    if with_w:
        return deconv, w, biases
    else:
        return deconv
