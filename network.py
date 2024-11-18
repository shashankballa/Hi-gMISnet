import os

# # Get the directory of the script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Change the working directory to the script's directory
# os.chdir(script_dir)

import random
import glob
import numpy
import pandas as pd
import natsort
import pywt
import albumentations as A
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from numpy import zeros, ones
from numpy.random import randint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG16, DenseNet201
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, concatenate, Dropout, 
    BatchNormalization, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Multiply, ELU, add
)
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_custom_objects, plot_model
from tensorflow.image import ssim
from tensorflow_wavelets.utils.helpers import *

tf.keras.backend.clear_session()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define a global variable for random seed
seed = 42

class DWT(layers.Layer):

    def __init__(self, wavelet_name='haar', concat=1, **kwargs):
        super(DWT, self).__init__(**kwargs)
        # self._name = self.name + "_" + name
        # get filter coeffs from 3rd party lib
        wavelet = pywt.Wavelet(wavelet_name)
        self.dec_len = wavelet.dec_len
        self.concat = concat
        # decomposition filter low pass and hight pass coeffs
        db2_lpf = wavelet.dec_lo
        db2_hpf = wavelet.dec_hi

        # covert filters into tensors and reshape for convolution math
        db2_lpf = tf.constant(db2_lpf[::-1])
        self.db2_lpf = tf.reshape(db2_lpf, (1, wavelet.dec_len, 1, 1))

        db2_hpf = tf.constant(db2_hpf[::-1])
        self.db2_hpf = tf.reshape(db2_hpf, (1, wavelet.dec_len, 1, 1))

        self.conv_type = "VALID"
        self.border_padd = "SYMMETRIC"
        self.wavelet_name = wavelet_name
        self.concat = concat

    def build(self, input_shape):
        # filter dims should be bigger if input is not gray scale
        if input_shape[-1] != 1:
            # self.db2_lpf = tf.repeat(self.db2_lpf, input_shape[-1], axis=-1)
            self.db2_lpf = tf.keras.backend.repeat_elements(self.db2_lpf, input_shape[-1], axis=-1)
            # self.db2_hpf = tf.repeat(self.db2_hpf, input_shape[-1], axis=-1)
            self.db2_hpf = tf.keras.backend.repeat_elements(self.db2_hpf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # symmetric column padding
        inputs_pad = tf.pad(inputs, [[0, 0], [0, 0], [self.dec_len-1, self.dec_len-1], [0, 0]], self.border_padd)

        # approximation conv only rows
        a = tf.nn.conv2d(
            inputs_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1], name="row_approx"
        )
        # details conv only rows
        d = tf.nn.conv2d(
            inputs_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1], name="row_detail"
        )
        # ds - down sample
        a_ds = a[:, :, 1:a.shape[2]:2, :]
        d_ds = d[:, :, 1:d.shape[2]:2, :]

        # symmetric row padding
        a_ds_pad = tf.pad(a_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)
        d_ds_pad = tf.pad(d_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)

        # convolution is done on the rows so we need to
        # transpose the matrix in order to convolve the colums
        a_ds_pad = tf.transpose(a_ds_pad, perm=[0, 2, 1, 3])
        d_ds_pad = tf.transpose(d_ds_pad, perm=[0, 2, 1, 3])

        # aa approximation approximation
        aa = tf.nn.conv2d(
            a_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1], name="LL"
        )
        # ad approximation details
        ad = tf.nn.conv2d(
            a_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1], name="LH"
        )
        # ad details aproximation
        da = tf.nn.conv2d(
            d_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1], name="HL"
        )
        # dd details details
        dd = tf.nn.conv2d(
            d_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1], name="HH"
        )

        # transpose back the matrix
        aa = tf.transpose(aa, perm=[0, 2, 1, 3], name="LL")
        ad = tf.transpose(ad, perm=[0, 2, 1, 3], name="LH")
        da = tf.transpose(da, perm=[0, 2, 1, 3], name="HL")
        dd = tf.transpose(dd, perm=[0, 2, 1, 3], name="HH")

        # down sample
        ll = aa[:, 1:aa.shape[1]:2, :, :]
        lh = ad[:, 1:ad.shape[1]:2, :, :]
        hl = da[:, 1:da.shape[1]:2, :, :]
        hh = dd[:, 1:dd.shape[1]:2, :, :]

        # concate all outputs ionto tensor
        if self.concat == 0:
            x = tf.concat([ll, lh, hl, hh], axis=-1, name="LL_LH_HL_HH")
        elif self.concat == 2:
            x = ll
        elif self.concat ==1:
            return ll,lh,hl,hh
        else:
            x = tf.concat([tf.concat([ll, lh], axis=1), tf.concat([hl, hh], axis=1)], axis=2, name="LL_LH-HL_HH")
        return x

    def get_config(self):
        config = super(DWT, self).get_config()
        config.update({'wavelet_name': self.wavelet_name, 'concat': self.concat})
        return config

# Post wavelet conv block
def wavelet_conv_block(x, in_channels,name_prefix=''):

    _np = name_prefix

    if _np != '':
        _np = _np + '_'

    ll, lh, hl, hh = DWT(concat=1)(x)
    
    y = tf.concat([ll, lh, hl, hh], axis=3)
    
    conv1 = tf.keras.layers.Conv2D(in_channels * 2, kernel_size=1, dilation_rate=1, padding='valid', name=_np+'Post-DWT-in-c1')(y)
    conv2 = tf.keras.layers.Conv2D(in_channels, kernel_size=3, dilation_rate=1, padding='same', name=_np+'Post-DWT-c1-c2')(conv1)
    conv2 = tf.keras.layers.BatchNormalization(name=_np+'Post-DWT-c2-bn2')(conv2,training=True)
    conv2 = tf.keras.layers.ReLU(name=_np+'Post-DWT-bn2-r2')(conv2)

    
    conv3 = tf.keras.layers.Conv2D(in_channels, kernel_size=5, dilation_rate=1, padding='same', name=_np+'Post-DWT-c1-c3')(conv1)
    conv3 = tf.keras.layers.BatchNormalization(name=_np+'Post-DWT-c3-bn3')(conv3,training=True)
    conv3 = tf.keras.layers.ReLU(name=_np+'Post-DWT-bn3-r3')(conv3)
    
    conv4 = tf.keras.layers.Conv2D(in_channels * 2, kernel_size=1, dilation_rate=1, padding='valid',
    name= _np + 'Post-DWT-r2_r3-out0')(tf.concat([conv2, conv3], axis=3))
    
    return conv4, ll, lh, hl, hh   

# Dual-mode Attention Gate (DAG)
def conv_block(x, filter_size, size, dropout, batch_norm=False, name_prefix=''):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same", name=name_prefix + '_in-c1')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3, name=name_prefix + '_c1-bn1')(conv,training=True)
        conv = layers.ReLU(name=name_prefix + '_bn1-r1')(conv)
    else:
        conv = layers.ReLU(name=name_prefix + '_c1-r1')(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same", name=name_prefix + '_r1-c2')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3, name=name_prefix + '_c2-bn2')(conv,training=True)
        conv = layers.ReLU(name=name_prefix + '_bn2-r2')(conv) 
    else:
        conv = layers.ReLU(name=name_prefix + '_c2-r2')(conv)  
    
    if dropout > 0:
        conv = layers.Dropout(dropout, name=name_prefix + '_r2-out0')(conv)

    return conv

def repeat_elem(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),arguments={'repnum': rep})(tensor)

def res_conv_block(x, filter_size, size, dropout, batch_norm=False, name_prefix=''):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same', name=name_prefix + '_in-c1')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3, name=name_prefix + '_c1-bn1')(conv,training=True)
        conv = layers.ReLU(name=name_prefix + '_bn1-r1')(conv)
    else:
        conv = layers.ReLU(name=name_prefix + '_c1-r1')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same', name=name_prefix + '_r1-c2')(conv)
    _ostr2 = 'c2'
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3, name=name_prefix + '_c2-bn2')(conv,training=True)
        _ostr2 = 'bn2'

    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout, name=name_prefix+'_'+_ostr2+'-do2')(conv)
        _ostr2 = 'do1'

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same', name=name_prefix + '_in-c3')(x)
    _ostr3 = 'c3'
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3, name=name_prefix + '_c3-bn3')(shortcut,training=True)
        _ostr3 = 'bn3'
    
    res_path = layers.add([shortcut, conv], name=name_prefix + '_'+_ostr3+'_'+_ostr2+'-add1')
    res_path = layers.ReLU(name=name_prefix + '_'+_ostr3+'_'+_ostr2+'-out0')(res_path)  #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    init = RandomNormal(stddev=0.02, seed=seed)
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization(axis=3)(x,training=True)
    x = layers.ReLU()(x)
    return x

def attention_block(x, gating, inter_shape, name_prefix=''):
    filters = x.shape[-1]
    filtersg = gating.shape[-1]
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    init = RandomNormal(stddev=0.02, seed=seed)
    xa = x[:, :, :, :filters // 2]
    xb = x[:, :, :, filters // 2: ]
    gating_a = gating[:, :, :, :filtersg // 2]
    gating_b = gating[:, :, :, filtersg // 2:]
# Getting the x signal to the same shape as the gating signal
    theta_xa = layers.Conv2D(inter_shape//2, (2, 2), strides=(2, 2), padding='same',name=name_prefix + 'theta_a')(xa)  # 16
    shape_theta_xa = K.int_shape(theta_xa)
    theta_xb = layers.Conv2D(inter_shape//2, (2, 2), strides=(2, 2), padding='same',name=name_prefix + 'theta_b')(xb)  # 16
    shape_theta_xb = K.int_shape(theta_xb)
# Getting the gating signal to the same number of filters as the inter_shape
    phi_ga = layers.Conv2D(inter_shape//2, (1, 1), padding='same')(gating_a)
    upsample_ga = layers.Conv2DTranspose(inter_shape//2, (3, 3),strides=(shape_theta_xa[1] // shape_g[1], shape_theta_xa[2] // shape_g[2]),padding='same',name=name_prefix + 'phi_ga')(phi_ga)  # 16
    phi_gb = layers.Conv2D(inter_shape//2, (1, 1), padding='same')(gating_b)
    upsample_gb = layers.Conv2DTranspose(inter_shape//2, (3, 3),strides=(shape_theta_xb[1] // shape_g[1], shape_theta_xb[2] // shape_g[2]),padding='same',name=name_prefix + 'phi_gb')(phi_gb)  # 16
    
    
    ###################################################
    concat_xg = layers.add([theta_xa,upsample_ga ],name=name_prefix + 'foreground_add')
    act_xg = layers.ReLU()(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi1 = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]),name=name_prefix + 'visual_fore')(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi1, xa.shape[3])
    ya = layers.multiply([upsample_psi, xa],name=name_prefix + 'foreground_out')

    
    ##################################################
    subtract_xg = layers.subtract([theta_xb,upsample_gb],name=name_prefix + 'background_add')
    sub_act_xg = layers.ReLU()(subtract_xg)
    sub_psi = layers.Conv2D(1, (1, 1), padding='same')(sub_act_xg)
    sub_sigmoid_xg = layers.Activation('sigmoid',name=name_prefix + 'before_reverse')(sub_psi)
    sub_sigmoid_xg = -1 * (sub_sigmoid_xg) + 1
    sub_upsample_psi1 = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]),name=name_prefix +'visual_back' )(sub_sigmoid_xg)  # 32
    sub_upsample_psi = repeat_elem(sub_upsample_psi1, xb.shape[3])
    yb = layers.multiply([sub_upsample_psi, xb],name=name_prefix + 'background_out')
    ##################################################
    y = layers.Concatenate(axis=3)([ya, yb])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization(axis=3)(result,training=True)
    result_bn = layers.ReLU(name=name_prefix + 'attention_out')(result_bn)
    return result_bn,upsample_psi1,sub_upsample_psi1

# Other Blocks
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1),activation='relu', name=None):

    init = RandomNormal(stddev=0.02, seed=seed)
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding,kernel_initializer=init )(x)
    x = BatchNormalization(axis=3)(x, training=True)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None,dropout=True):

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x,training=True)
    if dropout:
        x = Dropout(0.5)(x, training=True)
    
    return x

def MultiResBlock(U, inp, alpha = 1.67):

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out,training=True)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out,training=True)

    return out

def ResPath(filters, length, inp):

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out,training=True)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,activation=None, padding='same')
        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out,training=True)

    return out

def BasicConv2D(inputs,out_planes, kernel_size, stride=1, padding='same', dilation=1):
    conv = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            use_bias=False
        )(inputs)
    bn = tf.keras.layers.BatchNormalization()(conv,training=True)
    relu = tf.keras.layers.ReLU()(bn)
    return relu
    
def RFBModified(inputs, out_channel):
    relu = tf.keras.layers.ReLU()(inputs)

    # Define branch0
    branch0 = BasicConv2D(relu, out_channel, kernel_size=1)

    # Define branch1
    conv1_1x1 = BasicConv2D(relu, out_channel, kernel_size=1)
    conv1_1x3 = BasicConv2D(conv1_1x1, out_channel, kernel_size=(1, 3), padding='same')
    conv1_3x1 = BasicConv2D(conv1_1x3, out_channel, kernel_size=(3, 1), padding='same')
    conv1_3x3 = BasicConv2D(conv1_3x1, out_channel, kernel_size=3, padding='same', dilation=3)

    # Define branch2
    conv2_1x1 = BasicConv2D(relu, out_channel, kernel_size=1)
    conv2_1x5 = BasicConv2D(conv2_1x1, out_channel, kernel_size=(1, 5), padding='same')
    conv2_5x1 = BasicConv2D(conv2_1x5, out_channel, kernel_size=(5, 1), padding='same')
    conv2_3x3 = BasicConv2D(conv2_5x1, out_channel, kernel_size=3, padding='same', dilation=5)

    # Define branch3
    conv3_1x1 = BasicConv2D(relu, out_channel, kernel_size=1)
    conv3_1x7 = BasicConv2D(conv3_1x1, out_channel, kernel_size=(1, 7), padding='same')
    conv3_7x1 = BasicConv2D(conv3_1x7, out_channel, kernel_size=(7, 1), padding='same')
    conv3_3x3 = BasicConv2D(conv3_7x1, out_channel, kernel_size=3, padding='same', dilation=7)

    # Concatenate branches
    branches_concat = tf.keras.layers.Concatenate(axis=-1)([branch0, conv1_3x3, conv2_3x3, conv3_3x3])

    # Final convolution and residual connection
    conv_cat = BasicConv2D(branches_concat, out_channel, kernel_size=3, padding='same')
    conv_res = BasicConv2D(relu, out_channel, kernel_size=1)

    # Output
    output = tf.keras.layers.ReLU()(conv_cat + conv_res)

    return output

def aggregation(x1, x2, x3):
    channel=32
    upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

    x1_1 = x1
    x1_1_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_1)
    x1_1_2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_1_1)
    x2_1_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x2)
    x2_1 = tf.math.multiply(BasicConv2D(x1_1_1 ,channel, 3, padding='same'), x2)
    x3_1 = BasicConv2D(x1_1_2,channel, 3, padding='same') 
    x3_1 = tf.math.multiply(x3_1 , BasicConv2D(x2_1_1,channel, 3, padding='same'))
    x3_1 = tf.math.multiply(x3_1, x3)

    x2_2 = tf.concat([x2_1, BasicConv2D(upsample(x1_1),2*channel, 3, padding='same')], axis=-1)
    x2_2 = BasicConv2D(x2_2,2 * channel, 3, padding='same')

    x3_2 = tf.concat([x3_1, BasicConv2D(upsample(x2_2),2 * channel, 3, padding='same')], axis=-1)
    x3_2 = BasicConv2D(x3_2,3 * channel, 3, padding='same')

    x = BasicConv2D(x3_2,3 * channel, 3, padding='same')
    x = tf.keras.layers.Conv2D(1, 1)(x)
    
    return x

# GAN Architecture
def define_discriminator(image_shape):
    
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=seed) 
    # source image input
    in_src_image = Input(shape=image_shape) 
    # target image input
    in_target_image = Input(shape=(512,512,1)) 
    
    # concatenate images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    
    # C64: 4x4 kernel Stride 2x2
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 2x2
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization()(d)
    
    # C256: 4x4 kernel Stride 2x2
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization()(d)
    
    # C512: 4x4 kernel Stride 2x2 
    # Not in the original paper. Comment this block if you want.
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization()(d)
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(learning_rate=0.000009, beta_1=0.5)
    model.compile(loss='mae', optimizer=opt, loss_weights=[0.5])
    return model

def define_generator(height, width, n_channels,name_prefix='f_'):
    inputs = Input((height, width, n_channels))
    encoder = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    encoder1 = DenseNet201(include_top=False, weights="imagenet", input_tensor=inputs)
                           
    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    
    ######################################
    wav_pool256 = DWT(name="haar",concat=0)(inputs)
    s2 = encoder.get_layer("block2_conv2").output 
    e2 = encoder1.get_layer("conv1/relu").output
    mresblock2 = Concatenate()([pool1,s2,e2])
    mresblock2 = MultiResBlock(64, mresblock2)
    mresblock2 = Concatenate()([wav_pool256, mresblock2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(64, 3, mresblock2)

    ###########################################
    mres_wav_pool256= DWT(concat=2)(mresblock1)
    wav_cnn128 = Concatenate(name= name_prefix + 'wav_2_in')([wav_pool256, mresblock2,mres_wav_pool256])
    wav_cnn128,_,_,_,_ = wavelet_conv_block(wav_cnn128, 16,name_prefix= name_prefix + 'wav_2')
    
    wav_out1_ll = DWT(concat=2)(inputs)
    wav_pool128 = DWT(concat=0)(wav_out1_ll)
    
    s3 = encoder.get_layer("block3_conv2").output
    e3 = encoder1.get_layer("pool2_conv").output
    mresblock3 = Concatenate()([pool2,s3,e3])
    mresblock3 = MultiResBlock(128, mresblock3)
    mresblock3 = Concatenate()([wav_cnn128, mresblock3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(128, 2, mresblock3)

    #############################################
    
    
    mres_wav_pool128 = DWT(concat=2)(mresblock2)
    wav_cnn64 = Concatenate()([wav_pool128, mresblock3,mres_wav_pool128])
    wav_cnn64,_,_,_,_ = wavelet_conv_block(wav_cnn64, 32,name_prefix= name_prefix + 'wav_3')    
    
    wav_out2_ll = DWT(concat=2)(wav_out1_ll)
    wav_pool64 = DWT(concat=0)(wav_out2_ll)
    
    s4 = encoder.get_layer("block4_conv2").output
    e4 = encoder1.get_layer("pool3_conv").output
    mresblock4= Concatenate()([pool3,s4,e4])
    mresblock4 = MultiResBlock(256, mresblock4)
    mresblock4= Concatenate()([wav_cnn64, mresblock4])
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(256, 1, mresblock4)
    
    
    ###############################################
    
    mres_wav_pool64 = DWT(concat=2)(mresblock3)
    wav_cnn32 = Concatenate()([wav_pool64, mresblock4,mres_wav_pool64])
    wav_cnn32,_,_,_,_ = wavelet_conv_block(wav_cnn32, 64,name_prefix= name_prefix + 'wav_4')
    
    s5 = encoder.get_layer("block5_conv2").output
    e5 = encoder1.get_layer("pool4_conv").output
    mresblock5= Concatenate()([pool4,s5,e5])
    mresblock5 = MultiResBlock(512, mresblock5)
    mresblock5= Concatenate()([wav_cnn32, mresblock5])
    pool5 = MaxPooling2D(pool_size=(2, 2))(mresblock5)
    mresblock5 = ResPath(512, 1, mresblock5)
    ##############################

    mresblock6 = MultiResBlock(1024, pool5)

    ###########################################
    
    gating_8 = gating_signal(mresblock6, 512, batch_norm=True)
    att_8,_,_ = attention_block(mresblock5, gating_8, 512,name_prefix = name_prefix + 'at_1')
    up5 = concatenate([trans_conv2d_bn(mresblock6, filters=512, num_row=2, num_col=2, padding='same', strides=(2, 2), dropout=True), att_8], axis=3)
    mresblock7 = MultiResBlock(512, up5) 
    
    
    
    
    gating_16 = gating_signal(mresblock7, 256, batch_norm=True)
    att_16,_,_ = attention_block(mresblock4, gating_16, 256,name_prefix = name_prefix + 'at_2')
    up6 = concatenate([trans_conv2d_bn(mresblock7, filters=256, num_row=2, num_col=2, padding='same', strides=(2, 2), dropout=True), att_16], axis=3)
    mresblock8 = MultiResBlock(256, up6)
    
    

    gating_32 = gating_signal(mresblock8, 128, batch_norm=True)
    att_32,at3_fore,at3_back = attention_block(mresblock3, gating_32, 128,name_prefix = name_prefix + 'at_3')
    up7 = concatenate([trans_conv2d_bn(mresblock8, filters=128, num_row=2, num_col=2, padding='same', strides=(2, 2), dropout=True), att_32], axis=3)
    mresblock9 = MultiResBlock(128, up7)


    
    
    
    gating_64 = gating_signal(mresblock9, 64, batch_norm=True)
    att_64,at4_fore,at4_back = attention_block(mresblock2, gating_64, 64,name_prefix = name_prefix + 'at_4')
    up8 = concatenate([trans_conv2d_bn(mresblock9, filters=64, num_row=2, num_col=2, padding='same', strides=(2, 2), dropout=False), att_64], axis=3)
    mresblock10 = MultiResBlock(64, up8)

    
    

    
    
    gating_128 = gating_signal(mresblock10, 32, batch_norm=True)
    att_128,at5_fore,at5_back = attention_block(mresblock1, gating_128, 32,name_prefix = name_prefix + 'at_5')
    up9 = concatenate([trans_conv2d_bn(mresblock10, filters=32, num_row=2, num_col=2, padding='same', strides=(2, 2), dropout=False), att_128], axis=3)
    mresblock11 = MultiResBlock(32, up9)
    
    
    init = RandomNormal(stddev=0.02, seed=seed)
    conv10 = conv2d_bn(mresblock11, 1, 1, 1, activation='sigmoid')
                           
    
    x3_rfb=RFBModified(mresblock3,32)
    x4_rfb=RFBModified(mresblock4,32)
    x5_rfb=RFBModified(mresblock5,32)

    ra5_feat = aggregation(x5_rfb,x4_rfb,x3_rfb)


    crop_5 = tf.image.resize(ra5_feat, [32,32])
    x = -1 * (tf.math.sigmoid(crop_5)) + 1
    x = tf.keras.layers.Multiply()([x, mresblock5])
    
    x = BasicConv2D(x, 256, 1)
    x = BasicConv2D(x, 256, 5, padding='same')
    x = BasicConv2D(x, 256, 5, padding='same')
    x = BasicConv2D(x, 256, 5, padding='same')
    ra4_feat = BasicConv2D(x, 1, 1)
    x = tf.keras.layers.Add()([ra4_feat, crop_5])
    crop_4 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = -1 * (tf.math.sigmoid(crop_4)) + 1
    x = tf.keras.layers.Multiply()([x, mresblock4])
    x = BasicConv2D(x, 64, 1)
    x = BasicConv2D(x, 64, 3, padding='same')
    x = BasicConv2D(x, 64, 3, padding='same')
    ra3_feat = BasicConv2D(x, 1, 3, padding='same')
    x = tf.keras.layers.Add()([ra3_feat, crop_4])

    crop_3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = -1 * (tf.math.sigmoid(crop_3)) + 1
    x = tf.keras.layers.Multiply()([x, mresblock3])
    x = BasicConv2D(x, 64, 1)
    x = BasicConv2D(x, 64, 3, padding='same')
    x = BasicConv2D(x, 64, 3, padding='same')
    ra2_feat = BasicConv2D(x, 1, 3, padding='same')
    x = tf.keras.layers.Add()([ra2_feat, crop_3])
    lateral_map_2 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    lateral_map_2 = tf.keras.activations.sigmoid(lateral_map_2)

    out_image=conv10*lateral_map_2

    model = Model(inputs, [out_image,at5_fore,at5_back,at4_fore,at4_back,at3_fore,at3_back])
    

    return model

def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False       #Descriminator layers set to untrainable in the combined GAN but 
                                                #standalone descriminator will be trainable.
            
    # define the source image
    in_src = Input(shape=image_shape)
    # suppy the image as input to the generator 
    gen_out = g_model(in_src)
    # supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out[0]])
    # src image as input, generated image and disc. output as outputs
    model = Model(in_src, [dis_out, gen_out[0], gen_out[1], gen_out[2], gen_out[3], gen_out[4], gen_out[5], gen_out[6]])
    # compile model
    opt = Adam(learning_rate=0.00008, beta_1=0.5)
    
    model.compile(loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'],optimizer=opt, loss_weights=[1,25,5,5,5,5,3,3])
    return model

# Loss functions
def log_ssim_mse_loss(y_true, y_pred, alpha=0.4):
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = -tf.math.log(ssim) * alpha + mse * (1 - alpha)
    return loss*5

def boundary_loss(y_true, y_pred):
    # Compute the gradient of the predicted and ground truth masks
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

    # Compute the boundary term of the loss function
    term_1 = tf.abs(tf.reduce_mean(tf.abs(dy_true) - tf.abs(dy_pred)))
    term_2 = tf.abs(tf.reduce_mean(tf.abs(dx_true) - tf.abs(dx_pred)))

    # Return the sum of the two terms as the boundary loss
    return (term_1 + term_2)*5

def mae_loss(y_true, y_pred):
    return mean_absolute_error(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]))

# Augmentation functions
aug1 = A.HorizontalFlip(p=1)
aug2 = A.VerticalFlip(p=1)
aug3 = A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)
aug4 = A.Blur(blur_limit=11, always_apply=True, p=1)
# @SZB: No Cutout() in albumentations, changed to CoarseDropout()
aug5 = A.CoarseDropout(max_holes=8, max_height=96, max_width=96, fill_value=0, always_apply=True, p=1)
aug6 = A.Rotate(limit=90, interpolation=1, border_mode=2, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=True, p=1)
aug7 = A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=True, p=1)
aug8 = A.RandomBrightnessContrast (brightness_limit=0.4, contrast_limit=0.3, brightness_by_max=True, always_apply=True, p=1)
aug9 = A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.4, val_shift_limit=0.3, always_apply=True, p=1)
aug10 = A.Affine(scale=(0.2,0.3), translate_percent=0.2, rotate=(-30,30), shear=(-45,45), always_apply=True, p=1)

# I used one augmentation type at a time, you can do multiple
def augment(image,mask,n):
    if n==0:
        augmented = aug10(image=image, mask=mask)
    elif n==1:
        augmented = aug1(image=image, mask=mask)
    elif n==2: 
        augmented = aug2(image=image, mask=mask)
    elif n==3: 
        augmented = aug3(image=image, mask=mask)
    elif n==6:
        augmented = aug6(image=image, mask=mask)
    elif n==4: 
        augmented = aug4(image=image, mask=mask)
    elif n==5: 
        augmented = aug5(image=image, mask=mask)
    elif n==7:
        return image,mask
    elif n==8:
        augmented = aug7(image=image, mask=mask)
    elif n==9:
        augmented = aug8(image=image, mask=mask)
    elif n==10: 
        augmented = aug6(image=image, mask=mask)
    else: 
        augmented = aug9(image=image, mask=mask)
    
        
    image_aug= augmented['image']
    mask_aug = augmented['mask']
    return image_aug,mask_aug

def generate_real_samples(data_generator,mask_generator, n_samples,patch_shape, i):
    ix = randint(0, len(data_generator), n_samples)
    n = i % 12

    # Initialize lists to store augmented images and masks
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    X6 = []
    X7 = []


    for i in ix:
        augmented_image, augmented_mask = augment(
            np.reshape(data_generator[i], [512,512, 3]),  # Input image
            np.reshape(mask_generator[i], [512,512, 1]),  # Input mask
            n  # Augmentation type
        )
        X1.append(augmented_image)
        X2.append(augmented_mask)
        X3.append(1-augmented_mask)
        X4.append(tf.image.resize(augmented_mask,(256,256)))
        X5.append(tf.image.resize(1-augmented_mask,(256,256)))
        X6.append(tf.image.resize(augmented_mask,(128,128)))
        X7.append(tf.image.resize(1-augmented_mask,(128,128)))
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6), np.array(X7)],y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((2, patch_shape, patch_shape, 1))
    return X[0], y

def summarize_performance(step, g_model1,g_model2):

    filename2 = 'lesion_model_%06d.h5' % (step+1)
    g_model1.save(filename2)
    filename4 = 'background_model_%06d.h5' % (step+1)
    g_model2.save(filename4)
    print('>Saved: %s and %s' % (filename2,filename4))

def remove_prev_performance(score_epoch):
    filename3 = 'lesion_model_%06d.h5' % (score_epoch)
    filename5 = 'background_model_%06d.h5' % (score_epoch)
    
    filepath3 = "training_data/" + filename3
    filepath5 = "training_data/" + filename5
    
    if os.path.exists(filepath3):
        os.remove(filepath3)
        print(f'>Removed: {filename3}')
    else:
        print(f'>File {filename3} does not exist')
        
    if os.path.exists(filepath5):
        os.remove(filepath5)
        print(f'>Removed: {filename5}')
    else:
        print(f'>File {filename5} does not exist')