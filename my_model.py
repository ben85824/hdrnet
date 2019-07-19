import tensorflow as tf
import numpy as np
import os
from layers import conv, fc, bilateral_slice_apply

class hdrnet_model():
    def __init__(self):
        self.n_in = 4
        self.n_out = 3
    
    def model(self, lowres_input, fullres_input, is_training=True):
        with tf.variable_scope('coefficients'):
            bilateral_coeffs = self.coefficients(lowres_input, is_training)
            tf.add_to_collection('bilateral_coefficients', bilateral_coeffs)

        with tf.variable_scope('guide'):
            guide = self.guide(fullres_input, is_training)
            tf.add_to_collection('guide', guide)

        with tf.variable_scope('output'):
            output = self.output(fullres_input, guide, bilateral_coeffs)
            tf.add_to_collection('output', output)
        return output
    
    def coefficients(self, input_tensor, is_training = True):
        bs = input_tensor.get_shape().as_list()[0]
        gd = 8
        cm = 1
        spatial_bin = 16

        with tf.variable_scope('splat'):
            # log(256/16) = 4
            current_layer = input_tensor
            current_layer = conv(current_layer, 1*8*1, 3, stride=2,batch_norm=False, is_training=is_training,scope='conv0')
            current_layer = conv(current_layer, 1*8*2, 3, stride=2,batch_norm=True, is_training=is_training,scope='conv1')
            current_layer = conv(current_layer, 1*8*4, 3, stride=2,batch_norm=True, is_training=is_training,scope='conv2')
            current_layer = conv(current_layer, 1*8*8, 3, stride=2,batch_norm=True, is_training=is_training,scope='conv3')
            splat_features = current_layer
        
        with tf.variable_scope('global'):
            current_layer = splat_features
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=2,batch_norm=True, is_training=is_training,scope='conv0')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv1')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv2')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv3')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv4')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv5')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=2,batch_norm=True, is_training=is_training,scope='conv6')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv7')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv8')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv9')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv10')
            current_layer = conv(current_layer, 8*cm*gd, 3, stride=1,batch_norm=True, is_training=is_training,scope='conv11')
            _, lh, lw, lc = current_layer.get_shape().as_list()
            current_layer = tf.reshape(current_layer, [bs, lh*lw*lc])
            current_layer = fc(current_layer, 32*cm*gd,batch_norm=True, is_training=is_training,scope="fc1")
            current_layer = fc(current_layer, 16*cm*gd,batch_norm=True, is_training=is_training,scope="fc2")
            # don't normalize before fusion
            current_layer = fc(current_layer, 8*cm*gd, activation_fn=None, scope="fc3")
            global_features = current_layer
        
        with tf.variable_scope('local'):
            current_layer = splat_features
            current_layer = conv(current_layer, 8*cm*gd, 3, batch_norm=True, is_training=is_training,scope='conv0')
            current_layer = conv(current_layer, 8*cm*gd, 3, activation_fn=None,use_bias=False, scope='conv1')
            current_layer = conv(current_layer, 8*cm*gd, 3, activation_fn=None,use_bias=False, scope='conv2')
            current_layer = conv(current_layer, 8*cm*gd, 3, activation_fn=None,use_bias=False, scope='conv3')
            current_layer = conv(current_layer, 8*cm*gd, 3, activation_fn=None,use_bias=False, scope='conv4')
            current_layer = conv(current_layer, 8*cm*gd, 3, activation_fn=None,use_bias=False, scope='conv5')
            grid_features = current_layer
        
        with tf.name_scope('fusion'):
            fusion_grid = grid_features
            fusion_global = tf.reshape(global_features, [bs, 1, 1, 8*cm*gd])
            fusion = tf.nn.relu(fusion_grid+fusion_global)
        
        with tf.variable_scope('prediction'):
            current_layer = fusion
            current_layer = conv(current_layer, gd*self.n_out*self.n_in, 1,activation_fn=None, scope='conv0')
        
        with tf.name_scope('unroll_grid'):
            current_layer = tf.stack(tf.split(current_layer, self.n_out*self.n_in, axis=3), axis=4)
            current_layer = tf.stack(tf.split(current_layer, self.n_in, axis=4), axis=5)
        
        return  current_layer
        
    def guide(self, input_tensor, is_training = True):
        npts = 16  # number of control points for the curve
        nchans = input_tensor.get_shape().as_list()[-1]
        guidemap = input_tensor
        idtity = np.identity(nchans, dtype=np.float32) + np.random.randn(1).astype(np.float32)*1e-4
        ccm = tf.get_variable('ccm', dtype=tf.float32, initializer=idtity)
        with tf.name_scope('ccm'):
            ccm_bias = tf.get_variable('ccm_bias', shape=[nchans,], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            guidemap = tf.matmul(tf.reshape(input_tensor, [-1, nchans]), ccm)
            guidemap = tf.nn.bias_add(guidemap, ccm_bias, name='ccm_bias_add')
            guidemap = tf.reshape(guidemap, tf.shape(input_tensor))
        with tf.name_scope('curve'):
            shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
            shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, :]
            shifts_ = np.tile(shifts_, (1, 1, nchans, 1))

            guidemap = tf.expand_dims(guidemap, 4)
            shifts = tf.get_variable('shifts', dtype=tf.float32, initializer=shifts_)

            slopes_ = np.zeros([1, 1, 1, nchans, npts], dtype=np.float32)
            slopes_[:, :, :, :, 0] = 1.0
            slopes = tf.get_variable('slopes', dtype=tf.float32, initializer=slopes_)

            guidemap = tf.reduce_sum(slopes*tf.nn.relu(guidemap-shifts), reduction_indices=[4])
        guidemap = tf.contrib.layers.convolution2d(
            inputs=guidemap,
            num_outputs=1, kernel_size=1, 
            weights_initializer=tf.constant_initializer(1.0/nchans),
            biases_initializer=tf.constant_initializer(0),
            activation_fn=None, 
            variables_collections={'weights':[tf.GraphKeys.WEIGHTS], 'biases':[tf.GraphKeys.BIASES]},
            outputs_collections=[tf.GraphKeys.ACTIVATIONS],
            scope='channel_mixing')
        guidemap = tf.clip_by_value(guidemap, 0, 1)
        guidemap = tf.squeeze(guidemap, squeeze_dims=[3,])
        return guidemap
    
    def output(self, image, guide, coeffs):
        with tf.device('/gpu:0'):
            out = bilateral_slice_apply(coeffs, guide, image, has_offset=True, name='slice')
        return out

   