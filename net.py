import tensorflow as tf
import numpy as np
from model import Model


class DCGAN(Model):
    def __init__(self, cfg):
        self.alpha = cfg.leakyRelu_alpha
        input_size, _, nc = cfg.input_shape
        self.res = cfg.resolution
        self.min_res = cfg.min_resolution
        # number of times to upsample/downsample for full resolution:
        self.n_scalings = int(np.log2(input_size / self.min_res))
        # number of times to upsample/downsample for current resolution:
        self.n_layers = int(np.log2(self.res / self.min_res))
        self.nf_min = cfg.nf_min  # min feature depth
        self.nf_max = cfg.nf_max  # max feature depth
        self.batch_size = cfg.batch_size
        Model.__init__(self, cfg)

    def leaky_relu(self, input_):
        return tf.maximum(self.alpha * input_, input_)

    def add_minibatch_stddev_feat(self, input_):
        _, h, w, _ = input_.get_shape().as_list()
        new_feat_shape = [self.cfg.batch_size, h, w, 1]

        mean, var = tf.nn.moments(input_, axes=[0], keep_dims=True)
        stddev = tf.sqrt(tf.reduce_mean(var, keep_dims=True))
        new_feat = tf.tile(stddev, multiples=new_feat_shape)
        return tf.concat([input_, new_feat], axis=3)

    def pixelwise_norm(self, a):
        return a / tf.sqrt(tf.reduce_mean(a * a, axis=3, keep_dims=True) + 1e-8)

    def conv2d(self, input_, n_filters, k_size, padding='same'):
        if not self.cfg.weight_scale:
            return tf.layers.conv2d(input_, n_filters, k_size, padding=padding)

        n_feats_in = input_.get_shape().as_list()[-1]
        fan_in = k_size * k_size * n_feats_in
        c = tf.constant(np.sqrt(2. / fan_in), dtype=tf.float32)
        kernel_init = tf.random_normal_initializer(stddev=1.)
        w_shape = [k_size, k_size, n_feats_in, n_filters]
        w = tf.get_variable('kernel', shape=w_shape, initializer=kernel_init)
        w = c * w
        strides = [1, 1, 1, 1]
        net = tf.nn.conv2d(input_, w, strides, padding=padding.upper())
        b = tf.get_variable('bias', [n_filters],
                            initializer=tf.constant_initializer(0.))
        net = tf.nn.bias_add(net, b)
        return net

    def up_sample(self, input_):
        _, h, w, _ = input_.get_shape().as_list()
        new_size = [2 * h, 2 * w]
        return tf.image.resize_nearest_neighbor(input_, size=new_size)

    def down_sample(self, input_):
        return tf.layers.average_pooling2d(input_, 2, 2)

    def conv_module(self, input_, n_filters, training, k_sizes=None,
                    norms=None, padding='same'):
        conv = input_
        if k_sizes is None:
            k_sizes = [3] * len(n_filters)
        if norms is None:
            norms = [None, None]

        # series of conv + lRelu + norm
        for i, (nf, k_size, norm) in enumerate(zip(n_filters, k_sizes, norms)):
            var_scope = 'conv_block_' + str(i+1)
            with tf.variable_scope(var_scope):
                conv = self.conv2d(conv, nf, k_size, padding=padding)
                conv = self.leaky_relu(conv)
                if norm == 'batch_norm':
                    conv = tf.layers.batch_normalization(conv, training=training)
                elif norm == 'pixel_norm':
                    conv = self.pixelwise_norm(conv)
                elif norm == 'layer_norm':
                    conv = tf.contrib.layers.layer_norm(conv)
        return conv

    def to_image(self, input_, resolution):
        nc = self.cfg.input_shape[-1]
        var_scope = '{0:}x{0:}'.format(resolution)
        with tf.variable_scope(var_scope + '/to_image'):
            out = self.conv2d(input_, nc, 1)
            return out

    def from_image(self, input_, n_filters, resolution):
        var_scope = '{0:}x{0:}'.format(resolution)
        with tf.variable_scope(var_scope + '/from_image'):
            out = self.conv2d(input_, n_filters, 1)
            return self.leaky_relu(out)

    def build_generator(self, training):
        z = self.tf_placeholders['z']
        z_dim = self.cfg.z_dim
        feat_size = self.min_res
        norm = self.cfg.norm_g

        with tf.variable_scope('generator', reuse=(not training)):
            net = tf.reshape(z, (-1, 1, 1, z_dim))
            padding = int(feat_size / 2)
            net = tf.pad(net, [[0, 0], [padding - 1, padding],
                               [padding - 1, padding], [0, 0]])
            feat_depth = min(self.nf_max, self.nf_min * 2 ** self.n_scalings)
            r = self.min_res
            var_scope = '{0:}x{0:}'.format(r)
            with tf.variable_scope(var_scope):
                net = self.conv_module(net, [feat_depth, feat_depth],
                                       training, k_sizes=[4, 3],
                                       norms=[None, norm])
            layers = []
            for i in range(self.n_layers):
                net = self.up_sample(net)
                n = self.nf_min * 2 ** (self.n_scalings - i - 1)
                feat_depth = min(self.nf_max, n)
                r *= 2
                var_scope = '{0:}x{0:}'.format(r)
                with tf.variable_scope(var_scope):
                    net = self.conv_module(net, [feat_depth, feat_depth],
                                           training, norms=[norm, norm])
                layers.append(net)

            # final layer:
            assert r == self.res, \
                '{:} not equal to {:}'.format(r, self.res)
            net = self.to_image(net, self.res)
            if self.cfg.transition:
                alpha = self.tf_placeholders['alpha']
                branch = layers[-2]
                branch = self.up_sample(branch)
                branch = self.to_image(branch, r / 2)
                net = alpha * net + (1. - alpha) * branch
            if self.cfg.use_tanh:
                net = tf.tanh(net)
            return net

    def build_discriminator(self, input_, reuse, training):
        norm = self.cfg.norm_d
        if (self.cfg.loss_mode == 'wgan_gp') and (norm == 'batch_norm'):
            norm = None
        with tf.variable_scope('discriminator', reuse=reuse):
            feat_depths = [min(self.nf_max, self.nf_min * 2 ** i)
                           for i in range(self.n_scalings)]
            r = self.res
            net = self.from_image(input_, feat_depths[-self.n_layers], r)
            for i in range(self.n_layers):
                feat_depth_1 = feat_depths[-self.n_layers + i]
                feat_depth_2 = min(self.nf_max, 2 * feat_depth_1)
                var_scope = '{0:}x{0:}'.format(r)
                with tf.variable_scope(var_scope):
                    net = self.conv_module(net, [feat_depth_1, feat_depth_2],
                                           training, norms=[norm, norm])
                net = self.down_sample(net)
                r /= 2
                # add a transition branch if required
                if i == 0 and self.cfg.transition:
                    alpha = self.tf_placeholders['alpha']
                    input_low = self.down_sample(input_)
                    idx = -self.n_layers + 1
                    branch = self.from_image(input_low, feat_depths[idx],
                                             self.res / 2)
                    net = alpha * net + (1. - alpha) * branch

            # add final layer
            assert r == self.min_res, \
                '{:} not equal to {:}'.format(r, self.min_res)
            net = self.add_minibatch_stddev_feat(net)
            feat_depth = min(self.nf_max, self.nf_min * 2 ** self.n_scalings)
            var_scope = '{0:}x{0:}'.format(r)
            with tf.variable_scope(var_scope):
                net = self.conv_module(net, [feat_depth, feat_depth],
                                       training, k_sizes=[3, 4],
                                       norms=[norm, None])
                net = tf.reduce_mean(net, axis=[1, 2])
                net = tf.reshape(net, [self.cfg.batch_size, feat_depth])
                net = tf.layers.dense(net, 1)
            return net

