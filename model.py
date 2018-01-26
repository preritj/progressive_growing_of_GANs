import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import losses
import time
from utils import ImageLoader


class Model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tf_placeholders = {}
        self.create_tf_placeholders()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.d_train_op, self.g_train_op = None, None
        self.ema_op, self.ema_vars = None, {}
        self.d_loss, self.g_loss = None, None
        self.gen_images, self.eval_op = None, None
        self.image_loader = ImageLoader(self.cfg)

    def create_tf_placeholders(self):
        h, w, c = self.cfg.input_shape
        z_dim = self.cfg.z_dim
        z = tf.placeholder(tf.float32, [None, z_dim])
        learning_rate = tf.placeholder(tf.float32)
        alpha = tf.placeholder(tf.float32, shape=())
        self.tf_placeholders = {'z': z,
                                'learning_rate': learning_rate,
                                'alpha': alpha}

    def resize_image(self, image):
        _, input_size, _, _ = image.get_shape().as_list()
        res = self.cfg.resolution
        if input_size == res:
            return image
        new_size = [res, res]
        new_img = tf.image.resize_nearest_neighbor(image, size=new_size)
        if self.cfg.transition:
            alpha = self.tf_placeholders['alpha']
            low_res_img = tf.layers.average_pooling2d(new_img, 2, 2)
            low_res_img = \
                tf.image.resize_nearest_neighbor(low_res_img, size=new_size)
            new_img = alpha * new_img + (1. - alpha) * low_res_img
        return new_img

    def build_generator(self, training):
        raise NotImplementedError("Not yet implemented")

    def build_encoder(self, training):
        raise NotImplementedError("Not yet implemented")

    def build_discriminator(self, input_, reuse, training):
        raise NotImplementedError("Not yet implemented")

    def make_train_op(self, images):
        images_real = images
        tf.summary.image('images_real_original_size', images_real, 8)
        images_real = self.resize_image(images_real)
        tf.summary.image('images_real', images_real, 8)

        d_real = self.build_discriminator(images_real, reuse=False,
                                          training=True)

        images_fake = self.build_generator(training=True)
        tf.summary.image('images_fake', images_fake, 8)

        d_fake = self.build_discriminator(images_fake, reuse=True,
                                          training=True)

        d_loss, g_loss = None, None
        if self.cfg.loss_mode == 'js':
            smooth_factor = 0.9 if self.cfg.smooth_label else 1.
            d_loss, g_loss = losses.js_loss(d_real, d_fake, smooth_factor)
        elif self.cfg.loss_mode == 'wgan_gp':
            d_loss, g_loss = losses.wgan_loss(d_real, d_fake)
            # Gradient penalty
            lambda_gp = self.cfg.lambda_gp
            gamma_gp = self.cfg.gamma_gp
            batch_size = self.cfg.batch_size
            nc = self.cfg.input_shape[-1]
            res = self.cfg.resolution
            input_shape = [batch_size, res, res, nc]
            alpha = tf.random_uniform(shape=input_shape, minval=0., maxval=1.)
            differences = images_fake - images_real
            interpolates = images_real + alpha * differences
            gradients = tf.gradients(
                self.build_discriminator(interpolates, reuse=True, training=True),
                [interpolates, ])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            gradient_penalty = \
                lambda_gp * tf.reduce_mean((slopes / gamma_gp - 1.) ** 2)
            d_loss += gradient_penalty

        if self.cfg.drift_loss:
            eps = self.cfg.eps_drift
            drift_loss = eps * tf.reduce_mean(tf.nn.l2_loss(d_real))
            d_loss += drift_loss

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        beta1 = self.cfg.beta1
        beta2 = self.cfg.beta2
        learning_rate = self.tf_placeholders['learning_rate']
        d_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
        g_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.ema_op = ema.apply(g_vars)
        self.ema_vars = {ema.average_name(v): v for v in g_vars}

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.d_train_op = d_solver.minimize(d_loss, var_list=d_vars,
                                                global_step=self.global_step)
            self.g_train_op = g_solver.minimize(g_loss, var_list=g_vars)
            self.d_loss, self.g_loss = d_loss, g_loss

    def train(self):
        """ Train the model. """
        batch_size = self.cfg.batch_size
        n_iters = self.cfg.n_iters
        n_critic = self.cfg.n_critic
        z_dim = self.cfg.z_dim
        learning_rate = self.cfg.learning_rate
        display_period = self.cfg.display_period
        save_period = self.cfg.save_period
        image_loader = self.image_loader
        transition = self.cfg.transition
        # paths for save directories
        save_tag = '{0:}x{0:}'.format(self.cfg.resolution)
        if transition:
            save_tag += '_transition'
        img_save_dir = os.path.join(self.cfg.image_save_dir, save_tag)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        save_dir = os.path.join(self.cfg.model_save_dir, save_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, 'model')

        with tf.device("/cpu:0"):
            image_batch = image_loader.create_batch_pipeline()

        self.make_train_op(image_batch)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(self.cfg.summary_dir, time.strftime('%Y%m%d_%H%M%S')))

        # Create ops in graph before Session is created
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            tf.train.start_queue_runners(sess)
            load_model = self.cfg.load_model
            if self.cfg.load_model:
                self.load(sess, saver, load_model)
            elif transition:
                vars_to_load = []
                all_vars = tf.trainable_variables()
                r = self.cfg.min_resolution
                while r < self.cfg.resolution:
                    var_scope = '{0:}x{0:}'.format(r)
                    vars_to_load += [v for v in all_vars if var_scope in v.name]
                    r *= 2
                saver_restore = tf.train.Saver(vars_to_load)
                tag = '{0:}x{0:}'.format(self.cfg.resolution // 2)
                print(tag)
                self.load(sess, saver_restore, tag=tag)

            alpha = self.cfg.fade_alpha
            global_step = 0
            sum_g_loss, sum_d_loss = 0., 0.
            # batch_gen = image_loader.batch_generator()

            for i in range(self.cfg.n_iters):
                batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
                feed_dict = {self.tf_placeholders['z']: batch_z,
                             self.tf_placeholders['learning_rate']: learning_rate,
                             self.tf_placeholders['alpha']: alpha}
                if global_step % display_period == 0:
                    _, global_step, d_loss, merged_res = \
                        sess.run([self.d_train_op, self.global_step, self.d_loss, merged],
                                 feed_dict=feed_dict)
                else:
                    _, global_step, d_loss = \
                        sess.run([self.d_train_op, self.global_step, self.d_loss],
                             feed_dict=feed_dict)

                g_loss = 0.
                if global_step % n_critic == 0:
                    _, _, g_loss = \
                        sess.run([self.g_train_op, self.ema_op, self.g_loss],
                                 feed_dict=feed_dict)
                sum_g_loss += g_loss
                sum_d_loss += d_loss
                if transition:
                    alpha_step = 1. / n_iters
                    alpha = min(1., self.cfg.fade_alpha+global_step*alpha_step)
                if global_step % display_period == 0:
                    writer.add_summary(merged_res, global_step)
                    print("After {} iterations".format(global_step),
                          "Discriminator loss : {:3.5f}  "
                          .format(sum_d_loss / display_period),
                          "Generator loss : {:3.5f}"
                          .format(sum_g_loss / display_period * n_critic))
                    sum_g_loss, sum_d_loss = 0., 0.
                    if transition:
                        print("Using alpha = ", alpha)
                if global_step % save_period == 0:
                    print("Saving model in {}".format(save_dir))
                    saver.save(sess, save_dir, global_step)
                    if self.cfg.save_images:
                        gen_images = self.generate_images(save_tag, alpha=alpha)
                        plt.figure(figsize=(10, 10))
                        grid = image_loader.grid_batch_images(gen_images)
                        filename = os.path.join(img_save_dir, str(global_step) + '.png')
                        plt.imsave(filename, grid)
            print("Saving model in {}".format(save_dir))
            saver.save(sess, save_dir, global_step)

    def generate_images(self, model, batch_z=None, alpha=0.):
        """Runs generator to generate images"""
        batch_size = 64  # self.cfg.batch_size
        z_dim = self.cfg.z_dim
        if batch_z is None:
            batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
        # saver = tf.train.Saver(self.ema_vars)
        saver = tf.train.Saver()
        feed_dict = {self.tf_placeholders['z']: batch_z,
                     self.tf_placeholders['alpha']: alpha}
        image_loader = self.image_loader
        gen = self.build_generator(training=False)

        with tf.Session() as sess:
            self.load(sess, saver, model)
            gen_images = sess.run(gen, feed_dict=feed_dict)
            gen_images = image_loader.postprocess_image(gen_images)
            return gen_images

    def load(self, sess, saver, tag=None):
        """ Load the trained model. """
        if tag is None:
            tag = '{0:}x{0:}'.format(self.cfg.input_shape[0])

        load_dir = os.path.join(self.cfg.model_save_dir, tag, 'model')
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(load_dir))
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        saver.restore(sess, checkpoint.model_checkpoint_path)
