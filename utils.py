from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
from glob import glob
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        imgs = glob(cfg.data_dir + "/*.jpg") + \
               glob(cfg.data_dir + "/*.png") + \
               glob(cfg.data_dir + "/*.jpeg") + \
               glob(cfg.data_dir + "/*.bmp")

        self.images = np.array(imgs)
        self.train_idx, self.val_idx = None, None
        self.train_test_split()
        if self.cfg.preprocess == 'min-max':
            self.img_mean = self.img_stddev = 127.5
        else:
            self.img_mean = self.cfg.image_mean
            self.img_stddev = self.cfg.image_stddev

    def train_test_split(self):
        # build validation set

        val_idx = range(0, len(self.images), 10)
        train_idx = [i for i, _ in enumerate(self.images) if i not in val_idx]
        self.train_idx = np.array(train_idx)
        self.val_idx = np.array(val_idx)
        print("Size of training set : ", self.train_idx.size)
        print("Size of validation set : ", self.val_idx.size)

    def preprocess_image(self, img):
        image = np.copy(img)
        if self.cfg.train:
            new_img = self.random_crop(image)
            if self.cfg.flip:
                new_img = self.random_flip(new_img)
            if self.cfg.rotate:
                new_img = self.random_rotate(new_img)
            return (new_img - self.img_mean) / self.img_stddev
        else:
            # Pick predefined crops in testing mode
            new_images = self.test_crop(image)
            return (new_images - self.img_mean) / self.img_stddev

    def postprocess_image(self, imgs):
        new_imgs = imgs * self.img_stddev + self.img_mean
        new_imgs[new_imgs < 0] = 0
        new_imgs[new_imgs > 255] = 255
        return new_imgs

    def random_crop(self, img):
        """
        Applies random crops.
        Final image size given by self.cfg.input_shape
        """
        img_h, img_w, _ = img.shape
        new_h, new_w, _ = self.cfg.input_shape
        img = np.pad(img, [(0, max(0, new_h - img_h)), (0, max(0, new_w - img_w)), (0,0)], mode='mean')
        top = np.random.randint(0, max(0, img_h - new_h)+1)
        left = np.random.randint(0, max(0, img_w - new_w)+1)
        new_img = img[top:top + new_h, left:left + new_w, :]
        return new_img

    def random_flip(self, img):
        """Random horizontal and vertical flips"""
        new_img = np.copy(img)
        if np.random.uniform() > 0.5:
            new_img = cv2.flip(new_img, 1)
        if np.random.uniform() > 0.5:
            new_img = cv2.flip(new_img, 0)
        return new_img

    def random_rotate(self, img):
        """Random rotations by 0, 90, 180, 360 degrees"""
        theta = np.random.choice([0, 90, 180, 360])
        if theta == 0:
            return img
        h, w, _ = img.shape
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1)
        return cv2.warpAffine(img, mat, (w, h))

    def test_crop(self, img):
        new_images = []
        h, w, _ = self.cfg.input_shape
        for y, x in self.cfg.test_crops:
            new_img = img[y:y + h, x:x + w, :]
            new_images.append(new_img)
        return np.array(new_images)

    def load_batch(self, idx):
        """Loads batch of images and labels
        Arguments:
            idx: List of indices
        Returns:
            (images, labels): images and labels corresponding to indices
        """
        batch_imgs = []
        for index in idx:
            img_file = self.images[index]
            img = plt.imread(img_file)[:,:,:3]  # For png, which have 4 channels
            img = self.preprocess_image(img)
            batch_imgs.append(img)
        return np.array(batch_imgs)

    def batch_generator(self):
        batch_size = self.cfg.batch_size
        for _ in range(self.cfg.n_iters):
            indices = np.random.randint(len(self.train_idx), size=batch_size)
            batch_idx = self.train_idx[indices]
            batch_imgs = self.load_batch(batch_idx)
            yield batch_imgs

    def create_batch_pipeline(self):
        images_names_tensor = tf.convert_to_tensor(self.images, dtype=tf.string)
        single_image_name, = tf.train.slice_input_producer([images_names_tensor], shuffle=True, capacity=128)
        single_image_content = tf.read_file(single_image_name)
        single_image = tf.image.decode_image(single_image_content, channels=3)
        single_image.set_shape([None, None, 3])

        # Smart resize
        shp = tf.shape(single_image)
        r_size = shp[:2]
        dest_h = tf.random_uniform([1], 512, 1024, tf.int32)
        dest_h = tf.minimum(dest_h, r_size[0])
        ratio = tf.to_float(dest_h) / tf.to_float(r_size[0])
        n_size = tf.to_int32(tf.to_float(r_size) * ratio)
        single_image = tf.cast(tf.image.resize_images(single_image, n_size), np.uint8)

        # single_image = tf.image.random_brightness(single_image, .3)
        # single_image = tf.image.random_contrast(single_image, 0.9, 1.1)

        nH, nW = self.cfg.input_shape[:2]
        rH = tf.shape(single_image)[0]
        rW = tf.shape(single_image)[1]
        dH = tf.maximum(nH, rH) - rH
        dW = tf.maximum(nW, rW) - rW

        n = int(single_image.shape[-1])
        single_image = tf.pad(single_image,
                tf.convert_to_tensor([[dH // 2, (dH + 1) // 2], [dW // 2, (dW + 1) // 2], [0, 0]]))
        single_image = tf.random_crop(single_image, [nH, nW, n], seed=123)
        single_image.set_shape([nH, nW, n])

        angs = tf.to_float(tf.random_uniform([1], 0, 4, tf.int32)) * np.pi / 2
        single_image = tf.contrib.image.rotate(single_image, angs[0])
        single_image = tf.image.random_flip_left_right(single_image)

        single_image = (tf.to_float(single_image) - self.img_mean) / self.img_stddev

        image_batch = tf.train.batch(
            [single_image],
            batch_size=self.cfg.batch_size,
            num_threads=16,
            capacity=128)

        return image_batch

    def grid_batch_images(self, images):
        n, h, w, c = images.shape
        a = int(math.floor(np.sqrt(n)))
        # images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)
        images = images.astype(np.uint8)
        images_in_square = np.reshape(images[:a * a], (a, a, h, w, c))
        new_img = np.zeros((h * a, w * a, c), dtype=np.uint8)
        for col_i, col_images in enumerate(images_in_square):
            for row_i, image in enumerate(col_images):
                new_img[col_i * h: (1 + col_i) * h, row_i * w: (1 + row_i) * w] = image
        resolution = self.cfg.resolution
        if self.cfg.resolution != h:
            scale = resolution / h
            new_img = cv2.resize(new_img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
        return new_img
