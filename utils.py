from __future__ import print_function
from __future__ import division

import os
from glob import glob
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


class ImageLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        imgs, labels = [], []
        for cat in self.cfg.classes.keys():
            cat_dir = os.path.join(cfg.data_dir, cat)
            files = glob(cat_dir + '/*.tif')
            imgs += files
            labels += [cfg.classes[cat]] * len(files)
        self.images = np.array(imgs)
        self.labels = np.array(labels)
        self.train_idx, self.val_idx = None, None
        self.train_test_split()
        if self.cfg.preprocess == 'min-max':
            self.img_mean = self.img_stddev = 127.5
        else:
            self.img_mean = self.cfg.image_mean
            self.img_stddev = self.cfg.image_stddev

    def train_test_split(self):
        # build validation set
        val_idx = []
        for cat, file_ids in self.cfg.validation_set.items():
            for f in file_ids:
                filename = self.cfg.class_abr[cat] + str(f).zfill(3) + '.tif'
                filename = os.path.join(self.cfg.data_dir, cat, filename)
                val_idx.append(list(self.images).index(filename))
                # build train set
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
        top = np.random.randint(0, img_h - new_h)
        left = np.random.randint(0, img_w - new_w)
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
        batch_imgs, batch_labels = [], []
        for index in idx:
            img_file = self.images[index]
            img = plt.imread(img_file)
            label = self.labels[index]
            img = self.preprocess_image(img)
            batch_imgs.append(img)
            batch_labels.append(label)
        return np.array(batch_imgs), np.array(batch_labels)

    def batch_generator(self):
        batch_size = self.cfg.batch_size
        for _ in range(self.cfg.n_iters):
            indices = np.random.randint(len(self.train_idx), size=batch_size)
            batch_idx = self.train_idx[indices]
            batch_imgs, batch_labels = self.load_batch(batch_idx)
            yield batch_imgs, batch_labels

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
