import tensorflow as tf
# Reference : https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py


def js_loss(logits_real, logits_fake, smooth_factor=0.9):
    # discriminator loss for real/fake classification
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_real, labels=tf.ones_like(logits_real) * smooth_factor))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    d_loss = d_loss_real + d_loss_fake

    # generator loss for fooling discriminator
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fake, labels=tf.ones_like(logits_fake)))
    return d_loss, g_loss


def wgan_loss(d_real, d_fake):
    # Standard WGAN loss
    g_loss = -tf.reduce_mean(d_fake)
    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
    return d_loss, g_loss


