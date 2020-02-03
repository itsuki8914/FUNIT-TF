import tensorflow as tf
import math
import numpy as np
def l1_loss(y, y_hat):
    return tf.reduce_mean(tf.abs(y - y_hat))


def l2_loss(y, y_hat):
    return tf.reduce_mean(tf.square(y - y_hat))


def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def discriminator_loss(real, fake, images):
    real_loss = tf.reduce_mean(tf.nn.relu(1 - real))
    fake_loss = tf.reduce_mean(tf.nn.relu(1 + fake))
    grad = tf.square(tf.gradients(tf.reduce_mean(real), [images])[0])
    gp = 10 * tf.reduce_mean(tf.reduce_sum(grad, axis=[1, 2, 3]))

    return real_loss + fake_loss + gp


def generator_loss(fake):
    fake_loss = -tf.reduce_mean(fake)

    return fake_loss




def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r
