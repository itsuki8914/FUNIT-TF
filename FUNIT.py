import os
import tensorflow as tf
from modules import *
from utils import *
from datetime import datetime


class FUNIT(object):
    def __init__(self, img_size, num_classes, batch_size=16, rec_weight=0.1, feature_weight=1,
                 mode='train', log_dir='./log'):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_shape = [None, img_size, img_size, 3]
        self.label_shape = [None, num_classes]

        self.rec_weight = rec_weight
        self.feature_weight = feature_weight

        self.mode = mode

        if self.mode == 'train':
            self.build_model()
            self.optimizer_initializer()

        if self.mode == 'eval':
            self. build_model_eval()



        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir)
            self.generator_summaries, self.discriminator_summaries = self.summary()


    def build_model(self):
        self.content_image = tf.placeholder(tf.float32, shape=self.input_shape, name='content_image')
        self.class_image = tf.placeholder(tf.float32, shape=self.input_shape, name='class_image')
        self.content_label = tf.placeholder(tf.float32, shape=self.label_shape, name='content_label')
        self.class_label = tf.placeholder(tf.float32, shape=self.label_shape, name='class_label')

        self.content_code = content_encoder(self.content_image)
        self.class_style_code = class_encoder(self.class_image)
        self.content_style_code = class_encoder(self.content_image)
        self.class_style_mus, self.class_style_vars = MLP(self.class_style_code)
        self.content_style_mus, self.content_style_vars = MLP(self.content_style_code)
        self.fake_img = decoder(self.content_code, self.class_style_mus, self.class_style_vars)
        self.rec_img = decoder(self.content_code, self.content_style_mus, self.content_style_vars)


        real_logit, style_fm = discriminator(self.class_image, self.class_label, self.num_classes)
        fake_logit, fake_fm = discriminator(self.fake_img, self.class_label, self.num_classes)

        rec_logit, rec_fm = discriminator(self.rec_img, self.content_label, self.num_classes)
        _, content_fm = discriminator(self.content_image, self.content_label, self.num_classes)

        self.d_adv_loss = discriminator_loss(real_logit, fake_logit, self.class_image)
        self.g_adv_loss = (generator_loss(fake_logit) + generator_loss(rec_logit)) / 2

        self.g_rec_loss = self.rec_weight * (l1_loss(self.rec_img, self.content_image))

        self.content_fm = tf.reduce_mean(tf.reduce_mean(content_fm, axis=2), axis=1)
        self.rec_fm = tf.reduce_mean(tf.reduce_mean(rec_fm, axis=2), axis=1)
        self.fake_fm = tf.reduce_mean(tf.reduce_mean(fake_fm, axis=2), axis=1)
        self.style_fm = tf.reduce_mean(tf.reduce_mean(style_fm, axis=2), axis=1)

        self.g_feature_loss = self.feature_weight * (l1_loss(self.rec_fm, self.content_fm) + l1_loss(self.fake_fm, self.style_fm))

        self.g_loss = tf.reduce_mean(self.g_adv_loss) + tf.reduce_mean(self.g_rec_loss) + tf.reduce_mean(self.g_feature_loss)

        self.d_loss = tf.reduce_mean(self.d_adv_loss)

        trainable_variables = tf.trainable_variables()
        self.d_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.g_vars =  [var for var in trainable_variables if 'coder' in var.name or 'MLP' in var.name]

    def optimizer_initializer(self):
        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name='discriminator_learning_rate')
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate,
                                                              beta1=0.5).minimize(self.d_loss,
                                                                                  var_list=self.d_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate,
                                                          beta1=0.5).minimize(self.g_loss,
                                                                              var_list=self.g_vars)

    def train(self, content_image, class_image, content_label, class_label, generator_learning_rate, discriminator_learning_rate):

        _, d_loss, summary_str = self.sess.run([self.discriminator_optimizer, self.d_loss, self.discriminator_summaries], feed_dict={
            self.content_image:content_image, self.class_image:class_image, self.content_label:content_label,
            self.class_label:class_label, self.discriminator_learning_rate:discriminator_learning_rate
        })
        self.writer.add_summary(summary_str, self.train_step)

        _, g_loss, summary_str = self.sess.run([self.generator_optimizer, self.g_loss, self.generator_summaries], feed_dict={
            self.content_image:content_image, self.class_image:class_image, self.content_label:content_label,
            self.class_label:class_label, self.generator_learning_rate:generator_learning_rate
        })
        self.writer.add_summary(summary_str, self.train_step)

        self.train_step += 1

        return g_loss, d_loss

    def test(self, content_image, class_image):

        generation = self.sess.run(self.fake_img, feed_dict={
            self.content_image:content_image, self.class_image:class_image
        })

        return generation

    def build_model_eval(self):
        self.content_image = tf.placeholder(tf.float32, shape=self.input_shape, name='content_image')
        self.class_image = tf.placeholder(tf.float32, shape=self.input_shape, name='class_image')
        self.content_code = content_encoder(self.content_image)
        self.class_style_code = class_encoder(self.class_image)
        self.class_style_code = tf.reduce_mean(self.class_style_code, axis=0, keepdims=True)
        self.class_style_mus, self.class_style_vars = MLP(self.class_style_code)
        self.fake_img_eval= decoder(self.content_code, self.class_style_mus, self.class_style_vars)

    def eval(self,content_image, class_image):
        generation = self.sess.run(self.fake_img_eval, feed_dict={
            self.content_image:content_image, self.class_image:class_image
        })
        return generation

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):
        with tf.name_scope('generator_summaries'):
            feature_loss_summary = tf.summary.scalar('feature_loss', self.g_feature_loss)
            rec_loss_summary = tf.summary.scalar('recon_loss', self.g_rec_loss)
            adv_loss_summary = tf.summary.scalar('adversarial_loss', self.g_adv_loss)
            g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)

            generator_summaries = tf.summary.merge(
                [feature_loss_summary, rec_loss_summary, adv_loss_summary,
                    g_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            d_loss = tf.summary.scalar('d_loss', self.d_loss)

            discriminator_summaries = tf.summary.merge(
                [d_loss])

        return generator_summaries, discriminator_summaries
