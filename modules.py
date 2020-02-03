import tensorflow as tf

def conv2d_layer(
        inputs,
        filters,
        kernel_size,
        strides,
        padding='valid',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)
    bias = tf.get_variable(name + '_bias', filters, initializer=tf.constant_initializer(0.0))

    return conv_layer + bias

def deconv2d_layer(inputs,
        filters,
        kernel_size,
        strides,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    deconv_layer = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)
    bias = tf.get_variable(name + '_bias', filters, initializer=tf.constant_initializer(0.0))

    return deconv_layer + bias

def fully_connected_layer(inputs, units, activation=None, name=None):
    x = tf.layers.dense(inputs=inputs, units=units, activation=activation, name=name)
    bias = tf.get_variable(name + '_bias', units, initializer=tf.constant_initializer(0.0))
    return x + bias

def instance_norm_layer(
        inputs,
        epsilon=1e-06,
        activation_fn=None,
        name=None):
    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs,
        epsilon=epsilon,
        activation_fn=activation_fn)

    return instance_norm_layer

def adaIN(x, mu, var, epsilon=1e-8):
    # instance norm
    x -= tf.reduce_mean(x, axis=[1,2], keepdims=True)
    x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True) + epsilon)
    # style mod
    h = x * var + mu
    return h

def pad(x, padding):
    if padding>0:
        y = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')
    else:
        y = x
    return y

def activation(x):
    return tf.nn.leaky_relu(x)

def global_avg_pooling(inputs):
    y = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
    return y

def average_pooling2d(x, ps=3, scale=2):
    return tf.layers.average_pooling2d(x, pool_size=ps, strides=scale, padding='SAME')

def conv2dBlock(inputs, filters, kernel_size, strides, padding, norm='IN', name='conv'):
    h = pad(inputs, padding)
    h = conv2d_layer(h, filters, kernel_size, strides, name=name)

    if norm == 'IN':
        h = instance_norm_layer(h)
    else:
        h = h
    y = activation(h)
    return y

def deconv2dBlock(inputs, filters, kernel_size, strides, padding, name='deconv'):
    h = pad(inputs, padding)
    h = deconv2d_layer(h, filters, kernel_size, strides, name=name)
    h = instance_norm_layer(h)
    y = activation(h)
    return y

def resBlk(inputs, filters, kernel_size, strides, padding, norm='none', name='resblk'):
    h = conv2dBlock(inputs=inputs, filters=filters, kernel_size=3, strides=1, padding=1,
                name=name+'_1')
    y = conv2dBlock(inputs=h, filters=filters, kernel_size=3, strides=1, padding=1,
                name=name+'_2')
    return y + inputs

def AdaINResBlk(inputs, filters, mus, vars, kernel_size, strides, padding, norm='none', name='AdaINresblk'):
    h = pad(inputs, padding)
    h = conv2d_layer(h, filters, kernel_size, strides, name=name+'_1')
    h = adaIN(h,mus[0], vars[0])
    h = activation(h)
    h = pad(h, padding)
    h = conv2d_layer(h, filters, kernel_size, strides, name=name+'_2')
    h = adaIN(h, mus[1], vars[1])
    y = activation(h)
    return y + inputs

def actResBlk(inputs, filters, kernel_size, strides, padding, norm='none', name='actResblk'):
    h = activation(inputs)
    h = pad(inputs, padding)
    h = conv2d_layer(h, filters, kernel_size, strides, name=name+'_1')
    h = instance_norm_layer(h)
    h = activation(h)
    h = pad(h, padding)
    h = conv2d_layer(h, filters, kernel_size, strides, name=name+'_2')
    y = instance_norm_layer(h)
    if tf.shape(inputs)[3] != tf.shape(y)[3]:
        inputs = conv2d_layer(inputs, filters, 1, 1, name=name+'_shortcut')
    return y + inputs

def content_encoder(inputs, filters=64):
    with tf.variable_scope('content_encoder', reuse=tf.AUTO_REUSE) as scope:
        h = conv2dBlock(inputs=inputs, filters=filters, kernel_size=7, strides=1, padding=3,
                    name='convi')
        h = conv2dBlock(inputs=h, filters=filters*2, kernel_size=4, strides=2, padding=1,
                    name='ds1')
        h = conv2dBlock(inputs=h, filters=filters*4, kernel_size=4, strides=2, padding=1,
                    name='ds2')
        h = conv2dBlock(inputs=h, filters=filters*8, kernel_size=4, strides=2, padding=1,
                    name='ds3')
        h = resBlk(inputs=h, filters=filters*8, kernel_size=3, strides=1, padding=1,
                    name='resblk1')
        y = resBlk(inputs=h, filters=filters*8, kernel_size=3, strides=1, padding=1,
                    name='resblk2')
        return y

def class_encoder(inputs, filters=64, latent_dim=64):
    with tf.variable_scope('class_encoder', reuse=tf.AUTO_REUSE) as scope:
        h = conv2dBlock(inputs=inputs, filters=filters, kernel_size=7, strides=1, padding=3, norm=None,
                    name='convi')
        h = conv2dBlock(inputs=h, filters=filters*2, kernel_size=4, strides=2, padding=1, norm=None,
                    name='ds1')
        h = conv2dBlock(inputs=h, filters=filters*4, kernel_size=4, strides=2, padding=1, norm=None,
                    name='ds2')
        h = conv2dBlock(inputs=h, filters=filters*4, kernel_size=4, strides=2, padding=1, norm=None,
                    name='ds3')
        h = conv2dBlock(inputs=h, filters=filters*4, kernel_size=4, strides=2, padding=1, norm=None,
                    name='ds4')
        h = global_avg_pooling(h)
        y = conv2d_layer(h, filters=latent_dim, kernel_size=1, strides=1, padding='valid', name='convo')
        return y

def MLP(inputs, dims=256):
    with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE) as scope:
        h = fully_connected_layer(inputs, dims, name='fc1')
        h = activation(h)
        h = fully_connected_layer(h, dims, name='fc2')
        h = activation(h)

        mus = []
        vars = []
        for i in range(4):
            m = fully_connected_layer(h, dims * 2, name='fc3_mu_{}'.format(i+1))
            v = fully_connected_layer(h, dims * 2, name='fc3_var_{}'.format(i+1))
            m = tf.reshape(m, shape=[-1, 1, 1, dims * 2])
            v = tf.reshape(v, shape=[-1, 1, 1, dims * 2])

            mus.append(m)
            vars.append(v)

        return mus, vars


def decoder(inputs, mus, vars, filters=64):

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
        idx = 0
        h = AdaINResBlk(inputs=inputs, filters=filters*8, mus=mus[idx:idx+2], vars=vars[idx:idx+2],
                    kernel_size=3, strides=1, padding=1, name='adaINresblk1')
        idx += 2;
        h = AdaINResBlk(inputs=h, filters=filters*8, mus=mus[idx:idx+2], vars=vars[idx:idx+2],
                    kernel_size=3, strides=1, padding=1, name='adaINresblk2')

        h = deconv2dBlock(inputs=h, filters=filters*4, kernel_size=4, strides=2, padding=0,
                    name='us3')
        h = deconv2dBlock(inputs=h, filters=filters*2, kernel_size=4, strides=2, padding=0,
                    name='us2')
        h = deconv2dBlock(inputs=h, filters=filters, kernel_size=4, strides=2, padding=0,
                    name='us1')
        h = pad(h, 3)
        h = conv2d_layer(h, 3, 7, 1, name='convo')
        y = tf.nn.tanh(h)
        return y

def generator(content_img, class_img, filters=64):
    content_code = content_encoder(content_img, filters)
    class_code = class_encoder(class_img, filters)
    mus, vars = MLP(class_code, filters*4)
    y = decoder(content_code, mus, vars, filters)
    return y

def discriminator(inputs, class_onehot, num_classes, filters=32):
    class_onehot = tf.reshape(class_onehot, shape=[-1, 1, 1, num_classes])
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        h = conv2dBlock(inputs=inputs, filters=filters, kernel_size=7, strides=1, padding=3, norm=None,
                    name='convi')
        nf = filters
        for i in range(4):
            h = actResBlk(inputs=h, filters=nf, kernel_size=3, strides=1, padding=1,
                        name='actResblk{}'.format(2*i+1))
            h = actResBlk(inputs=h, filters=nf, kernel_size=3, strides=1, padding=1,
                        name='actResblk{}'.format(2*i+2))
            h = average_pooling2d(h)
            nf *= 2

        h = actResBlk(inputs=h, filters=nf, kernel_size=3, strides=1, padding=1,
                    name='actResblk{}'.format(9))
        h = actResBlk(inputs=h, filters=nf, kernel_size=3, strides=1, padding=1,
                    name='actResblk{}'.format(10))

        f = h
        h = activation(h)
        h = conv2d_layer(h, filters=num_classes, kernel_size=1, strides=1, padding='valid', name='convo')
        y = tf.reduce_sum(h * class_onehot, axis=-1, keepdims=True)

    return y, f
